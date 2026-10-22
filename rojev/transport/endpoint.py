"""The server side every service shares on every transport: admission before the body, at-most-once execution after.

A service writes only its routing handler, `handler(method, path, headers, body) -> (status, body, content_type)`
(headers lower-cased). `Endpoint(handler, token, max_body)` wraps it; `HttpServer` and `H3Server` call

  admit(headers, length)              once the headers are in, before any body byte is buffered: a wrong bearer token
                                      → 401, a declared length above `max_body` → 413; the server then never buffers
                                      the body (HTTP/1.1 discards a small one so the connection stays usable, closes on
                                      a large one; HTTP/3 answers and stops the stream)
  handle(method, path, headers, body) after the body: the handler, at most once per request id; an exception → 500

Replay safety. A client retries a request once when the connection fails, and the failure can come after the service
already ran it (the reply was lost). Each logical request therefore carries a client-made id (`REQUEST_ID` header,
`new_request_id`) that its retry reuses; `ReplayCache` runs the handler for the first copy only and answers every later
copy with the stored response (a copy arriving while the first still runs waits for it). An id reused for a different
request (method, path or body differ) → 409. Requests without an id run every time.
"""
import hashlib
import hmac
import logging
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field

Response = tuple[int, bytes, str]
Handler = Callable[[str, str, dict[str, str], bytes], Response]

REQUEST_ID = "x-request-id"
UNAUTHORIZED: Response = (401, b"unauthorized", "text/plain")
TOO_LARGE: Response = (413, b"request body too large", "text/plain")
BAD_LENGTH: Response = (400, b"malformed content-length", "text/plain")
REUSED_ID: Response = (409, b"request id reused for a different request", "text/plain")
UNFINISHED: Response = (500, b"request did not complete", "text/plain")

log = logging.getLogger(__name__)


def new_request_id() -> str:
    """A fresh id for one logical request (128 random bits); its retry sends the same one."""
    return uuid.uuid4().hex


def parse_length(value: str | None) -> int | None:
    """A Content-Length header value: None when absent; ValueError unless a non-negative integer."""
    if value is None:
        return None
    length = int(value)
    if length < 0:
        raise ValueError(f"negative content-length {length}")
    return length


@dataclass
class _Entry:
    fingerprint: bytes
    done: threading.Event = field(default_factory=threading.Event)
    response: Response = UNFINISHED


class ReplayCache:
    """At-most-once execution per request id, bounded: keeps the last `capacity` completed responses (a retry follows
    its failed attempt at once, so it is found unless `capacity` other requests completed in between); entries still
    running are never evicted. Thread-safe. `replays` counts copies answered from the cache."""

    def __init__(self, capacity: int = 256):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self.replays = 0
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()

    def run(self, request_id: str, fingerprint: bytes, execute: Callable[[], Response]) -> Response:
        with self._lock:
            entry = self._entries.get(request_id)
            first = entry is None
            if first:
                entry = self._entries[request_id] = _Entry(fingerprint)
                self._evict()
        if not first:
            if entry.fingerprint != fingerprint:
                return REUSED_ID
            entry.done.wait()
            with self._lock:
                self.replays += 1
            return entry.response
        try:
            entry.response = execute()
        finally:
            entry.done.set()
        return entry.response

    def _evict(self) -> None:
        done = [key for key, e in self._entries.items() if e.done.is_set()]
        for key in done[:max(0, len(self._entries) - self.capacity)]:
            del self._entries[key]


class Endpoint:
    """A service handler behind bearer-token admission, a body limit and replay-safe execution (module docstring)."""

    def __init__(self, handler: Handler, token: str, max_body: int, replay_capacity: int = 256):
        self.handler, self.max_body = handler, max_body
        self._authorization = f"Bearer {token}".encode()
        self.replay = ReplayCache(replay_capacity)

    def admit(self, headers: dict[str, str], length: int | None) -> Response | None:
        """The refusal to send before reading the body, or None to read it. `length` = the declared body length."""
        if not hmac.compare_digest(headers.get("authorization", "").encode(), self._authorization):
            return UNAUTHORIZED
        if length is not None and length > self.max_body:
            return TOO_LARGE
        return None

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        request_id = headers.get(REQUEST_ID)
        if request_id is None:
            return self._execute(method, path, headers, body)
        fingerprint = hashlib.sha256(b"\0".join((method.encode(), path.encode(), body))).digest()
        return self.replay.run(request_id, fingerprint, lambda: self._execute(method, path, headers, body))

    def _execute(self, method: str, path: str, headers: dict[str, str], body: bytes) -> Response:
        try:
            return self.handler(method, path, headers, body)
        except Exception as e:
            log.exception("%s %s failed", method, path)
            return 500, f"{type(e).__name__}: {e}".encode(), "text/plain"
