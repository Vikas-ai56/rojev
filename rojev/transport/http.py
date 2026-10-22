"""HTTP/1.1 over one persistent TCP connection (keep-alive): the fallback transport, e.g. through an SSH tunnel.

A new connection per request costs extra round trips (set-up, then TCP slow start on a ~150 KB request); measured Mac →
us-east through an SSH tunnel: 680 ms of network per request against a 216 ms ping, 255 ms with one kept connection.

  HttpTransport(url, token)       client; a connection the server or the tunnel dropped (restart, or lost mid-request)
                                  is replaced and the request resent once with the same request id, so a service that
                                  already ran it answers from its replay cache instead of running it again; any other
                                  failure propagates. `request(..., timeout_s)` bounds the whole call, retry included:
                                  past it the connection (state unknown) is closed and `TimeoutError` raised
  HttpServer(endpoint, host, port) threaded keep-alive server for an `Endpoint`: admission before reading the body
"""
import http.client
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .endpoint import BAD_LENGTH, REQUEST_ID, Endpoint, Response, new_request_id, parse_length

STALE = (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError, http.client.CannotSendRequest)
LENGTH_REQUIRED: Response = (411, b"content-length required (no transfer-encoding)", "text/plain")
DISCARD_CHUNK = 64 * 1024


class HttpTransport:
    def __init__(self, url: str, token: str, timeout_s: float = 120.0):
        parts = urlsplit(url.rstrip("/"))
        self.scheme, self.host, self.port, self.base = parts.scheme, parts.hostname, parts.port, parts.path
        self.token, self.timeout_s = token, timeout_s
        self._conn: http.client.HTTPConnection | None = None

    def request(self, method: str, path: str, body: bytes = b"", timeout_s: float | None = None) -> bytes:
        request_id = new_request_id()
        deadline = time.monotonic() + (self.timeout_s if timeout_s is None else timeout_s)
        try:
            return self._exchange(method, path, body, request_id, deadline)
        except STALE:
            return self._exchange(method, path, body, request_id, deadline)

    def health(self) -> str:
        conn = self._connection(30.0)
        try:
            conn.request("GET", self.base + "/health", headers={"Authorization": f"Bearer {self.token}"})
            return conn.getresponse().read().decode()
        finally:
            conn.close()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _connection(self, timeout_s: float) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout_s)

    def _exchange(self, method: str, path: str, body: bytes, request_id: str, deadline: float) -> bytes:
        budget = deadline - time.monotonic()
        if budget <= 0:
            raise TimeoutError(f"{method} {path}: deadline passed before sending")
        if self._conn is None:
            self._conn = self._connection(budget)
        self._conn.timeout = budget
        if self._conn.sock is not None:
            self._conn.sock.settimeout(budget)
        try:
            self._conn.request(method, self.base + path, body=body,
                               headers={"Authorization": f"Bearer {self.token}", REQUEST_ID: request_id,
                                        "Content-Type": "application/octet-stream", "Content-Length": str(len(body))})
            response = self._conn.getresponse()
            data = response.read()
        except STALE:
            self.close()
            raise
        except TimeoutError as e:
            self.close()
            raise TimeoutError(f"{method} {path}: no answer within {budget:.2f} s") from e
        if response.status != 200:
            raise ConnectionError(f"{method} {path}: HTTP {response.status} {response.reason}")
        return data


class _Handler(BaseHTTPRequestHandler):
    """Reads a body only after the endpoint admits the headers. A refused request's body is discarded unread when it
    is within the limit (the client gets a clean status and keeps its connection), else the connection is closed."""

    protocol_version = "HTTP/1.1"
    server: "HttpServer"

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        self._answer("GET")

    def do_POST(self):
        self._answer("POST")

    def _answer(self, method: str) -> None:
        endpoint = self.server.endpoint
        headers = {k.lower(): v for k, v in self.headers.items()}
        if "transfer-encoding" in headers:
            return self._reply(LENGTH_REQUIRED, close=True)
        try:
            length = parse_length(headers.get("content-length")) or 0
        except ValueError:
            return self._reply(BAD_LENGTH, close=True)
        refusal = endpoint.admit(headers, length)
        if refusal is not None:
            within = length <= endpoint.max_body
            if within:
                self._discard(length)
            return self._reply(refusal, close=not within)
        body = self.rfile.read(length)
        if len(body) < length:
            self.close_connection = True
            return
        self._reply(endpoint.handle(method, self.path, headers, body))

    def _discard(self, length: int) -> None:
        while length > 0:
            data = self.rfile.read(min(length, DISCARD_CHUNK))
            if not data:
                return
            length -= len(data)

    def _reply(self, response: Response, close: bool = False) -> None:
        status, out, ctype = response
        try:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(out)))
            if close:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(out)
        except ConnectionError:
            close = True
        self.close_connection = self.close_connection or close


class HttpServer(ThreadingHTTPServer):
    """HTTP/1.1 keep-alive for `endpoint` on (host, port): a client keeps one connection across requests; threaded, so
    a health check never waits behind a held connection. `REQUEST_ID` is passed through to the endpoint."""

    daemon_threads = True

    def __init__(self, endpoint: Endpoint, host: str, port: int):
        self.endpoint = endpoint
        super().__init__((host, port), _Handler)

