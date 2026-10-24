"""The server half of a remote S1 (`RemotePolicy` is the client): any `Policy` served over the wire.

`PolicyService.handle(method, path, headers, body) -> (status, body, type)` routes GET /health, POST /reset and POST
/act (`wire` request → `Policy.act_samples` → `wire` response); it is what `rojev.transport.Endpoint` wraps (token,
body limit and the replay cache for retried request ids are the Endpoint's). One lock serializes the model; a
continuation reaches the policy only if it `accepts_continuation`. `serve_policy` runs it on `HttpServer` (127.0.0.1)
and optionally `H3Server`, and stops after `idle_min` minutes without a request. Model-specific code (loading,
inputs, outputs) stays in each executor's `Policy`.
"""
import json
import threading
import time
from pathlib import Path

from rojev.transport import Endpoint, H3Server, HttpServer

from . import wire
from .policy import Policy


class PolicyService:
    """`about` is what /health reports besides the request count; `last` feeds the idle watchdog."""

    def __init__(self, policy: Policy, about: dict, name: str = "serve"):
        self.policy, self.about, self.name = policy, about, name
        self.last = {"t": time.time(), "requests": 0}
        self.lock = threading.Lock()

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes, str]:
        if method == "GET" and path == "/health":
            return 200, json.dumps({**self.about, "requests": self.last["requests"]}).encode(), "application/json"
        if method != "POST" or path not in ("/reset", "/act"):
            return 404, b"not found", "text/plain"
        if path == "/act":
            try:
                request = wire.decode_request(body)
            except ValueError as e:
                return 400, str(e).encode(), "text/plain"
        with self.lock:
            self.last["t"], self.last["requests"] = time.time(), self.last["requests"] + 1
            if path == "/reset":
                self.policy.reset()
                return 200, b"ok", "text/plain"
            continuation = request.continuation if self.policy.accepts_continuation else None
            chunks = self.policy.act_samples(request.observation, request.subtask, request.samples,
                                             *((None, continuation) if continuation is not None else ()))
            print(f"[{self.name}] act #{self.last['requests']} subtask={request.subtask!r} k={request.samples} "
                  f"{chunks[0].timings_ms}", flush=True)
            return 200, wire.encode_response(chunks), "application/octet-stream"


def serve_policy(endpoint: Endpoint, service: PolicyService, port: int, idle_min: float,
                 h3: tuple[int, Path, Path] | None = None) -> None:
    """Serve until `idle_min` minutes pass without a request (frees the GPU)."""
    httpd = HttpServer(endpoint, "127.0.0.1", port)
    if h3 is not None:
        h3_port, cert, key = h3
        H3Server(endpoint, "0.0.0.0", h3_port, cert, key).start()
        print(f"[{service.name}] HTTP/3 on udp 0.0.0.0:{h3_port} (pinned certificate {cert})", flush=True)

    def watchdog():
        while time.time() - service.last["t"] < idle_min * 60:
            time.sleep(30)
        print(f"[{service.name}] idle for {idle_min} min, shutting down", flush=True)
        httpd.shutdown()

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[{service.name}] ready on 127.0.0.1:{port}", flush=True)
    httpd.serve_forever()
