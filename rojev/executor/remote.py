"""Executor running on a remote GPU (e.g. Kaggle behind a tunnel), reached over HTTPS with a bearer token.

Server side: `executors/robocasa-pi052/serve.py` (POST /act, POST /reset, GET /health).
"""
import time
import urllib.request

from . import wire
from .base import ActionChunk, Executor, Observation


class RemoteExecutor(Executor):
    def __init__(self, url: str, token: str, timeout_s: float = 120.0):
        self.url, self.token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def _post(self, path: str, body: bytes = b"") -> bytes:
        req = urllib.request.Request(self.url + path, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return r.read()

    def health(self) -> str:
        req = urllib.request.Request(self.url + "/health", headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        body = wire.encode_request(obs, subtask)
        t0 = time.perf_counter()
        chunk = wire.decode_response(self._post("/act", body))
        rtt = (time.perf_counter() - t0) * 1000
        server = sum(chunk.timings_ms.values())
        return ActionChunk(chunk.actions, {**chunk.timings_ms, "network": round(rtt - server, 1),
                                           "round_trip": round(rtt, 1), "request_kb": round(len(body) / 1024, 1)})

    def reset(self) -> None:
        self._post("/reset")
