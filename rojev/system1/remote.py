"""A System 1 policy running on a remote GPU, reached through `rojev.transport` (HTTP/3 or HTTP/1.1 keep-alive).

Server side: `executors/robocasa-pi052/serve.py` (POST /act, POST /reset, GET /health). The service description is
{url, token} plus, for an h3:// url, `cert_pem` (the server's pinned certificate); `codec` picks the frame encoding
(`wire.CODECS`; default `wire.DEFAULT_CODEC`). `act_samples(obs, subtask, k)` is one request carrying one observation
and K: the service draws the K chunks in one batch, so K samples cost one round trip, not K.
"""
import time

from rojev.protocol import Observation
from rojev.transport import open_transport

from . import wire
from .policy import ActionChunk, Policy


class RemotePolicy(Policy):
    def __init__(self, url: str, token: str, cert_pem: str | None = None, timeout_s: float = 120.0,
                 codec: str = wire.DEFAULT_CODEC):
        if codec not in wire.CODECS:
            raise ValueError(f"codec {codec!r} not in {wire.CODECS}")
        self.transport, self.codec = open_transport(url, token, cert_pem, timeout_s), codec

    def health(self) -> str:
        return self.transport.health()

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        return self.act_samples(obs, subtask, 1)[0]

    def act_samples(self, obs: Observation, subtask: str, k: int) -> list[ActionChunk]:
        """Per chunk: the service's stage times plus `network` = round trip − those stages, `round_trip` and
        `request_kb` of the one shared request."""
        body = wire.encode_request(obs, subtask, k, self.codec)
        t0 = time.perf_counter()
        chunks = wire.decode_response(self.transport.request("POST", "/act", body))
        rtt = (time.perf_counter() - t0) * 1000
        if len(chunks) != k:
            raise ValueError(f"asked the service for {k} samples, got {len(chunks)}")
        return [ActionChunk(c.actions, {**c.timings_ms, "network": round(rtt - sum(c.timings_ms.values()), 1),
                                        "round_trip": round(rtt, 1), "request_kb": round(len(body) / 1024, 1)},
                            c.info) for c in chunks]

    def reset(self) -> None:
        self.transport.request("POST", "/reset")
