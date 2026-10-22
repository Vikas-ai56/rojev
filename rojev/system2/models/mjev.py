"""MJev-Spatial evidence over one kept connection (`rojev.transport`; server: `services/mjev/serve_mjev.py`). Images
are sent losslessly as PNG. An `/ask` is a pure function of its images and questions, so a transport retry is safe."""
import base64
import io
import json
import time

import numpy as np
from PIL import Image

from rojev.transport import open_transport


def png_b64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class RemoteMJev:
    def __init__(self, url: str, token: str, timeout_s: float = 120.0, cert_pem: str | None = None):
        self.transport = open_transport(url, token, cert_pem, timeout_s)

    def health(self) -> str:
        return self.transport.health()

    def ask(self, images: list[np.ndarray], questions: list[dict]) -> dict:
        """`questions`: [{"id", "question", "answer_space"}]; every question sees every image.

        Returns {"answers": {id: {...}}, "timings_ms": {"mjev": server ms, "network": …, "round_trip": …}}.
        """
        body = json.dumps({"images": [png_b64(i) for i in images], "questions": questions}).encode()
        t0 = time.perf_counter()
        out = json.loads(self.transport.request("POST", "/ask", body))
        rtt = (time.perf_counter() - t0) * 1000
        out["timings_ms"].update(network=round(rtt - out["timings_ms"]["mjev"], 1), round_trip=round(rtt, 1),
                                 request_kb=round(len(body) / 1024, 1))
        return out

    def close(self) -> None:
        self.transport.close()
