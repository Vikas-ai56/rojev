"""MJev-Spatial evidence over HTTPS (server: `services/mjev/serve_mjev.py`). Images are sent losslessly as PNG."""
import base64
import io
import json
import time
import urllib.request

import numpy as np
from PIL import Image


def png_b64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class RemoteMJev:
    def __init__(self, url: str, token: str, timeout_s: float = 120.0):
        self.url, self.token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def health(self) -> str:
        req = urllib.request.Request(self.url + "/health", headers={"Authorization": f"Bearer {self.token}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    def ask(self, images: list[np.ndarray], questions: list[dict]) -> dict:
        """`questions`: [{"id", "question", "answer_space"}]; every question sees every image.

        Returns {"answers": {id: {...}}, "timings_ms": {"mjev": server ms, "network": …, "round_trip": …}}.
        """
        body = json.dumps({"images": [png_b64(i) for i in images], "questions": questions}).encode()
        req = urllib.request.Request(self.url + "/ask", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            out = json.loads(r.read())
        rtt = (time.perf_counter() - t0) * 1000
        out["timings_ms"].update(network=round(rtt - out["timings_ms"]["mjev"], 1), round_trip=round(rtt, 1),
                                 request_kb=round(len(body) / 1024, 1))
        return out
