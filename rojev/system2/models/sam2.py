"""SAM 2.1 object masks over one kept connection (`rojev.transport`; server: `services/sam2/serve_sam2.py`). A
`/segment` is a pure function of its images and prompts, so a transport retry is safe."""
import json
import time
from dataclasses import dataclass

import numpy as np

from rojev.transport import open_transport

from .images import mask_from_b64, png_b64


@dataclass(frozen=True)
class PointPrompt:
    """One object in one image: pixel points (x, y) with labels 1 = on the object, 0 = not on it."""

    id: str
    image: int
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Mask:
    mask: np.ndarray
    score: float


@dataclass(frozen=True)
class Track:
    """A streaming session's masks after one frame; `Mask.score` is P(object visible) here."""

    session: str
    masks: dict[str, Mask]


def _wire(prompts: list[PointPrompt]) -> list[dict]:
    return [{"id": p.id, "image": p.image, "points": [list(xy) for xy in p.points], "labels": list(p.labels)}
            for p in prompts]


class RemoteSam2:
    def __init__(self, url: str, token: str, timeout_s: float = 60.0, cert_pem: str | None = None):
        self.transport = open_transport(url, token, cert_pem, timeout_s)
        self.last_timings_ms: dict = {}

    def health(self) -> str:
        return self.transport.health()

    def segment(self, images: list[np.ndarray], prompts: list[PointPrompt]) -> dict[str, Mask]:
        """The mask of each prompted object (image size), with SAM's predicted IoU as its score."""
        out = self._post("/segment", {"images": [png_b64(i) for i in images], "prompts": _wire(prompts)})
        return _masks(out)

    def track_start(self, images: list[np.ndarray], prompts: list[PointPrompt]) -> Track:
        """Open a streaming session (one per camera on the server); objects are prompted on this first frame."""
        out = self._post("/track/start", {"images": [png_b64(i) for i in images], "prompts": _wire(prompts)})
        return Track(out["session"], _masks(out))

    def track_step(self, session: str, frame: int, images: list[np.ndarray]) -> Track:
        """Propagate the session's objects to frame `frame` (1, 2, … after the start frame 0; cameras in the start
        order). The server answers a repeated frame number from its stored reply, so a transport retry never advances
        the session twice."""
        return Track(session, _masks(self._post("/track/step", {"session": session, "frame": frame,
                                                                "images": [png_b64(i) for i in images]})))

    def track_end(self, session: str) -> None:
        self._post("/track/end", {"session": session})

    def _post(self, path: str, body: dict) -> dict:
        t0 = time.perf_counter()
        out = json.loads(self.transport.request("POST", path, json.dumps(body).encode()))
        self.last_timings_ms = {**out.get("timings_ms", {}), "round_trip": round((time.perf_counter() - t0) * 1000, 1)}
        return out

    def close(self) -> None:
        self.transport.close()


def _masks(out: dict) -> dict[str, Mask]:
    return {k: Mask(mask_from_b64(v["mask"]), v["score"]) for k, v in out["masks"].items()}
