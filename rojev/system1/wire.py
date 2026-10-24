"""Wire format for a remote executor: numpy-only, no pickle. Version `VERSION` (3); both sides refuse another one.

Request  = compressed .npz: `version`; per camera either `img/<camera>` (uint8 H×W×3, lossless, codec "raw") or
           `webp/<camera>` (WebP bytes, codec "webp90"); `state` (float32); `task` and `subtask` (unicode scalars);
           `samples` K ≥ 1 (one observation, K independently sampled chunks in one request); optionally the
           continuation (`policy.Continuation`): `prefix` (float32 n×d, the actions already queued from the
           observation's time) and `delay_steps`.
Response = compressed .npz: `version`; `actions` (float32 K×n×d); `timings_ms` and `info` (JSON lists, one entry per
           sample).

Codecs: "raw" (lossless, ~150 KB for three 256² cameras) and "webp90" (WebP quality 90, ~26 KB, ~3 ms encode per
camera). Measured on π0.5² with real RoboCasa frames: actions from WebP-90 frames differ from lossless ones by 0.078
mean (actions 0–5), against 0.070 between two lossless calls (the policy's own sampling noise), so the codec is
within noise; it cuts the request ~6× (fewer packets per request, fewer loss episodes on a long link). Hence
`DEFAULT_CODEC` = "webp90" (decision 2026-10-21); every encoder default reads it from here.
"""
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from rojev.protocol import Observation

from .policy import ActionChunk, Continuation

VERSION = 3
IMG, WEBP = "img/", "webp/"
CODECS = ("raw", "webp90")
DEFAULT_CODEC = "webp90"


@dataclass(frozen=True)
class ActRequest:
    """What one /act request asks for: `samples` chunks for `observation`, conditioned on `subtask`, continuing
    `continuation` when given."""

    observation: Observation
    subtask: str
    samples: int = 1
    continuation: Continuation | None = None


def encode_request(obs: Observation, subtask: str, samples: int = 1, codec: str = DEFAULT_CODEC,
                   continuation: Continuation | None = None) -> bytes:
    if codec not in CODECS:
        raise ValueError(f"codec {codec!r} not in {CODECS}")
    if samples < 1:
        raise ValueError(f"samples must be at least 1, got {samples}")
    arrays = {}
    for cam, img in obs.images.items():
        pixels = np.ascontiguousarray(img, dtype=np.uint8)
        if codec == "raw":
            arrays[IMG + cam] = pixels
        else:
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="WEBP", quality=90, method=4)
            arrays[WEBP + cam] = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    arrays.update(version=np.array(VERSION), state=np.asarray(obs.state, dtype=np.float32), task=np.array(obs.task),
                  subtask=np.array(subtask), samples=np.array(samples))
    if continuation is not None:
        arrays.update(prefix=np.asarray(continuation.actions, dtype=np.float32),
                      delay_steps=np.array(continuation.delay_steps))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def decode_request(data: bytes) -> ActRequest:
    """ValueError on another wire version or fewer than one sample."""
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        _check_version(z)
        samples = int(z["samples"])
        if samples < 1:
            raise ValueError(f"samples must be at least 1, got {samples}")
        images = {k[len(IMG):]: z[k] for k in z.files if k.startswith(IMG)}
        for k in z.files:
            if k.startswith(WEBP):
                images[k[len(WEBP):]] = np.asarray(Image.open(io.BytesIO(z[k].tobytes())).convert("RGB"))
        continuation = (Continuation(z["prefix"], int(z["delay_steps"])) if "prefix" in z.files else None)
        return ActRequest(Observation(images, z["state"], str(z["task"])), str(z["subtask"]), samples, continuation)


def encode_response(chunks: Sequence[ActionChunk]) -> bytes:
    """One request's samples; their action arrays must share one shape."""
    buf = io.BytesIO()
    np.savez_compressed(buf, version=np.array(VERSION),
                        actions=np.stack([np.asarray(c.actions, dtype=np.float32) for c in chunks]),
                        timings_ms=np.array(json.dumps([c.timings_ms for c in chunks])),
                        info=np.array(json.dumps([c.info for c in chunks])))
    return buf.getvalue()


def decode_response(data: bytes) -> list[ActionChunk]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        _check_version(z)
        timings, info = json.loads(str(z["timings_ms"])), json.loads(str(z["info"]))
        return [ActionChunk(a, t, i) for a, t, i in zip(z["actions"], timings, info, strict=True)]


def _check_version(z) -> None:
    version = int(z["version"]) if "version" in z.files else 1
    if version != VERSION:
        raise ValueError(f"wire version {version}, expected {VERSION}")
