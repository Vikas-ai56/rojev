"""Wire format for a remote executor: lossless, numpy-only, no pickle.

Request  = compressed .npz: `img/<camera>` (uint8 H×W×3), `state` (float32), `task` and `subtask` (unicode scalars).
Response = compressed .npz: `actions` (float32 n×d), `timings_ms` (JSON string).
"""
import io
import json

import numpy as np

from .base import ActionChunk, Observation

IMG = "img/"


def encode_request(obs: Observation, subtask: str) -> bytes:
    arrays = {IMG + cam: np.ascontiguousarray(img, dtype=np.uint8) for cam, img in obs.images.items()}
    arrays.update(state=np.asarray(obs.state, dtype=np.float32), task=np.array(obs.task), subtask=np.array(subtask))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def decode_request(data: bytes) -> tuple[Observation, str]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        images = {k[len(IMG):]: z[k] for k in z.files if k.startswith(IMG)}
        return Observation(images, z["state"], str(z["task"])), str(z["subtask"])


def encode_response(chunk: ActionChunk) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, actions=np.asarray(chunk.actions, dtype=np.float32),
                        timings_ms=np.array(json.dumps(chunk.timings_ms)))
    return buf.getvalue()


def decode_response(data: bytes) -> ActionChunk:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return ActionChunk(z["actions"], json.loads(str(z["timings_ms"])))
