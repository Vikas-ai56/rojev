"""Lossless image encoding shared by the S2 model clients: camera images out as base64 PNG, masks back as base64
1-bit PNG."""
import base64
import io

import numpy as np
from PIL import Image


def png_b64(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_from_b64(data: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data)))).astype(bool)
