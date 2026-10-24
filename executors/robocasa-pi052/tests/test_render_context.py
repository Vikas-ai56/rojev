"""Camera images stay real after a garbage collection frees the render context of a replaced simulator (robosuite
frees it in the live context otherwise: every later image is noise)."""
import gc

import numpy as np
import pytest

from realtime import START, RoboCasaBackend
from workflow import CAMS


@pytest.mark.slow
def test_camera_images_stay_real_after_a_garbage_collection():
    backend = RoboCasaBackend("CloseToasterOvenDoor", 107, horizon=200)
    gc.collect()
    for _ in range(5):
        obs = backend.step(START)
    backend.env.close()
    assert max(float(np.abs(np.diff(obs.images[c].astype(int), axis=1)).mean()) for c in CAMS) < 20
