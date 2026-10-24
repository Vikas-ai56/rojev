"""Camera-frame integrity during a run: are the images the policy receives real images?

A run can feed the policy corrupted frames without any error (2026-10-22: robosuite freed render contexts in the live
GL context; frames turned to noise or blank and every result of the run was meaningless). `FrameCheck` samples the
newest capture every `every_s` on its own thread and classifies each camera image with `frame_problem`; its `report`
goes into the run summary, and a run with any bad frame is not scored (`invariants.check`, `scenarios`).

`frame_problem` thresholds, measured 2026-10-22: real RoboCasa frames (963 across clean runs) have a mean horizontal
neighbour-pixel difference of at most 7.4 and a pixel standard deviation of at least 26.8; the corrupted run had noise
frames up to 159 and blank frames with standard deviation below 3. `NOISE` = 25 and `BLANK` = 3 sit well outside the
real range. A frozen camera (identical frames) is not detected: a still scene is legitimately identical.
"""
import threading
from collections.abc import Callable

import numpy as np

from rojev.protocol import Capture

NOISE = 25.0
BLANK = 3.0


def frame_problem(image: np.ndarray) -> tuple[str, float] | None:
    """("noise", neighbour difference) or ("blank", standard deviation) for a corrupted image, else None."""
    pixels = np.asarray(image, dtype=np.int16)
    spread = float(pixels.std())
    if spread < BLANK:
        return "blank", round(spread, 2)
    difference = float(np.abs(np.diff(pixels, axis=1)).mean())
    if difference > NOISE:
        return "noise", round(difference, 2)
    return None


class FrameCheck:
    def __init__(self, latest: Callable[[], Capture], cameras: list[str], every_s: float = 1.0):
        self.latest, self.cameras, self.every_s = latest, cameras, every_s
        self.checked = 0
        self.bad = {c: 0 for c in cameras}
        self.first_bad: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-check", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def check_once(self) -> None:
        capture = self.latest()
        self.checked += 1
        for camera in self.cameras:
            problem = frame_problem(capture.observation.images[camera])
            if problem is None:
                continue
            self.bad[camera] += 1
            if self.first_bad is None:
                self.first_bad = {"t": round(capture.t, 3), "camera": camera, "problem": problem[0],
                                  "value": problem[1]}

    def report(self) -> dict:
        return {"every_s": self.every_s, "checked": self.checked, "bad": self.bad,
                "valid": self.checked > 0 and not any(self.bad.values()), "first_bad": self.first_bad}

    def _run(self) -> None:
        self.check_once()
        while not self._stop.wait(self.every_s):
            self.check_once()
