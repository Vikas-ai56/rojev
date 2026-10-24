"""What a person watching the run would see, written to disk while it goes: `<dir>/now.png` (the cameras side by side)
and `<dir>/status.json` (simulated time, the task in force, what the robot is doing and holding, what it last said,
whether the task's own check holds now and since when). Both are replaced atomically every `every_s` simulated seconds,
so a live user (a person or an agent typing into `realtime.py --input-file`) reads a whole frame, never half of one.
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


class LiveView:
    def __init__(self, folder: Path, every_s: float):
        self.folder, self.every_s = folder, every_s
        self.next_t = 0.0
        folder.mkdir(parents=True, exist_ok=True)

    def due(self, t: float) -> bool:
        return t >= self.next_t

    def write(self, t: float, images: list[np.ndarray], status: dict) -> None:
        self.next_t = t + self.every_s
        frame = self.folder / "now.tmp.png"
        Image.fromarray(np.concatenate(images, axis=1)).save(frame)
        os.replace(frame, self.folder / "now.png")
        state = self.folder / "status.tmp.json"
        state.write_text(json.dumps({"t": round(t, 2), **status}, indent=1, default=str))
        os.replace(state, self.folder / "status.json")
