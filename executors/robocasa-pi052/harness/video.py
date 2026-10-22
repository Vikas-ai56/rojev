"""Run video: the three robot cameras side by side with a text panel (time, active command, lane, latest decision).

During the run `add` only copies the camera images and the panel text into memory; composing and H.264 encoding happen
in `close`, after the run. Encoding while the robot runs is not an option: encoder start-up holds the Python GIL for
~0.2 s and stalls the control loop (measured: a 236 ms step, 3 starved ticks, a corrupted S1 latency sample). Memory:
3 × 256² × 3 bytes ≈ 0.6 MB per frame, about 6 MB per second of run at 10 fps.
"""
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np

PANEL_H, LINE_H = 70, 20


class VideoRecorder:
    def __init__(self, path: Path, fps: int):
        self.path, self.fps = path, fps
        self.frames: list[tuple[list[np.ndarray], list[str]]] = []

    def add(self, images: list[np.ndarray], lines: list[str]) -> None:
        self.frames.append(([im.copy() for im in images], lines))

    def close(self) -> None:
        if not self.frames:
            return
        writer = iio.imopen(self.path, "w", plugin="pyav")
        writer.init_video_stream("libx264", fps=self.fps)
        for images, lines in self.frames:
            writer.write_frame(self._compose(images, lines))
        writer.close()
        self.frames = []

    @staticmethod
    def _compose(images: list[np.ndarray], lines: list[str]) -> np.ndarray:
        row = np.concatenate(images, axis=1)
        panel = np.zeros((PANEL_H, row.shape[1], 3), dtype=np.uint8)
        for i, text in enumerate(lines[:3]):
            cv2.putText(panel, text[:110], (6, 16 + i * LINE_H), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1,
                        cv2.LINE_AA)
        frame = np.concatenate([row, panel], axis=0)
        return frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
