"""A run's video: every camera side by side with a text panel (time, active command, its lane, the latest decision, the
task memory).

During the run `add` only copies the images and the panel lines into memory; composing and H.264 encoding happen in
`close`, after the run. Encoding while the robot runs stalls the control loop (measured in the RoboCasa app: encoder
start-up held the GIL ~0.2 s, one 236 ms step, 3 starved ticks). Memory: ~0.4 MB per 360² camera image per frame.
"""
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

PANEL_LINES = 4
LINE_H = 16


class VideoRecorder:
    def __init__(self, path: Path, fps: int):
        self.path, self.fps = Path(path), fps
        self.frames: list[tuple[list[np.ndarray], list[str]]] = []

    def add(self, images: list[np.ndarray], lines: list[str]) -> None:
        self.frames.append(([np.array(im, copy=True) for im in images], list(lines)))

    def close(self) -> None:
        if not self.frames:
            return
        writer = iio.imopen(self.path, "w", plugin="pyav")
        writer.init_video_stream("libx264", fps=self.fps)
        for images, lines in self.frames:
            writer.write_frame(compose(images, lines))
        writer.close()
        self.frames = []


def compose(images: list[np.ndarray], lines: list[str]) -> np.ndarray:
    height = max(im.shape[0] for im in images)
    padded = [np.pad(im, ((0, height - im.shape[0]), (0, 0), (0, 0))) for im in images]
    row = np.concatenate(padded, axis=1)
    panel = Image.new("RGB", (row.shape[1], PANEL_LINES * LINE_H + 6), (0, 0, 0))
    draw = ImageDraw.Draw(panel)
    for i, text in enumerate(lines[:PANEL_LINES]):
        draw.text((6, 3 + i * LINE_H), text[:160], fill=(235, 235, 235))
    frame = np.concatenate([row, np.asarray(panel)], axis=0)
    return frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
