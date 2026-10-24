"""Observation history for a policy that conditions on past frames (executor side; the core passes one observation).

`FrameHistory` keeps the observations of the last `capacity` control ticks, fed by the control loop's tick hook (and
the starting observation). `HistoryPolicy` wraps a policy that wants `length` frames `interval` ticks apart ending at
the observation it is asked about: it finds that observation in the history by identity (the runner sends the very
object the loop captured; one the hook has not filed yet is the newest, so the window ends with it), takes the
earlier frames the way the model's own evaluation does (indices clamped at the episode start), and sends them as one
observation: camera `c`'s frame `k` (0 = oldest) as image `c@k`, the states stacked in the same order. `to_actions`
maps the returned chunk into the robot's action space.
"""
import collections
import dataclasses
import threading
from collections.abc import Callable

import numpy as np

from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy


class FrameHistory:
    def __init__(self, capacity: int):
        self._frames: collections.deque[Observation] = collections.deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, observation: Observation) -> None:
        with self._lock:
            self._frames.append(observation)

    def window(self, observation: Observation, length: int, interval: int) -> list[Observation]:
        """`length` observations `interval` ticks apart ending at `observation` (clamped at the oldest held)."""
        with self._lock:
            frames = list(self._frames)
        index = next((i for i in range(len(frames) - 1, -1, -1) if frames[i].images is observation.images), None)
        frames = frames[:index + 1] if index is not None else frames + [observation]
        last = len(frames) - 1
        return [frames[max(0, last - (length - 1 - k) * interval)] for k in range(length)]


class HistoryPolicy(Policy):
    def __init__(self, inner: Policy, history: FrameHistory, length: int, interval: int,
                 to_actions: Callable[[np.ndarray], np.ndarray]):
        self.inner, self.history, self.length, self.interval, self.to_actions = (inner, history, length, interval,
                                                                                  to_actions)

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        frames = self.history.window(obs, self.length, self.interval)
        images = {f"{camera}@{k}": f.images[camera] for k, f in enumerate(frames) for camera in obs.images}
        state = np.concatenate([np.asarray(f.state, dtype=np.float32) for f in frames])
        chunk = self.inner.act(Observation(images, state, obs.task), subtask, timeout_s)
        return dataclasses.replace(chunk, actions=self.to_actions(chunk.actions).astype(np.float32))

    def reset(self) -> None:
        self.inner.reset()
