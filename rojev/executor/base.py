"""Executor contract: a frozen low-level policy that turns (observation, subtask) into a chunk of actions.

Hardware-shaped: a real robot, a simulator and a remote GPU service all sit behind the same interface. The supervisor
only ever sees `Observation` in and `ActionChunk` out; the policy's own pre/post-processing stays behind `act`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Observation:
    images: dict[str, np.ndarray]  # camera name -> H×W×3 uint8, as the robot/sim produced it
    state: np.ndarray  # proprioception, float32 (robot-specific layout)
    task: str  # the user's instruction for the episode


@dataclass(frozen=True)
class ActionChunk:
    actions: np.ndarray  # (n, action_dim) float32, executed in order, one per control step
    timings_ms: dict[str, float] = field(default_factory=dict)  # per stage: e.g. preprocess, infer, network


class Executor(ABC):
    @abstractmethod
    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        """Actions for the next chunk, conditioned on `subtask` (the text the policy was trained to follow)."""

    @abstractmethod
    def reset(self) -> None:
        """Clear policy state between episodes."""
