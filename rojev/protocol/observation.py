"""What every layer sees of the world: one observation, and an observation stamped with its robot-clock time."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Observation:
    """`images`: camera name → H×W×3 uint8 as the robot produced it; `state`: proprioception (robot-specific layout);
    `task`: the instruction the observation is interpreted under."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    task: str


@dataclass(frozen=True)
class Capture:
    """An observation and the robot-clock time it describes; `obs_id` increases by one per control tick."""

    obs_id: int
    t: float
    observation: Observation
