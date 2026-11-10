"""What the core needs to know about a robot to run it, whichever robot or simulator it is.

An `Embodiment` is the robot plug-in's half of a run: its `ActionSpec` (action dimensions, bounds, switches, control
rate), the camera names its observations carry, the command it starts from (the action that holds it still where it
is; None = the spec's hold value from the first observation), and optionally its `EffectModel` for the execution
monitor. The backend (the simulator or the hardware driver) is built by the robot plug-in itself, because it needs the
task, the scene and the seed; the core only sees it through `rojev.control.Backend`.
"""
from dataclasses import dataclass

import numpy as np

from rojev.robot import ActionSpec, EffectModel


@dataclass(frozen=True)
class Embodiment:
    name: str
    spec: ActionSpec
    cameras: tuple[str, ...]
    start: np.ndarray | None = None
    effect: EffectModel | None = None

    def __post_init__(self):
        if not self.cameras:
            raise ValueError(f"{self.name}: an embodiment needs at least one camera")
        if self.start is not None and np.shape(self.start) != (self.spec.size,):
            raise ValueError(f"{self.name}: start has shape {np.shape(self.start)}, the action spec has "
                             f"{self.spec.size} dimensions")
