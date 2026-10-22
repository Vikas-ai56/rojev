"""What an action should do to a robot's measured state: the robot plug-in the execution monitor needs.

An `EffectModel` names the measured quantities to track (`tracked`, from the observation state vector), what an
action commands of them (`command`), and the smallest change of each that is real (`resolution`: sensor resolution or
noise floor, in tracked units; it bounds the monitor's residual scale from below, so an axis that never moved during
calibration cannot turn measurement noise into a huge score). The monitor learns the linear map from commands to
measured change on nominal runs (`rojev.system1.monitor`), so a plug-in only states which numbers correspond, never
controller gains. Integrations write their own (RoboCasa PandaOmron in `executors/robocasa-pi052/embodiment.py`); `JointTargetEffect` covers every
`Robot` driven by joint targets.
"""
from abc import ABC, abstractmethod

import numpy as np


class EffectModel(ABC):
    @abstractmethod
    def tracked(self, state: np.ndarray) -> np.ndarray:
        """The measured quantities the monitor watches, from an observation state vector."""

    @abstractmethod
    def command(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """What `action`, applied in `state`, asks of the tracked quantities (any units: the monitor fits the map)."""

    @property
    @abstractmethod
    def resolution(self) -> np.ndarray | float:
        """Smallest real change per tracked quantity (> 0)."""


class JointTargetEffect(EffectModel):
    """Joint position targets: tracked = joint positions (the state vector), command = target − measured position;
    `resolution` = the encoders' resolution in the state's units."""

    def __init__(self, resolution: np.ndarray | float):
        self._resolution = resolution

    @property
    def resolution(self) -> np.ndarray | float:
        return self._resolution

    def tracked(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float)

    def command(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=float) - np.asarray(state, dtype=float)
