"""A robot's action vector, described for the control layer: the plug-in point for any robot.

The execution runtime and guard never assume a particular arm. A robot is described by an `ActionSpec`: per action
dimension its name, bounds, how to hold still, whether it is a discrete switch (e.g. gripper open/close), and whether
changing it is irreversible (no undo in the physical world: gripper, contact). `joint_target_spec` builds one for any
`Robot` from its joint limits; other integrations write their own (RoboCasa PandaOmron in `executors/robocasa-pi052/`).
"""
from dataclasses import dataclass

import numpy as np

from .base import Robot

KEEP = "keep"  


@dataclass(frozen=True)
class ActionDim:
    name: str
    low: float
    high: float
    hold: float | str  
    threshold: float | None = None  
    irreversible: bool = False  


@dataclass(frozen=True)
class ActionSpec:
    robot: str
    dims: tuple[ActionDim, ...]
    control_hz: float

    def __post_init__(self):
        for d in self.dims:
            if not d.low < d.high:
                raise ValueError(f"{self.robot}.{d.name}: empty bounds [{d.low}, {d.high}]")
            if d.irreversible and d.threshold is None:
                raise ValueError(f"{self.robot}.{d.name}: irreversible dimensions must be discrete (threshold)")

    @property
    def size(self) -> int:
        return len(self.dims)

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def low(self) -> np.ndarray:
        return np.array([d.low for d in self.dims])

    @property
    def high(self) -> np.ndarray:
        return np.array([d.high for d in self.dims])

    @property
    def continuous(self) -> np.ndarray:
        return np.array([d.threshold is None for d in self.dims])

    def hold(self, last: np.ndarray | None) -> np.ndarray:
        """Action that keeps the robot still with every switch (gripper, mode) as it is.

        KEEP dimensions repeat the last applied value; without one there is no safe guess (a joint target or a
        gripper state could jump), so the runtime must be seeded with the robot's current command first."""
        if last is None and any(d.hold == KEEP for d in self.dims):
            raise ValueError(f"{self.robot}: no last action to hold; seed the runtime with the current command")
        return np.array([last[i] if d.hold == KEEP else d.hold for i, d in enumerate(self.dims)], dtype=float)

    def switches(self, a: np.ndarray) -> np.ndarray:
        """Discrete state per dimension (True = on); False for continuous dimensions."""
        return np.array([d.threshold is not None and a[i] >= d.threshold for i, d in enumerate(self.dims)])

    def irreversible_change(self, last: np.ndarray | None, a: np.ndarray) -> list[str]:
        """Names of irreversible switches that `a` would flip relative to `last`."""
        if last is None:
            return []
        flips = self.switches(last) != self.switches(a)
        return [d.name for i, d in enumerate(self.dims) if d.irreversible and flips[i]]

    def keep_switches(self, last: np.ndarray | None, a: np.ndarray, names: list[str]) -> np.ndarray:
        """`a` with the named dimensions reset to their last applied value."""
        out = a.copy()
        if last is not None:
            for i, d in enumerate(self.dims):
                if d.name in names:
                    out[i] = last[i]
        return out

    def limit(self, a: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Clip into bounds; report every clamp (a correct pipeline produces none)."""
        if a.shape != (self.size,):
            raise ValueError(f"{self.robot}: action shape {a.shape} != ({self.size},)")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{self.robot}: non-finite action {a}")
        out = np.clip(a, self.low, self.high)
        clamped = [f"{d.name}={a[i]:.4f} outside [{d.low}, {d.high}]" for i, d in enumerate(self.dims) if out[i] != a[i]]
        return out, clamped


def joint_target_spec(robot: Robot, switches: dict[str, float] | None = None,
                      irreversible: tuple[str, ...] = ()) -> ActionSpec:
    """One `KEEP` dimension per joint with the robot's position limits (actions are joint position targets).
    `switches` maps a joint (e.g. the gripper) to the target at which it counts as "on" (e.g. open); `irreversible`
    names the switches that cannot be undone."""
    switches = switches or {}
    lim = robot.limits
    dims = tuple(ActionDim(n, float(lim.lower[i]), float(lim.upper[i]), KEEP, threshold=switches.get(n),
                           irreversible=n in irreversible) for i, n in enumerate(robot.joint_names))
    return ActionSpec(type(robot).__name__, dims, control_hz=robot.control_rate_hz)
