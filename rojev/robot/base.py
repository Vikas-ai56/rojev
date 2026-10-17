"""The robot interface every backend implements: simulator now, the real SO-101 later.

Shape follows LeRobot's Robot (connect / observation / action) with ros2_control's read/write split. Time only advances
through the robot's `Clock`, so a control loop written against this interface runs unchanged on either backend.
"""
import time
from abc import ABC, abstractmethod

import numpy as np

from .limits import JointLimits, SafetyFilter
from .types import CommandResult, JointCommand, Observation


class Clock(ABC):
    @abstractmethod
    def now(self) -> float: ...

    @abstractmethod
    def wait_until(self, t: float) -> None:
        """Return once `now() >= t`. On hardware this sleeps; in simulation it advances physics."""


class MonotonicClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    def wait_until(self, t: float) -> None:
        time.sleep(max(0.0, t - time.monotonic()))


class Rate:
    """Fixed-rate loop timing on a robot clock: `sleep()` returns at the next tick and reports how late it started."""

    def __init__(self, clock: Clock, hz: float):
        self.clock, self.period = clock, 1.0 / hz
        self._next = clock.now()

    def sleep(self) -> float:
        self._next += self.period
        overrun = max(0.0, self.clock.now() - self._next)
        self.clock.wait_until(self._next)
        return overrun


class Robot(ABC):
    """Subclasses implement `_connect`, `_disconnect`, `_read` and `_write` in canonical units."""

    def __init__(self, limits: JointLimits, clock: Clock, control_rate_hz: float):
        self.limits, self.clock, self.control_rate_hz = limits, clock, control_rate_hz
        self.safety = SafetyFilter(limits, dt=1.0 / control_rate_hz)
        self._last: Observation | None = None
        self.connected = False

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.limits.names

    @property
    @abstractmethod
    def camera_names(self) -> tuple[str, ...]: ...

    def connect(self) -> "Robot":
        self._connect()
        self.connected = True
        return self

    def disconnect(self) -> None:
        if self.connected:
            self._disconnect()
            self.connected = False

    def __enter__(self) -> "Robot":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.disconnect()

    def read(self) -> Observation:
        self._require_connected()
        self._last = self._read()
        return self._last

    def write(self, cmd: JointCommand) -> CommandResult:
        """Send joint targets through the safety filter, against the latest `read()`."""
        self._require_connected()
        if self._last is None:
            raise RuntimeError("write() before any read(): the safety filter needs the measured joint positions")
        applied, violations = self.safety.apply(cmd, self._last.joints.position)
        self._write(applied)
        return CommandResult(cmd, applied, tuple(violations))

    def command(self, position: np.ndarray) -> JointCommand:
        return JointCommand(self.joint_names, np.asarray(position, dtype=float))

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError(f"{type(self).__name__} is not connected")

    @abstractmethod
    def _connect(self) -> None: ...

    @abstractmethod
    def _disconnect(self) -> None: ...

    @abstractmethod
    def _read(self) -> Observation: ...

    @abstractmethod
    def _write(self, cmd: JointCommand) -> None: ...
