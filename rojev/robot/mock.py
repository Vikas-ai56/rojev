"""A backend with no physics: commanded positions become measured positions on the next read (ros2_control's
mock-components behaviour). For contract tests and for exercising pipelines without a simulator."""
import numpy as np

from .base import Clock, Robot
from .limits import JointLimits
from .types import JointCommand, JointState, Observation, Stamp


class ManualClock(Clock):
    def __init__(self, t: float = 0.0):
        self.t = t

    def now(self) -> float:
        return self.t

    def wait_until(self, t: float) -> None:
        self.t = max(self.t, t)


class MockRobot(Robot):
    def __init__(self, limits: JointLimits, initial: np.ndarray | None = None, control_rate_hz: float = 30.0):
        super().__init__(limits, ManualClock(), control_rate_hz)
        self._q = np.array(initial if initial is not None else np.clip(0.0, limits.lower, limits.upper), dtype=float)
        self._seq = 0

    @property
    def camera_names(self) -> tuple[str, ...]:
        return ()

    def _connect(self) -> None:
        pass

    def _disconnect(self) -> None:
        pass

    def _read(self) -> Observation:
        t = self.clock.now()
        state = JointState(Stamp(t, t, self._seq), self.joint_names, self._q.copy(), None)
        self._seq += 1
        return Observation(state, {})

    def _write(self, cmd: JointCommand) -> None:
        self._q = cmd.position.copy()
