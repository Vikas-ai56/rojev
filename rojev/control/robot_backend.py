"""Any `rojev.robot.Robot` (the MuJoCo SO-101 now, hardware later) as a control-loop backend.

Actions are joint position targets in the robot's canonical units, one per joint, in `robot.joint_names` order. Every
target still passes the robot's own `SafetyFilter` (position range, velocity step) inside `Robot.write`, and the robot's
clock paces the control loop. Holding = repeating the last target (`KEEP`), which the safety filter keeps within one
velocity step of the measured position; the runtime is seeded with the measured joints (`current_command`).
Its `ActionSpec` comes from `rojev.robot.joint_target_spec`.
"""
import numpy as np

from rojev.protocol import Observation
from rojev.robot import CommandResult, Robot


class RobotBackend:
    """`observe` / `step` over `Robot.read` / `Robot.write`; `clock` is the robot's. `last_result` keeps the requested
    vs applied command of the latest step (safety-filter clamps included)."""

    def __init__(self, robot: Robot, task: str = ""):
        self.robot, self.task, self.clock = robot, task, robot.clock
        self.last_result: CommandResult | None = None
        self._obs = robot.read()

    def current_command(self) -> np.ndarray:
        return self._obs.joints.position.astype(float).copy()

    def observe(self) -> Observation:
        return Observation({name: f.rgb for name, f in self._obs.cameras.items()},
                           self._obs.joints.position.astype(np.float32), self.task)

    def step(self, action) -> Observation:
        self.last_result = self.robot.write(self.robot.command(np.asarray(action, dtype=float)))
        self._obs = self.robot.read()
        return self.observe()
