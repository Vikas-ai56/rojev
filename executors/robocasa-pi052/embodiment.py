"""RoboCasa PandaOmron (Franka Panda arm on an Omron base) as a runtime `ActionSpec`: the first robot plug-in.

Layout = LeRobot RoboCasa wrapper / π0.5² action (vendor/lerobot-steerable/src/lerobot/envs/robocasa.py:37):
ee_pos(3) + ee_rot(3) + gripper(1) + base_motion(4) + control_mode(1), all in [-1, 1].
Hold values from robosuite's default_pandaomron.json: arm OSC_POSE input "delta", torso JOINT_POSITION (default
"delta"), base JOINT_VELOCITY, so 0 keeps each still. Gripper and base mode are switches at 0.5
(robocasa/wrappers/gym_wrapper.py:125,137: < 0.5 → -1, else 1); releasing or closing the gripper cannot be undone.
`CONTROL_HZ` = robocasa kitchen.py `control_freq` default = LeRobot RoboCasaEnv fps.

`PandaOmronEffect` (execution monitor plug-in): tracked = end-effector position relative to the base (agent_pos[0:3],
vendor/lerobot-steerable/src/lerobot/envs/robocasa.py:36); commanded = the OSC position delta and the torso input
(action 0:3 and 10), whose map to motion the monitor fits; resolution 0.1 mm (the simulator has no sensor noise, so
this is the smallest hand motion treated as real). Orientation and base motion are not monitored yet.
"""
import numpy as np

from rojev.robot import KEEP, ActionDim, ActionSpec, EffectModel

CONTROL_HZ = 20

PANDA_OMRON = ActionSpec("robocasa-PandaOmron", (
    *(ActionDim(f"ee_{a}", -1, 1, 0.0) for a in ("x", "y", "z", "rx", "ry", "rz")),
    ActionDim("gripper_close", -1, 1, KEEP, threshold=0.5, irreversible=True),
    *(ActionDim(f"base_{a}", -1, 1, 0.0) for a in ("x", "y", "yaw")),
    ActionDim("torso", -1, 1, 0.0),
    ActionDim("base_mode", -1, 1, KEEP, threshold=0.5),
), control_hz=CONTROL_HZ)


class PandaOmronEffect(EffectModel):
    def tracked(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state[:3], dtype=float)

    def command(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=float)[[0, 1, 2, 10]]

    @property
    def resolution(self) -> float:
        return 1e-4


ARM_DIMS = (0, 1, 2, 3, 4, 5, 10)


def to_action_space(actions: np.ndarray) -> np.ndarray:
    """Policies regress PandaOmron's on/off switches (gripper, base mode) as numbers (π0.5² overshoots ±1 slightly,
    measured up to 1.018); the robot takes them as switches. Map each to the level RoboCasa's own gym wrapper uses
    (robocasa/wrappers/gym_wrapper.py:125,137: below the threshold → low, else → high), so the chunk is in the
    robot's action space; continuous dimensions are left as the policy produced them."""
    out = actions.copy()
    for i, d in enumerate(PANDA_OMRON.dims):
        if d.threshold is not None:
            out[:, i] = np.where(out[:, i] >= d.threshold, d.high, d.low)
    return out
