"""LIBERO's Franka Panda as a runtime `Embodiment`: the action spec, the cameras and the command a run starts from.

Action = LeRobot's LIBERO action (lerobot 0.6.1 `envs/libero.py:95-97, 254-256`): Box(-1, 1, (7,)), relative control
(`control_mode="relative"` sets robosuite's OSC `use_delta`, `envs/libero.py:357-359`): end-effector position delta
(3), axis-angle rotation delta (3), gripper (-1 open, +1 close). A zero delta holds the arm still. robosuite 1.4.0's
Panda gripper moves by the sign of its input (`models/grippers/panda_gripper.py:55-57`: closes above 0, opens below),
so the gripper is a switch at 0.0 that holds its last value; closing or opening it cannot be undone
(`rojev.control.SwitchCues` reports both as cues).
`CONTROL_HZ` = LeRobot's LIBERO `fps` (`envs/configs.py:325`) = robosuite `control_freq`.
`START` = LeRobot's no-op action `get_libero_dummy_action()` (`envs/libero.py:90-92`), the action its reset settles
the scene with: arm still, gripper open.
Cameras: `top` = LIBERO's `agentview` (LeRobot `image`), `wrist` = `robot0_eye_in_hand` (LeRobot `image2`).
"""
import numpy as np

from rojev.app import Embodiment
from rojev.robot import KEEP, ActionDim, ActionSpec

CONTROL_HZ = 20

LIBERO_PANDA = ActionSpec("libero-panda", (
    *(ActionDim(f"ee_{axis}", -1.0, 1.0, 0.0) for axis in ("x", "y", "z", "rx", "ry", "rz")),
    ActionDim("gripper_close", -1.0, 1.0, KEEP, threshold=0.0, irreversible=True),
), control_hz=CONTROL_HZ)

START = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

EMBODIMENT = Embodiment("libero-panda", LIBERO_PANDA, ("top", "wrist"), START)
