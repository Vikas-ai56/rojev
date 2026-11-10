"""Switch cues: a switch dimension crossing its threshold on an applied action is reported once, with its name and new
side; continuous dimensions and actions that stay on one side are not."""
import numpy as np

from rojev.control import Mode, SwitchCues, TickRecord, TickResult
from rojev.protocol import Capture, Observation
from rojev.robot import KEEP, ActionDim, ActionSpec

SPEC = ActionSpec("stand-in", (ActionDim("v", -1, 1, 0.0),
                               ActionDim("gripper", -1, 1, KEEP, threshold=0.5, irreversible=True)), control_hz=50)
CAPTURE = Capture(1, 0.0, Observation({}, np.zeros(1, np.float32), "task"))


def tick(v, gripper):
    result = TickResult(0.0, np.array([v, gripper]), Mode.RUNNING, 1, "chunk", 0)
    return TickRecord(result, CAPTURE, CAPTURE, 0.1, 0.0)


def test_only_a_switch_crossing_its_threshold_is_a_cue():
    cues = []
    watch = SwitchCues(SPEC, cues.append)
    for v, gripper in [(0.0, -1.0), (0.9, -1.0), (-0.9, -0.9), (0.0, 1.0), (0.0, 0.8), (0.0, -1.0)]:
        watch(tick(v, gripper))
    assert cues == ["gripper on", "gripper off"]
