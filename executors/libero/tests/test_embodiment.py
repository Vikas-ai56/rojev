"""The LIBERO Panda embodiment: action layout and bounds, what holding means, the gripper switch the done-check cues
watch, and the start command, checked against LeRobot's own LIBERO definitions."""
import numpy as np
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.libero import ACTION_DIM, ACTION_HIGH, ACTION_LOW, get_libero_dummy_action

from rojev.control import Mode, SwitchCues, TickRecord, TickResult
from rojev.protocol import Capture, Observation

from embodiment import CONTROL_HZ, EMBODIMENT, LIBERO_PANDA, START


def test_the_action_matches_lerobots_libero_action_space():
    assert LIBERO_PANDA.size == ACTION_DIM == 7
    assert np.all(LIBERO_PANDA.low == ACTION_LOW) and np.all(LIBERO_PANDA.high == ACTION_HIGH)
    assert CONTROL_HZ == LIBERO_PANDA.control_hz == LiberoConfig().fps


def test_holding_keeps_the_arm_still_and_the_gripper_where_it_was():
    closed = np.array([0.3, -0.2, 0.1, 0.0, 0.05, -0.1, 1.0])
    assert LIBERO_PANDA.hold(closed).tolist() == [0, 0, 0, 0, 0, 0, 1.0]
    assert LIBERO_PANDA.hold(START).tolist() == [0, 0, 0, 0, 0, 0, -1.0]


def test_the_gripper_is_an_irreversible_switch_at_zero():
    gripper = LIBERO_PANDA.dims[6]
    assert gripper.threshold == 0.0 and gripper.irreversible
    assert [d.threshold for d in LIBERO_PANDA.dims[:6]] == [None] * 6
    grasp = START.copy()
    grasp[6] = 0.4
    assert LIBERO_PANDA.irreversible_change(START, grasp) == ["gripper_close"]
    assert LIBERO_PANDA.irreversible_change(grasp, START) == ["gripper_close"]
    assert LIBERO_PANDA.irreversible_change(START, START * 0.5) == []


def test_switch_cues_report_a_grasp_and_a_release():
    cues: list[str] = []
    watch = SwitchCues(LIBERO_PANDA, cues.append)
    capture = Capture(0, 0.0, Observation({}, np.zeros(8), ""))
    closing = START.copy()
    closing[6] = 1.0
    for index, action in enumerate((START, START, closing, closing, START)):
        tick = TickResult(t=index * LIBERO_PANDA.dt, action=action, mode=Mode.RUNNING, cmd_id=1, source="chunk",
                          index=index)
        watch(TickRecord(tick, capture, capture, 0.0, 0.0))
    assert cues == ["gripper_close on", "gripper_close off"]


def test_the_start_command_is_lerobots_no_op_action():
    assert START.tolist() == list(map(float, get_libero_dummy_action()))
    assert EMBODIMENT.start.tolist() == START.tolist()
    assert EMBODIMENT.cameras == ("top", "wrist")
    assert EMBODIMENT.spec is LIBERO_PANDA
