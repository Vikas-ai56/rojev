"""SO-101 LeRobot units (`rojev.robot.so101.lerobot_units`): joint order, body degrees, gripper 0–100, round trips."""
import numpy as np
import pytest

from rojev.robot import so101


def test_joint_order_is_lerobots_and_matches_the_simulated_arm():
    units = so101.lerobot_units()
    assert units.names == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    assert so101.simulated().joint_names == units.names


def test_body_joints_are_degrees_around_zero_and_the_gripper_spans_its_range():
    units = so101.lerobot_units()
    lo, hi = units.gripper_rad
    q = np.array([np.pi / 2, -np.pi / 4, 0.0, 0.1, -0.2, hi])
    v = units.to_lerobot(q)
    assert v[:5] == pytest.approx(units.sign * np.degrees(q[:5] - units.zero_rad))
    assert v[5] == pytest.approx(100.0)
    assert units.to_lerobot(np.array([0, 0, 0, 0, 0, lo]))[5] == pytest.approx(0.0)


def test_round_trip_holds_for_chunks():
    units = so101.lerobot_units()
    chunk = np.random.default_rng(0).uniform(-1.5, 1.5, size=(16, 6))
    assert units.from_lerobot(units.to_lerobot(chunk)) == pytest.approx(chunk)


def test_sign_and_offset_are_applied():
    units = so101.LeRobotUnits(("a", "b", "g"), np.array([-1.0, 1.0]), np.array([0.5, 0.0]), (0.0, 2.0))
    assert units.to_lerobot(np.array([0.5, 0.0, 1.0])) == pytest.approx([0.0, 0.0, 50.0])
    assert units.from_lerobot(np.array([90.0, 0.0, 0.0])) == pytest.approx([0.5 - np.pi / 2, 0.0, 0.0])


def test_wrong_joint_count_is_refused():
    with pytest.raises(ValueError, match="expected 6 joints"):
        so101.lerobot_units().to_lerobot(np.zeros(5))
