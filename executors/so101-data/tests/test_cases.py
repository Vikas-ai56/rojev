"""Step ends from a synthetic gripper trace with the recording's signature, and the case plan cut from them: every
case shows its moment's frame alone (the newest frame, as the live system sends), next-step truths follow the scene in all three ledger
variants (a contradicting ledger still expects the undone step), and the carry frame is asked about two steps with
opposite truths."""
import numpy as np
import pytest

from cases import COMPLETE, GOAL, OPEN, PICK, STACK, plan
from steps import step_ends


def trace() -> np.ndarray:
    return np.concatenate([np.full(100, 33.0), np.linspace(33, 2, 10), np.full(200, 2.0), np.linspace(2, 35, 10),
                           np.full(80, 35.0), np.linspace(35, 16, 10), np.full(250, 16.0), np.linspace(16, 30, 10),
                           np.full(120, 30.0)])


def test_step_ends_are_the_plateau_edges():
    ends = step_ends(trace())
    assert 100 <= ends.lid_closed <= 110 and 310 <= ends.drawer_open <= 320
    assert 400 <= ends.grasped <= 410 and 660 <= ends.released <= 670 and ends.length == 790


def test_a_trace_without_the_hold_plateau_is_refused():
    gripper = trace()
    gripper[400:] = 35.0
    with pytest.raises(ValueError, match="hold plateau"):
        step_ends(gripper)


def test_plan_cuts_frames_at_moments_and_pairs_opposite_truths():
    ends = step_ends(trace())
    labels = {"episodes": {"3": {k: getattr(ends, k) for k in ("lid_closed", "drawer_open", "grasped", "released",
                                                               "length")}}}
    cases = plan(labels)
    nexts = {v: [c for c in cases if c["kind"] == "next_step" and c["variant"] == v]
             for v in ("record", "no_record", "contradiction")}
    assert [c["truth"] for c in nexts["record"]] == [OPEN, PICK, STACK, COMPLETE]
    assert [c["record"] for c in nexts["record"]] == [[], [OPEN], [OPEN, PICK], [OPEN, PICK, STACK]]
    assert [c["truth"] for c in nexts["no_record"]] == [OPEN, PICK, STACK, COMPLETE]
    assert all(c["record"] is None for c in nexts["no_record"])
    for case in nexts["contradiction"]:
        assert case["truth"] in case["record"]
    by_frame = {c["frame"]: c["truth"] for c in nexts["record"]}
    assert all(by_frame[c["frame"]] == c["truth"] for c in nexts["contradiction"])
    for case in cases:
        for frames in case["frames"].values():
            assert frames == [case["frame"]]
    status = {(c["step"], c["frame"]): c["truth"] for c in cases if c["kind"] == "status"}
    carry = (ends.grasped + ends.released) // 2
    assert status[(PICK, carry)] == "done" and status[(STACK, carry)] == "in_progress"
    assert status[(GOAL, ends.released + 25)] == "done"
