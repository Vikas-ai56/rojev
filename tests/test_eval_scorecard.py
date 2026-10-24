"""Revision-benchmark scorecard on synthetic run records: every metric, the edge cases (no answer, a change at the run
end, no scores), the act-class mapping, ECE, aggregation per arm and both renderers."""
import json

import pytest

from rojev.eval import Case, Injection, RunRecords, aggregate, render_json, render_markdown, score_run
from rojev.eval.scorecard import expected_calibration_error

HZ = 20
DT = 1 / HZ


def ticks(*segments):
    """(start, stop, source, cmd) segments → one tick per control period in [start, stop)."""
    out = []
    for start, stop, source, cmd in segments:
        k = round(start * HZ)
        while k < round(stop * HZ):
            out.append({"t": round(k * DT, 4), "source": source, "cmd": cmd, "action": [0.0] * 7})
            k += 1
    return out


def cmd(t, n, op, subtask=None, kind="command", trigger="instruction", lane="interaction", decider="jev"):
    return {"t": t, "kind": kind, "cmd": n, "op": op, "subtask": subtask, "lane": lane, "trigger": trigger,
            "decider": decider}


def said(t, rev, text="actually put it on the plate"):
    return {"t": t, "kind": "instruction", "rev": rev, "text": text}


def cont(t, n, trigger="instruction", lane="interaction"):
    return {"t": t, "kind": "continue", "cmd": n, "renewed": True, "lane": lane, "trigger": trigger, "decider": "x"}


def records(events, tick_list, decisions=(), sim_s=10.0, success=False, hz=HZ):
    summary = {"sim_s": sim_s, "control_hz": hz}
    if success is not None:
        summary["task_success"] = success
    return RunRecords(summary, list(events), tick_list, list(decisions))


def swap(t=4.0, expected="revise"):
    return Injection(t, "goal_swap", expected, goal="plate", old_goal="microwave", phase="transport")


START = cmd(0.05, 1, "switch_to", "pick up the steak", trigger="heartbeat", lane="planning")
TO_MICROWAVE = cmd(2.0, 2, "switch_to", "place the steak in the microwave", trigger="heartbeat", lane="planning")


def goal_swap_run():
    """Swap at 4.0 while placing in the microwave; revised to the plate at 4.5, first tick under it at 4.8; a
    heartbeat command at 6.0 wrongly goes back to the microwave until 7.0, then the plate again."""
    events = [START, TO_MICROWAVE, said(4.0, 1), {"t": 4.0, "kind": "review", "cmd": 2},
              cont(4.2, 2, trigger="heartbeat", lane="planning"),
              cmd(4.5, 3, "switch_to", "place the steak on the plate"),
              cmd(6.0, 4, "switch_to", "place the steak in the microwave", trigger="heartbeat", lane="planning"),
              cmd(7.0, 5, "switch_to", "place the steak on the plate", trigger="heartbeat", lane="planning")]
    t = ticks((0, 0.4, "hold", None), (0.4, 2.3, "chunk", 1), (2.3, 4.8, "chunk", 2), (4.8, 6.2, "chunk", 3),
              (6.2, 7.3, "chunk", 4), (7.3, 10.0, "chunk", 5))
    return records(events, t)


def test_injection_and_case_validation():
    with pytest.raises(ValueError):
        Injection(1.0, "teleport", "revise")
    with pytest.raises(ValueError):
        Injection(1.0, "stop", "halt")
    with pytest.raises(ValueError):
        Injection(1.0, "goal_swap", "revise", goal="plate")
    with pytest.raises(ValueError):
        Case("c", "jev", (Injection(5.0, "stop", "abort"), Injection(5.0, "harmless", "continue")))
    assert Injection(1.0, "deviation", "revise").trigger == "deviation"
    assert Injection(1.0, "ambiguous", "ask").trigger == "instruction"


def test_goal_swap_latencies_act_wrong_goal_and_wasted_motion():
    s = score_run(Case("swap", "jev", (swap(),)), goal_swap_run()).injections[0]
    assert s.change_t == 4.0 and s.answer_t == 4.5 and s.answer_kind == "command" and s.answer_lane == "interaction"
    assert s.answer_latency_s == pytest.approx(0.5)
    assert s.execution_latency_s == pytest.approx(0.8)
    assert (s.act, s.correct) == ("revise", True)
    assert s.wrong_goal_commitments == 1
    assert s.wasted_motion_ticks == 16 + 22
    assert s.wasted_motion_s == pytest.approx(38 / HZ)
    assert s.pause_ticks == 0 and s.pause_fraction == 0.0
    assert s.injection.phase == "transport"


def test_heartbeat_after_the_change_is_not_its_answer():
    run = records([START, said(4.0, 1), cont(4.2, 1, trigger="heartbeat", lane="planning")],
                  ticks((0, 10, "chunk", 1)))
    s = score_run(Case("c", "a", (Injection(4.0, "harmless", "continue"),)), run).injections[0]
    assert s.act is None and s.correct is False and "no applied answer" in s.notes


@pytest.mark.parametrize("event,act", [
    (cmd(4.1, 2, "switch_to", "pick up the steak", kind="reaffirm"), "continue"),
    (cont(4.1, 1), "continue"),
    (cmd(4.1, 2, "retry", "pick up the steak"), "revise"),
    (cmd(4.1, 2, "guarded_wait"), "wait"),
    (cmd(4.1, 2, "ask"), "ask"),
    (cmd(4.1, 2, "abort"), "abort"),
    (cmd(4.1, 2, "complete"), "complete"),
])
def test_act_class_of_the_applied_answer(event, act):
    run = records([START, said(4.0, 1), event], ticks((0, 4.3, "chunk", 1), (4.3, 10, "hold", 2)))
    s = score_run(Case("c", "a", (Injection(4.0, "harmless", "continue"),)), run).injections[0]
    assert s.act == act and s.correct == (act == "continue")


def test_continue_has_no_execution_latency():
    run = records([START, said(4.0, 1), cont(4.1, 1)], ticks((0, 10, "chunk", 1)))
    s = score_run(Case("c", "a", (Injection(4.0, "harmless", "continue"),)), run).injections[0]
    assert s.answer_latency_s == pytest.approx(0.1) and s.execution_latency_s is None


def test_stop_pause_clipped_to_the_next_change():
    events = [START, said(4.0, 1, "stop"), cmd(4.0, 2, "abort"), said(5.0, 2, "go on"),
              cmd(5.0, 3, "switch_to", "pick up the steak")]
    run = records(events, ticks((0, 4.1, "chunk", 1), (4.1, 5.0, "hold", 2), (5.0, 10, "chunk", 3)))
    case = Case("c", "a", (Injection(4.0, "stop", "abort"), Injection(5.0, "harmless", "revise")))
    first, second = score_run(case, run, pause_window_s=3.0).injections
    assert (first.pause_ticks, first.pause_fraction) == (18, pytest.approx(18 / 20))
    assert first.execution_latency_s == pytest.approx(0.1)
    assert (second.act, second.correct, second.pause_ticks) == ("revise", True, 0)


def test_ask_precision_counts_stray_asks_as_misses():
    events = [START, cmd(1.0, 2, "ask", trigger="heartbeat"), said(4.0, 1, "put it over there"),
              cmd(4.0, 3, "ask"), said(6.0, 2), cmd(6.0, 4, "ask")]
    run = records(events, ticks((0, 10, "hold", 1)))
    case = Case("c", "a", (Injection(4.0, "ambiguous", "ask"), swap(6.0)))
    r = score_run(case, run)
    assert (r.asks, r.asks_expected) == (3, 1)
    assert [s.asks for s in r.injections] == [1, 1]
    assert aggregate([r])[0].ask_precision.value == pytest.approx(1 / 3)


def test_the_owners_safety_hold_for_an_unanswered_change_is_no_decision():
    events = [START, said(4.0, 1, "wait"),
              {"t": 5.0, "kind": "unanswered", "request": 6, "trigger": "instruction", "why": "timeout"},
              cmd(5.0, 2, "guarded_wait", decider=None)]
    run = records(events, ticks((0, 5.0, "chunk", 1), (5.0, 10, "hold", 2)))
    score = score_run(Case("c", "a", (Injection(4.0, "pause", "wait"),)), run)
    s = score.injections[0]
    assert (s.act, s.correct, s.unanswered, s.answer_decider) == ("unanswered", False, 1, None)
    assert s.answer_latency_s == pytest.approx(1.0)
    [arm] = aggregate([score])
    assert arm.act_accuracy.value == 0.0 and arm.unanswered_rate.value == 1.0


def test_pause_is_judged_apart_for_changes_that_should_and_should_not_stop_the_robot():
    events = [START, said(2.0, 1, "nice work"), cmd(2.0, 2, "switch_to", kind="reaffirm"),
              said(6.0, 2, "stop"), cmd(6.0, 3, "abort")]
    run = records(events, ticks((0, 3.0, "chunk", 1), (3.0, 4.0, "hold", 2), (4.0, 6.0, "chunk", 2),
                                (6.0, 10, "hold", 3)))
    case = Case("c", "a", (Injection(2.0, "harmless", "continue"), Injection(6.0, "stop", "abort")))
    [arm] = aggregate([score_run(case, run)])
    assert arm.pause_when_moving.n == 1 and 0.0 < arm.pause_when_moving.value < 1.0
    assert arm.hold_when_asked.n == 1 and arm.hold_when_asked.value == pytest.approx(1.0)


def test_no_answer_near_the_run_end_is_not_judged():
    run = records([START, said(9.0, 1)], ticks((0, 10, "chunk", 1)), sim_s=10.0)
    s = score_run(Case("c", "a", (Injection(9.0, "stop", "abort"),)), run).injections[0]
    assert s.change_t == 9.0 and s.act is None and s.correct is None
    assert any("of the window left" in n for n in s.notes)


def test_change_at_or_after_the_run_end_is_not_reached():
    run = records([START], ticks((0, 10, "chunk", 1)), sim_s=10.0)
    s = score_run(Case("c", "a", (Injection(10.0, "stop", "abort"),)), run).injections[0]
    assert s.change_t is None and s.correct is None and s.notes == ("change at or after the run end",)
    missing = score_run(Case("c", "a", (Injection(4.0, "stop", "abort"),)), run).injections[0]
    assert missing.change_t is None and missing.notes == ("no instruction event for this change in the records",)
    assert aggregate([score_run(Case("c", "a", (Injection(4.0, "stop", "abort"),)), run)])[0].changes == 0


def test_deviation_change_is_the_fault_onset():
    events = [START, {"t": 6.4, "kind": "deviation", "cmd": 1, "score": 3.0},
              cont(6.4, 1, trigger="heartbeat", lane="planning"),
              cmd(6.5, 2, "retry", "pick up the steak", trigger="deviation")]
    run = records(events, ticks((0, 6.8, "chunk", 1), (6.8, 10, "chunk", 2)))
    s = score_run(Case("c", "a", (Injection(6.0, "deviation", "revise"),)), run).injections[0]
    assert s.change_t == 6.0 and s.answer_latency_s == pytest.approx(0.5)
    assert s.execution_latency_s == pytest.approx(0.8) and (s.act, s.correct) == ("revise", True)


def test_goal_metrics_need_goal_text_and_subtasks():
    no_goal = score_run(Case("c", "a", (Injection(4.0, "goal_swap", "revise"),)), goal_swap_run()).injections[0]
    assert no_goal.wrong_goal_commitments is None and no_goal.wasted_motion_ticks is None
    events = [START, said(4.0, 1), cmd(4.5, 2, "switch_to", None)]
    run = records(events, ticks((0, 4.8, "chunk", 1), (4.8, 10, "chunk", 2)))
    s = score_run(Case("c", "native", (swap(),)), run).injections[0]
    assert s.wrong_goal_commitments is None and s.wasted_motion_ticks is None and s.wasted_motion_s is None
    assert any("S1 plans its own" in n for n in s.notes) and any("[2]" in n for n in s.notes)


def test_wasted_motion_without_control_rate_stays_in_ticks():
    run = goal_swap_run()
    run.summary.pop("control_hz")
    s = score_run(Case("c", "a", (swap(),)), run).injections[0]
    assert s.wasted_motion_ticks == 38 and s.wasted_motion_s is None


def test_success_from_summary_or_none():
    assert score_run(Case("c", "a", ()), records([], [], success=True)).success is True
    assert score_run(Case("c", "a", ()), records([], [], success=None)).success is None


def scored(t, probs, trigger="instruction", lane="interaction", request=6):
    return {"t_start": t, "t_end": t, "lane": lane, "request": request, "trigger": trigger, "scores": probs}


def test_act_confidence_sums_labels_of_a_class():
    decisions = [scored(4.0, {"continue": 0.2, "revise": 0.3, "retry": 0.25, "ask": 0.25})]
    run = records([START, said(4.0, 1), cmd(4.0, 2, "switch_to", "x")], ticks((0, 10, "chunk", 1)), decisions)
    s = score_run(Case("c", "a", (Injection(4.0, "harmless", "revise"),)), run).injections[0]
    assert s.confidence == pytest.approx(0.55) and s.confident_correct is True


def test_act_confidence_ignores_unscored_planning_and_other_triggers():
    decisions = [{"t_start": 4.0, "lane": "interaction", "trigger": "instruction", "request": 5},
                 scored(4.0, {"continue": 1.0}, lane="planning"), scored(4.1, {"continue": 1.0}, trigger="hold"),
                 scored(4.2, {"continue": 0.1, "abort": 0.9})]
    run = records([START, said(4.0, 1)], ticks((0, 10, "chunk", 1)), decisions)
    s = score_run(Case("c", "a", (Injection(4.0, "stop", "wait"),)), run).injections[0]
    assert s.confidence == pytest.approx(0.9) and s.confident_correct is False


def test_no_scores_gives_no_ece():
    r = score_run(Case("c", "a", (swap(),)), goal_swap_run())
    assert r.injections[0].confidence is None and r.calibration_pairs() == []
    arm = aggregate([r])[0]
    assert arm.act_ece.value is None and arm.act_ece.n == 0


@pytest.mark.parametrize("probs", [{"continue": 0.5, "revise": 0.2}, {"continue": 0.5, "dance": 0.5}])
def test_scores_must_be_a_distribution_over_known_acts(probs):
    run = records([START, said(4.0, 1)], ticks((0, 10, "chunk", 1)), [scored(4.0, probs)])
    with pytest.raises(ValueError):
        score_run(Case("c", "a", (Injection(4.0, "harmless", "continue"),)), run)


def test_expected_calibration_error():
    assert expected_calibration_error([]) is None
    assert expected_calibration_error([(1.0, True), (1.0, True)]) == pytest.approx(0.0)
    pairs = [(0.95, True), (0.95, False), (0.15, False), (0.15, False)]
    assert expected_calibration_error(pairs) == pytest.approx(0.5 * 0.45 + 0.5 * 0.15)
    assert expected_calibration_error([(0.1, True)], bins=10) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        expected_calibration_error([(1.2, True)])


def test_aggregate_per_arm_and_render():
    jev = score_run(Case("swap", "jev", (swap(),)), goal_swap_run())
    late = records([START, said(4.0, 1), cmd(5.0, 2, "switch_to", "place the steak on the plate")],
                   ticks((0, 5.1, "chunk", 1), (5.1, 10, "chunk", 2)),
                   [scored(4.0, {"revise": 0.8, "continue": 0.2})], success=True)
    jev2 = score_run(Case("swap-2", "jev", (swap(),)), late)
    silent = score_run(Case("swap", "native", (swap(),)),
                       records([START, said(4.0, 1)], ticks((0, 10, "chunk", 1)), success=None))
    arms = aggregate([jev, silent, jev2])
    assert [a.arm for a in arms] == ["jev", "native"]
    a, n = arms
    assert (a.runs, a.changes, a.missed) == (2, 2, 0)
    assert a.success.value == pytest.approx(0.5) and a.success.n == 2
    assert a.answer_latency_s.value == pytest.approx(0.75) and a.answer_latency_s.n == 2
    assert a.act_accuracy.value == 1.0 and a.wrong_goal_commitments.value == pytest.approx(0.5)
    assert a.act_ece.n == 1 and a.act_ece.value == pytest.approx(0.2)
    assert a.ask_precision.value is None and a.ask_precision.n == 0
    assert (n.missed, n.success.n, n.answer_latency_s.n) == (1, 0, 0) and n.act_accuracy.value == 0.0
    md = render_markdown(arms, [jev, silent])
    assert md.count("\n| jev |") == 1 and "| swap | native | 4 | goal_swap | transport | revise | — |" in md
    data = json.loads(json.dumps(render_json(arms, [jev])))
    assert data["arms"][0]["act_ece"] == {"value": pytest.approx(0.2), "n": 1}
    assert data["runs"][0]["injections"][0]["injection"]["goal"] == "plate"
