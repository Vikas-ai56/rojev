"""Harness logic without the simulator: the interaction rules, the revisable planner, and the invariant checks."""
import json
from pathlib import Path

import numpy as np

from harness.deciders import (KeywordChooser, RevisableSequence, RewriteReviser, RuleBasedInteraction, destination,
                              rewrite)
from embodiment import PANDA_OMRON
from harness.invariants import Run, active_monotonic, changes_answered, no_stale_action, switches_held
from rojev.protocol import USER, Capture, Hold, Observation, ObservationSnapshot, Operation, OpKind, Turn
from rojev.system2.jev import CLARIFY, JevInteraction

PICK = Operation(OpKind.SWITCH_TO, subtask="pick up the steak")
PLACE = Operation(OpKind.SWITCH_TO, subtask="place the steak in the microwave")
SEQ = ["pick up the steak", "place the steak in the microwave", "press the start button"]


def snap(text="task", op=None, trigger="instruction", rev=1, since=None, now=0.0, interrupted=None, task=None):
    """`text` = what the user just said; `task` = the task text in force (default: the same words, i.e. adopted)."""
    hold = Hold(op, now, interrupted) if op is not None and not op.moves_robot() else None
    task = text if task is None else task
    return ObservationSnapshot(1, 1, now, Capture(0, now, Observation({}, np.zeros(16), task)), rev, task,
                               1 if op else None, op, since, (), trigger, hold=hold,
                               dialogue=(Turn(now, USER, rev, text=text),))


def test_destination_and_rewrite():
    assert destination("actually put it on the plate") == "on the plate"
    assert rewrite("place the steak in the microwave", "on the plate") == "place the steak on the plate"
    assert rewrite("pick up the steak", "on the plate") == "pick up the steak"


def test_rules_map_utterances_to_acts():
    r = RuleBasedInteraction()
    assert r.decide(snap("stop now", PICK)).operation.kind == OpKind.ABORT
    assert r.decide(snap("nice work", PICK)).operation.kind == OpKind.CONTINUE
    assert r.decide(snap("put it in the microwave instead", PLACE)).operation.kind == OpKind.CONTINUE
    revised = r.decide(snap("no, on the plate", PLACE)).operation
    assert revised.kind == OpKind.SWITCH_TO and revised.subtask == "place the steak on the plate"
    assert r.decide(snap("x", PICK, trigger="deviation")).operation == Operation(OpKind.RETRY, subtask=PICK.subtask)


def test_wait_and_ask_resume_the_step_the_hold_interrupted():
    r = RuleBasedInteraction()
    assert r.decide(snap("wait", PICK)).operation.kind == OpKind.GUARDED_WAIT
    resumed = r.decide(snap("go on", Operation(OpKind.GUARDED_WAIT), interrupted=PICK)).operation
    assert resumed == PICK
    assert r.decide(snap("put it over there", PLACE)).operation == Operation(OpKind.ASK, template=CLARIFY)
    answered = r.decide(snap("on the plate", Operation(OpKind.ASK, template=CLARIFY), interrupted=PLACE)).operation
    assert answered == Operation(OpKind.SWITCH_TO, subtask="place the steak on the plate")


def test_jev_with_keyword_scores_and_the_rewrite_reviser_answers_like_the_rules():
    jev = JevInteraction(KeywordChooser(), reviser=RewriteReviser())
    assert jev.decide(snap("stop now", PICK)).operation.kind == OpKind.ABORT
    assert jev.decide(snap("wait", PICK)).operation.kind == OpKind.GUARDED_WAIT
    assert jev.decide(snap("go on", Operation(OpKind.GUARDED_WAIT), interrupted=PICK)).operation == PICK
    revised = jev.decide(snap("no, on the plate", PLACE)).operation
    assert revised == Operation(OpKind.SWITCH_TO, subtask="place the steak on the plate")
    unsure = JevInteraction(KeywordChooser(confidence=0.3), reviser=RewriteReviser())
    assert not unsure.decide(snap("hmm", PICK)).operation.moves_robot()
    assert jev.decide(snap("x", PICK, trigger="deviation")).operation == Operation(OpKind.GUARDED_WAIT)


def test_revisable_sequence_follows_the_instruction_and_its_budget():
    p = RevisableSequence(SEQ, budget_s=5.0)
    assert p.decide(snap(op=None, rev=0)).operation.subtask == SEQ[0]
    assert p.decide(snap("put it on the plate", PICK, "heartbeat", since=0.0, now=6.0)).operation.subtask == \
        "place the steak on the plate"
    assert p.decide(snap(op=Operation(OpKind.GUARDED_WAIT), trigger="heartbeat")).operation.kind == OpKind.CONTINUE


def run_of(ticks, events, sim_s=10.0):
    return Run(Path("."), ticks, events, {"sim_s": sim_s})


def tick(t, source="chunk", cmd=1, gripper=-1.0, review=False):
    return {"t": t, "source": source, "cmd": cmd, "review": review, "violations": [],
            "action": [0.0] * 6 + [gripper] + [0.0] * 5}


def test_invariants_catch_what_they_promise():
    ticks = [tick(0.0), tick(0.05, gripper=1.0, review=True), tick(0.1, cmd=1)]
    events = [{"kind": "ack", "status": "revoked", "cmd": 1, "t": 0.07},
              {"kind": "ack", "status": "activated", "cmd": 2, "t": 0.0},
              {"kind": "ack", "status": "activated", "cmd": 1, "t": 0.1},
              {"kind": "instruction", "text": "stop", "t": 1.0, "rev": 1}]
    run = run_of(ticks, events)
    assert no_stale_action(run) and switches_held(run, PANDA_OMRON) and active_monotonic(run) and changes_answered(run)
    clean = run_of([tick(0.0), tick(0.05)], [{"kind": "instruction", "text": "go", "t": 1.0, "rev": 1},
                                             {"kind": "continue", "t": 1.2}])
    assert not (no_stale_action(clean) or switches_held(clean, PANDA_OMRON) or active_monotonic(clean) or changes_answered(clean))
    assert json.dumps(run.summary)


def test_jev_decisions_become_calibration_cases_a_person_can_label(tmp_path):
    from calibrate_jev import images, labelled
    from harness.recording import InteractionCases

    cams = {"left": np.zeros((4, 4, 3), np.uint8), "right": np.full((4, 4, 3), 9, np.uint8)}
    s = ObservationSnapshot(1, 7, 3.0, Capture(5, 3.0, Observation(cams, np.zeros(16), "task")), 1, "task", 1,
                            PICK, 0.0, (), "instruction", dialogue=(Turn(3.0, USER, 1, text="stop now"),))
    recorder = InteractionCases(JevInteraction(KeywordChooser(), reviser=RewriteReviser()), ["left", "right"],
                                tmp_path, "run1")
    assert recorder.decide(s).operation.kind == OpKind.ABORT and recorder.saved == 1
    lines = (tmp_path / "labels.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    assert row["predicted"] == "abort" and row["label"] is None and row["question"] == "act"
    assert labelled(tmp_path) == []
    (tmp_path / "labels.jsonl").write_text(lines[0].replace('"label": null', '"label": "abort"') + "\n")
    [(case, label)] = labelled(tmp_path)
    assert label == "abort" and "abort" in case["options"] and case["situation"]["user_just_said"] == "stop now"
    assert [i[0, 0, 0] for i in images(tmp_path, case)] == [0, 9] and labelled(tmp_path, "task") == []


class TaskStub:
    """Scores a revision, then the trained sentence "put the steak on the plate"."""

    name = "stub"

    def choose(self, s, options, question=None):
        best = "put the steak on the plate" if "put the steak on the plate" in [o.label for o in options] else "revise"
        return {o.label: (0.9 if o.label == best else 0.1 / (len(options) - 1)) for o in options}


def test_a_revision_records_both_questions_as_cases(tmp_path):
    from harness.recording import InteractionCases
    from rojev.system2.jev import CanonicalTask

    cams = {"left": np.zeros((4, 4, 3), np.uint8)}
    s = ObservationSnapshot(1, 8, 3.0, Capture(5, 3.0, Observation(cams, np.zeros(16), "task")), 1, "task", 1,
                            PICK, 0.0, (), "instruction", dialogue=(Turn(3.0, USER, 1, text="plate pls"),))
    stub = TaskStub()
    decider = JevInteraction(stub, CanonicalTask(stub, ["put the steak in the microwave", "put the steak on the plate"]))
    v = InteractionCases(decider, ["left"], tmp_path, "run1").decide(s)
    assert v.task == "put the steak on the plate"
    rows = [json.loads(line) for line in (tmp_path / "labels.jsonl").read_text().splitlines()]
    assert [(r["question"], r["predicted"]) for r in rows] == [("act", "revise"), ("task", "put the steak on the plate")]
    case = json.loads((tmp_path / "run1-r8-task" / "case.json").read_text())
    assert "none" in case["options"] and case["question_text"].startswith("The user asked the robot for a new task")


def test_a_calibration_is_judged_on_the_cases_it_decides():
    from calibrate_jev import decided_accuracy
    from rojev.system2.jev import Calibration

    rows = [{"a": 0.9, "b": 0.1}, {"a": 0.9, "b": 0.1}, {"a": 0.55, "b": 0.45}]
    assert decided_accuracy(rows, ["a", "b", "b"], Calibration(margin=0.5)) == (0.5, 2 / 3)
    assert decided_accuracy(rows, ["a", "a", "b"], Calibration(margin=0.95)) == (None, 0.0)
