"""Harness logic without the simulator: the interaction rules, the revisable planner, and the invariant checks."""
import json
from pathlib import Path

import numpy as np

from harness.deciders import (KeywordChooser, RevisableSequence, RewriteReviser, RuleBasedInteraction, destination,
                              rewrite)
from harness.invariants import Run, active_monotonic, changes_answered, gripper_held, no_stale_action
from rojev.protocol import Capture, Hold, Observation, ObservationSnapshot, Operation, OpKind
from rojev.system2.jev import CLARIFY, JevInteraction

PICK = Operation(OpKind.SWITCH_TO, subtask="pick up the steak")
PLACE = Operation(OpKind.SWITCH_TO, subtask="place the steak in the microwave")
SEQ = ["pick up the steak", "place the steak in the microwave", "press the start button"]


def snap(text="task", op=None, trigger="instruction", rev=1, since=None, now=0.0, interrupted=None):
    hold = Hold(op, now, interrupted) if op is not None and not op.moves_robot() else None
    return ObservationSnapshot(1, 1, now, Capture(0, now, Observation({}, np.zeros(16), text)), rev, text,
                               1 if op else None, op, since, (), trigger, hold=hold)


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
    assert unsure.decide(snap("hmm", PICK)).operation == Operation(OpKind.ASK, template=CLARIFY)


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
    assert no_stale_action(run) and gripper_held(run) and active_monotonic(run) and changes_answered(run)
    clean = run_of([tick(0.0), tick(0.05)], [{"kind": "instruction", "text": "go", "t": 1.0, "rev": 1},
                                             {"kind": "continue", "t": 1.2}])
    assert not (no_stale_action(clean) or gripper_held(clean) or active_monotonic(clean) or changes_answered(clean))
    assert json.dumps(run.summary)
