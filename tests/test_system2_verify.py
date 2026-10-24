"""End-state checks and the step watcher with stub verifiers: confirmation streaks, one event per command, no checks
for commands without a step, failures counted; `MJevVerifier` builds one MJev request for all steps."""
import time

import numpy as np
import pytest

from rojev.protocol import AcceptedGoal, Capture, Observation, Operation, OpKind, StepEvent, Transition
from rojev.system2.verify import STATES, MJevVerifier, StepWatcher, placement, question

OBS = Observation({"left": np.zeros((4, 4, 3), np.uint8)}, np.zeros(1, np.float32), "put the cube in A")
PICK = Operation(OpKind.SWITCH_TO, subtask="pick the cube")


def goal(cmd, op=PICK):
    return AcceptedGoal(1, cmd, 0, op, Transition.IMMEDIATE, 100.0, "put the cube in A")


class ScriptedVerifier:
    name = "stub"

    def __init__(self, p_done):
        self.p_done, self.calls = list(p_done), 0

    def check(self, images, steps):
        self.calls += 1
        p = self.p_done.pop(0)
        if isinstance(p, Exception):
            raise p
        return {s: {"done": p, "possible": 1 - p, "blocked": 0.0} for s in steps}


def watcher(verifier, running, **kw):
    posted, captures = [], iter(range(1, 100))
    w = StepWatcher(verifier, running, lambda: Capture(next(captures), 1.0, OBS), posted.append, lambda: 1.0,
                    ["left"], **kw)
    return w, posted


def test_a_step_is_reported_once_after_consecutive_confirmations():
    w, posted = watcher(ScriptedVerifier([0.9, 0.2, 0.8, 0.95, 0.99]), lambda: goal(3), confirmations=2)
    events = [w.check_once() for _ in range(5)]
    assert [e is not None for e in events] == [False, False, False, True, False]
    assert len(posted) == 1 and posted[0].cmd_id == 3 and posted[0].source == "stub"
    assert posted[0].step == "pick the cube" and posted[0].detail["p_done"] == 0.95
    assert w.stats.checks == 4


def test_the_streak_restarts_for_a_new_command():
    running = iter([goal(1), goal(2), goal(2)])
    w, posted = watcher(ScriptedVerifier([0.9, 0.9, 0.9]), lambda: next(running), confirmations=2)
    for _ in range(3):
        w.check_once()
    assert [e.cmd_id for e in posted] == [2]


def test_commands_without_a_step_are_not_checked_and_failures_are_counted():
    native = goal(1, Operation(OpKind.SWITCH_TO, subtask=None))
    verifier = ScriptedVerifier([RuntimeError("down")])
    w, posted = watcher(verifier, lambda: native)
    assert w.check_once() is None and verifier.calls == 0
    w2, _ = watcher(verifier, lambda: goal(1))
    assert w2.check_once() is None and w2.stats.errors == 1 and "down" in w2.stats.last_error
    w3, _ = watcher(verifier, lambda: goal(1, Operation(OpKind.GUARDED_WAIT)))
    assert w3.check_once() is None


class FakeMJev:
    def __init__(self):
        self.asked = []

    def ask(self, images, questions):
        self.asked.append(questions)
        return {"answers": {q["id"]: {"probabilities": {"done": 0.7, "possible": 0.2, "blocked": 0.1}}
                            for q in questions}, "timings_ms": {"mjev": 5.0}}


def test_mjev_checks_every_step_in_one_request():
    mjev = FakeMJev()
    out = MJevVerifier(mjev).check([OBS.images["left"]], ["pick the cube", "open the door"])
    assert out["open the door"]["done"] == 0.7 and len(mjev.asked) == 1
    q = mjev.asked[0][1]
    assert q["question"] == question("open the door")
    assert [o["id"] for o in q["answer_space"]["options"]] == list(STATES)


def test_a_claimed_step_counts_only_when_the_check_confirms_it():
    claim = StepEvent(1, 4, 2.0, "s1-planner", "pick the cube", {"next": "place the cube"})
    w, posted = watcher(ScriptedVerifier([0.2, 0.9]), lambda: goal(4, Operation(OpKind.SWITCH_TO)))
    assert w.confirm(claim) is None and w.stats.unconfirmed == 1 and posted == []
    event = w.confirm(claim)
    assert event.checked and event.step == "pick the cube" and event.source == "stub"
    assert event.detail["claimed_by"] == "s1-planner" and event.detail["next"] == "place the cube"
    assert posted == [event] and w.stats.claims == 2


def test_the_watcher_thread_confirms_claims_between_periodic_checks():
    w, posted = watcher(ScriptedVerifier([0.95] * 50), lambda: goal(4, Operation(OpKind.SWITCH_TO)), every_s=5.0)
    w.start()
    w.claim(StepEvent(1, 4, 2.0, "s1-planner", "pick the cube"))
    deadline = time.monotonic() + 2.0
    while not posted and time.monotonic() < deadline:
        time.sleep(0.01)
    w.stop()
    assert len(posted) == 1 and posted[0].checked


def test_onejev_answers_the_same_question_with_neutral_option_ids(monkeypatch):
    from types import SimpleNamespace

    from rojev.system2.verify import OneJevVerifier
    from rojev.system2.verify import checks as checks_module

    asked = []

    def fake_system_one(client, image, state, questions, **kw):
        asked.append(questions["q0"])
        return SimpleNamespace(answers={k: SimpleNamespace(probabilities={"o1": 0.7, "o2": 0.2, "o3": 0.1})
                                        for k in questions})

    monkeypatch.setattr(checks_module.onejev, "system_one", fake_system_one)
    out = OneJevVerifier(None).check([OBS.images["left"]], ["open the door"])
    assert out == {"open the door": {"done": 0.7, "possible": 0.2, "blocked": 0.1}}
    assert asked[0].instructions == question("open the door") and set(asked[0].criteria) == {"o1", "o2", "o3"}


def test_postcondition_templates_turn_steps_into_concrete_yes_no_facts():
    from rojev.system2.verify import fact

    assert fact("place the steak in the microwave") == "Is the steak inside the microwave?"
    assert fact("put steak on the plate") == "Is the steak on the plate?"
    assert fact("close the freezer door") == "Is the freezer door closed?"
    assert fact("pick up the steak") == "Is the robot holding the steak?"
    assert fact("go to refrigerator") is None


def test_mjev_in_postcondition_mode_asks_the_fact_and_cannot_confirm_an_untemplated_step():
    class YesNoMJev:
        def __init__(self):
            self.asked = []

        def ask(self, images, questions):
            self.asked.append(questions)
            return {"answers": {q["id"]: {"probabilities": {"yes": 0.8, "no": 0.2}} for q in questions},
                    "timings_ms": {"mjev": 5.0}}

    mjev = YesNoMJev()
    out = MJevVerifier(mjev, postconditions=True).check([OBS.images["left"]], ["close the door", "go to the fridge"])
    assert out["close the door"] == {"done": 0.8, "possible": pytest.approx(0.2), "blocked": 0.0}
    assert out["go to the fridge"]["done"] == 0.0
    assert [q["question"] for q in mjev.asked[0]] == ["Is the door closed?"]


def test_placement_names_the_object_relation_and_destination_of_placing_steps_only():
    assert placement("place the chicken drumstick in the freezer") == ("chicken drumstick", "inside", "freezer")
    assert placement("put steak onto the plate.") == ("steak", "on", "plate")
    assert placement("close the microwave door") is None
