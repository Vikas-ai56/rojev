"""Jev v1 interaction decider with a stub chooser: legal candidates, calibration, abstention, budgets, dwell, and the
operations each act becomes; then through a live owner (the D4 stop-and-resume path)."""
import numpy as np
import pytest

from rojev.control import ExecutionRuntime, Guard
from rojev.eval import NativePlanner
from rojev.owner import StateOwner
from rojev.protocol import (ROBOT, UNANSWERED, USER, Capture, Deviation, Hold, Keyframe, Observation,
                            ObservationSnapshot, Operation, OpKind, Turn, UserInstruction)
from rojev.robot import KEEP, ActionDim, ActionSpec
from rojev.system2 import DecisionWorker
from rojev.system2.jev import (ABORT, ASK, CLARIFY, CONTINUE, RESUME, RETRY, REVISE, WAIT, WAITING, Calibration,
                               JevInteraction, candidates, fit_calibration, fit_margin, fit_temperature)
from rojev.system2.jev.chooser import situation

OBS = Observation({"left": np.zeros((4, 4, 3), np.uint8), "right": np.zeros((4, 4, 3), np.uint8)},
                  np.zeros(1, np.float32), "put the cube in A")
STEP = Operation(OpKind.SWITCH_TO, subtask="place the cube in A")


class StubChooser:
    """Returns fixed probabilities and records what it was offered."""

    name = "stub"

    def __init__(self, probs):
        self.probs, self.offered = probs, []

    def choose(self, snapshot, options):
        self.offered.append([o.label for o in options])
        return {o.label: self.probs.get(o.label, 0.0) for o in options}


def snap(trigger="instruction", op=STEP, hold=None, dialogue=(), now=5.0, active_since=0.0, instruction="put it in B"):
    return ObservationSnapshot(1, 1, now, Capture(1, now, OBS), 1, instruction, 1 if op else None, op, active_since,
                               (), trigger, hold=hold, dialogue=tuple(dialogue),
                               deviation=Deviation(1, 1, now, 3.0, 2.0) if trigger == "deviation" else None)


def labels(s, max_asks=2):
    return [o.label for o in candidates(s, max_asks)]


def test_candidates_are_the_legal_acts_for_each_situation():
    assert labels(snap()) == [CONTINUE, REVISE, WAIT, ASK, ABORT]
    assert labels(snap(op=None)) == [CONTINUE, REVISE, ASK, ABORT]
    assert labels(snap(trigger="deviation")) == [CONTINUE, RETRY, REVISE, ASK]
    waiting = Hold(Operation(OpKind.GUARDED_WAIT), 1.0, STEP)
    assert labels(snap(op=Operation(OpKind.GUARDED_WAIT), hold=waiting)) == [RESUME, CONTINUE, REVISE, ASK, ABORT]
    stopped = Hold(Operation(OpKind.ABORT), 1.0, None)
    assert labels(snap(op=Operation(OpKind.ABORT), hold=stopped)) == [CONTINUE, REVISE, ASK]


def test_the_question_budget_and_no_repeat_without_new_evidence():
    asked = Turn(1.0, ROBOT, 1, template=CLARIFY)
    assert ASK not in labels(snap(dialogue=[Turn(0.5, USER, 1, "put it there"), asked]))
    assert ASK in labels(snap(dialogue=[asked, Turn(2.0, USER, 1, "there")]))
    two = [asked, Turn(2.0, USER, 1, "x"), Turn(3.0, ROBOT, 1, template=CLARIFY), Turn(4.0, USER, 1, "y")]
    assert ASK not in labels(snap(dialogue=two))


@pytest.mark.parametrize("act, kind, subtask", [(CONTINUE, OpKind.CONTINUE, None), (REVISE, OpKind.SWITCH_TO, None),
                                                (WAIT, OpKind.GUARDED_WAIT, None), (ABORT, OpKind.ABORT, None),
                                                (ASK, OpKind.ASK, None)])
def test_each_confident_act_becomes_its_operation(act, kind, subtask):
    v = JevInteraction(StubChooser({act: 0.9})).decide(snap())
    assert v.operation.kind == kind and v.operation.subtask == subtask
    assert v.scores[act] >= 0.9 and act in v.reason


def test_resume_returns_the_interrupted_step_and_retry_repeats_the_running_one():
    held = snap(op=Operation(OpKind.GUARDED_WAIT), hold=Hold(Operation(OpKind.GUARDED_WAIT), 1.0, STEP))
    assert JevInteraction(StubChooser({RESUME: 0.9})).decide(held).operation == STEP
    retry = JevInteraction(StubChooser({RETRY: 0.9})).decide(snap(trigger="deviation"))
    assert retry.operation == Operation(OpKind.RETRY, subtask=STEP.subtask)


def test_a_low_margin_abstains_to_a_question_then_to_a_wait():
    unsure = StubChooser({CONTINUE: 0.45, ABORT: 0.4, REVISE: 0.15})
    v = JevInteraction(unsure).decide(snap())
    assert v.operation == Operation(OpKind.ASK, template=CLARIFY) and "abstain" in v.reason
    after_question = snap(dialogue=[Turn(1.0, ROBOT, 1, template=CLARIFY)])
    assert JevInteraction(unsure).decide(after_question).operation.kind == OpKind.GUARDED_WAIT


def test_waiting_while_already_holding_keeps_the_hold():
    held = snap(op=Operation(OpKind.ASK, template=CLARIFY), hold=Hold(Operation(OpKind.ASK), 1.0, STEP),
                dialogue=[Turn(1.0, ROBOT, 1, template=CLARIFY)])
    v = JevInteraction(StubChooser({CONTINUE: 0.4, REVISE: 0.35, RESUME: 0.25})).decide(held)
    assert v.operation.kind == OpKind.CONTINUE and "already holding" in v.reason


def test_a_deviation_right_after_a_step_started_is_given_time():
    chooser = StubChooser({RETRY: 0.9})
    v = JevInteraction(chooser, min_dwell_s=1.0).decide(snap(trigger="deviation", now=5.0, active_since=4.5))
    assert v.operation.kind == OpKind.CONTINUE and not chooser.offered


def test_a_spent_wait_budget_reports_once_then_keeps_holding():
    jev = JevInteraction(StubChooser({}))
    waiting = Hold(Operation(OpKind.GUARDED_WAIT), 1.0, STEP)
    first = jev.decide(snap(trigger="hold", op=Operation(OpKind.GUARDED_WAIT), hold=waiting))
    assert first.operation == Operation(OpKind.GUARDED_WAIT, template=WAITING)
    reported = snap(trigger="hold", op=first.operation, hold=waiting, dialogue=[Turn(2.0, ROBOT, 1, template=WAITING)])
    assert jev.decide(reported).operation.kind == OpKind.CONTINUE
    stopped = snap(trigger="hold", op=Operation(OpKind.ABORT), hold=Hold(Operation(OpKind.ABORT), 1.0, STEP))
    assert jev.decide(stopped).operation.kind == OpKind.CONTINUE


def test_temperature_scaling_and_fit():
    sharp = Calibration(temperature=0.5).apply({"a": 0.6, "b": 0.4})
    assert sharp["a"] > 0.6 and sum(sharp.values()) == pytest.approx(1.0)
    rows = [{"a": 0.9, "b": 0.1}] * 6 + [{"a": 0.9, "b": 0.1}] * 4
    assert fit_temperature(rows, ["a"] * 6 + ["b"] * 4) > 1.0


def test_the_margin_is_the_smallest_that_reaches_the_target_accuracy():
    confident = [{"a": 0.95, "b": 0.05}] * 8
    unsure = [{"a": 0.55, "b": 0.45}] * 4
    rows, labels = confident + unsure, ["a"] * 8 + ["a", "b", "b", "b"]
    margin, coverage = fit_margin(rows, labels, 1.0, target_accuracy=0.95)
    assert 0.1 < margin <= 0.9 and coverage == 8 / 12
    assert fit_margin(rows, labels, 1.0, target_accuracy=0.5) == (0.0, 1.0)
    assert fit_margin([{"a": 0.9, "b": 0.1}] * 3, ["b"] * 3, 1.0, target_accuracy=0.9) == (1.0, 0.0)
    fitted = fit_calibration(rows, labels, target_accuracy=0.95)
    assert fitted.fitted and fitted.cases == 12 and fitted.coverage == 8 / 12
    assert Calibration.from_json(fitted.to_json()) == fitted


def test_the_model_sees_instruction_step_progress_dialogue_and_hold():
    s = ObservationSnapshot(1, 1, 5.0, Capture(1, 5.0, OBS), 1, "wait", 2, Operation(OpKind.GUARDED_WAIT), 4.0,
                            (Keyframe(1, "pick the cube", 0, Capture(1, 1.0, OBS), "test"),), "instruction",
                            hold=Hold(Operation(OpKind.GUARDED_WAIT), 4.0, STEP),
                            dialogue=(Turn(4.0, USER, 1, "wait"), Turn(4.1, ROBOT, 1, template=WAITING)))
    state = situation(s, ["left", "right"])
    assert state["steps_done"] == ["pick the cube"] and state["holding"]["interrupted_step"] == STEP.subtask
    assert state["conversation"][0] == {"user": "wait"} and "waiting" in state["conversation"][1]["robot"]


class Recorder:
    busy = False

    def __init__(self):
        self.requests = []

    def request(self, snapshot):
        self.requests.append(snapshot)


class WordChooser:
    """Stand-in for OneJev: "stop" → abort, "go on" → resume, anything else → continue."""

    name = "words"

    def choose(self, snapshot, options):
        text = snapshot.latest_utterance
        best = ABORT if "stop" in text else RESUME if "go on" in text else CONTINUE
        return {o.label: (0.9 if o.label == best else 0.1 / (len(options) - 1)) for o in options}


def test_with_the_owner_a_stop_holds_through_heartbeats_and_go_on_resumes_the_step():
    spec = ActionSpec("stand-in", (ActionDim("v", -1, 1, 0.0),
                                   ActionDim("gripper", -1, 1, KEEP, threshold=0.5, irreversible=True)), control_hz=50)
    runtime = ExecutionRuntime(spec, epoch=1, last=np.array([0.0, -1.0]), blend_s=0.0)
    clock = {"t": 0.0}
    planning, interaction = Recorder(), Recorder()
    owner = StateOwner(runtime, Guard(runtime), planning, lambda: Capture(1, clock["t"], OBS), lambda: clock["t"],
                       "put the cube in A", interaction=interaction, decide_every_s=0.5)
    plan = DecisionWorker(NativePlanner(), lambda _: None, lambda: clock["t"])
    jev = DecisionWorker(JevInteraction(WordChooser()), lambda _: None, lambda: clock["t"])
    owner.tick()
    owner.handle(plan.decide(planning.requests[-1]))
    step = owner.desired.operation
    owner.handle(UserInstruction("stop"))
    owner.handle(jev.decide(interaction.requests[-1]))
    assert owner.desired.operation.kind == OpKind.ABORT and owner.hold is not None
    for t in (1.0, 2.0, 3.0, 4.0):
        clock["t"] = t
        owner.tick()
    assert owner.desired.operation.kind == OpKind.ABORT and len(planning.requests) == 1
    owner.handle(UserInstruction("go on"))
    owner.handle(jev.decide(interaction.requests[-1]))
    assert owner.hold is None and owner.desired.operation == step


def test_a_hold_taken_because_nothing_answered_is_decided_on_the_recheck():
    held = snap(trigger="hold", op=Operation(OpKind.GUARDED_WAIT, template=UNANSWERED),
                hold=Hold(Operation(OpKind.GUARDED_WAIT, template=UNANSWERED), 1.0, STEP), instruction="stop")
    chooser = StubChooser({ABORT: 0.9})
    v = JevInteraction(chooser).decide(held)
    assert v.operation.kind == OpKind.ABORT and chooser.offered[-1][0] == RESUME


def test_only_revise_adopts_new_task_text():
    said = snap(dialogue=[Turn(4.0, USER, 1, "actually, put the cube on the plate")])
    revise = JevInteraction(StubChooser({REVISE: 0.9})).decide(said)
    assert revise.operation == Operation(OpKind.SWITCH_TO) and revise.task == "actually, put the cube on the plate"
    for act in (CONTINUE, WAIT, ABORT):
        assert JevInteraction(StubChooser({act: 0.9})).decide(said).task is None


def test_onejev_sees_neutral_option_ids_and_scores_map_back_to_acts(monkeypatch):
    from types import SimpleNamespace

    from rojev.system2.jev import OneJevChooser, Option
    from rojev.system2.jev import chooser as chooser_module

    seen = {}

    def fake_system_one(client, image, state, questions, **kw):
        seen["criteria"] = questions["act"].criteria
        return SimpleNamespace(answers={"act": SimpleNamespace(probabilities={"o1": 0.2, "o2": 0.8})})

    monkeypatch.setattr(chooser_module.onejev, "system_one", fake_system_one)
    options = [Option(CONTINUE, "keep holding still"), Option(RESUME, "go on with the interrupted step")]
    scores = OneJevChooser(None, ["left", "right"]).score(list(OBS.images.values()), {}, options)
    assert set(seen["criteria"]) == {"o1", "o2"} and "continue" not in " ".join(seen["criteria"])
    assert scores == {CONTINUE: 0.2, RESUME: 0.8}
