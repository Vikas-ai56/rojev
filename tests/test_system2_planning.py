"""Commit and verify: S1's planned subtask is committed, released only by a checked step event (or the declared
bound), and a verified step is never committed again; then live through the owner."""
import numpy as np
import pytest

from rojev.control import ExecutionRuntime, Guard
from rojev.owner import StateOwner
from rojev.protocol import (Capture, Keyframe, Observation, ObservationSnapshot, Operation, OpKind, PlannedSubtask,
                            StepEvent, TimedChunk)
from rojev.robot import KEEP, ActionDim, ActionSpec
from rojev.system2 import CommitVerifyPlanner, DecisionWorker

OBS = Observation({}, np.zeros(1, np.float32), "move the steak to the freezer")
NATIVE = Operation(OpKind.SWITCH_TO)
PLACE = Operation(OpKind.SWITCH_TO, subtask="put steak in freezer")


def snap(op, planned=None, step_event=None, keyframes=(), now=5.0, since=4.0, cmd=3):
    return ObservationSnapshot(1, 1, now, Capture(1, now, OBS), 0, OBS.task, cmd, op, since, tuple(keyframes),
                               "planned", step_event=step_event,
                               planned=None if planned is None else PlannedSubtask(1, cmd, now, planned))


def verified(subtask):
    return Keyframe(2, subtask, 0, Capture(1, 1.0, OBS), "mjev: {}", checked=True)


def test_no_command_lets_s1_plan_and_a_plan_is_committed():
    p = CommitVerifyPlanner(max_commit_s=20.0)
    assert p.decide(snap(None)).operation == NATIVE
    assert p.decide(snap(NATIVE)).operation.kind == OpKind.CONTINUE
    assert p.decide(snap(NATIVE, planned="put steak in freezer")).operation == PLACE


def test_a_committed_step_is_released_only_by_a_checked_event_about_it():
    p = CommitVerifyPlanner(max_commit_s=20.0)
    unchecked = StepEvent(1, 3, 5.0, "s1-planner", "put steak in freezer")
    other = StepEvent(1, 3, 5.0, "mjev", "close the door", checked=True)
    assert p.decide(snap(PLACE, step_event=unchecked)).operation.kind == OpKind.CONTINUE
    assert p.decide(snap(PLACE, step_event=other)).operation.kind == OpKind.CONTINUE
    released = p.decide(snap(PLACE, step_event=StepEvent(1, 3, 5.0, "mjev", "put steak in freezer", checked=True)))
    assert released.operation == NATIVE and "verified" in released.reason


def test_a_verified_step_is_not_committed_again_and_the_bound_releases_an_unverifiable_one():
    p = CommitVerifyPlanner(max_commit_s=10.0)
    again = p.decide(snap(NATIVE, planned="put steak in freezer", keyframes=[verified("put steak in freezer")]))
    assert again.operation.kind == OpKind.CONTINUE and p.reentries == 1
    unverified = Keyframe(2, "put steak in freezer", 0, Capture(1, 1.0, OBS), "decider: bound", checked=False)
    assert p.decide(snap(NATIVE, planned="put steak in freezer", keyframes=[unverified])).operation == PLACE
    expired = p.decide(snap(PLACE, now=15.0, since=4.0))
    assert expired.operation == NATIVE and p.expired == 1


SPEC = ActionSpec("stand-in", (ActionDim("v", -1, 1, 0.0),
                               ActionDim("gripper", -1, 1, KEEP, threshold=0.5, irreversible=True)), control_hz=50)


class Recorder:
    busy = False

    def __init__(self):
        self.requests = []

    def request(self, snapshot):
        self.requests.append(snapshot)


def test_with_the_owner_s1s_flip_cannot_reach_the_robot_and_a_verified_step_becomes_a_checked_keyframe():
    runtime = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, -1.0]), blend_s=0.0)
    clock = {"t": 0.0}
    planning = Recorder()
    owner = StateOwner(runtime, Guard(runtime), planning, lambda: Capture(1, clock["t"], OBS), lambda: clock["t"],
                       OBS.task, decide_every_s=100.0)
    worker = DecisionWorker(CommitVerifyPlanner(max_commit_s=30.0), lambda _: None, lambda: clock["t"])

    def answer():
        owner.handle(worker.decide(planning.requests[-1]))

    def run(cmd):
        runtime.submit(TimedChunk(1, cmd, 1, clock["t"], SPEC.dt, np.tile([0.1, -1.0], (50, 1))), clock["t"])
        runtime.tick(clock["t"])
        owner.tick()

    owner.tick()
    answer()
    run(1)
    owner.handle(PlannedSubtask(1, 1, 0.1, "put steak in freezer"))
    answer()
    assert owner.desired.operation == PLACE
    run(2)
    owner.handle(PlannedSubtask(1, 1, 0.2, "put steak in fridge"))
    assert owner.desired.operation == PLACE
    owner.handle(StepEvent(1, 2, 0.3, "mjev", "put steak in freezer", {"p_done": 0.9}, checked=True))
    answer()
    assert owner.desired.operation == NATIVE
    assert [(k.subtask, k.checked) for k in owner.keyframes] == [("put steak in freezer", True)]
