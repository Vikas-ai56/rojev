"""State owner + S2 decision worker with the runtime, loop and S1 runner (stand-in robot, policy and deciders).

Unit tests drive the owner synchronously with a recording worker; the end-to-end tests run every thread live.
"""
import itertools
import threading
import time

import numpy as np
import pytest

from rojev.eval import AlwaysContinue, NativePlanner, ScriptedSequence, TruthSequence
from rojev.control import ControlLoop, ExecutionRuntime, Guard, Mode
from rojev.owner import INTERACTION, PLANNING, EpisodicLog, StateOwner
from rojev.protocol import (UNANSWERED, Capture, DecisionFailure, DecisionProposal, Deviation, Observation, Operation,
                            OpKind, Precondition, StepEvent, StepStatus, TimedChunk, Transition, UserInstruction)
from rojev.robot import KEEP, ActionDim, ActionSpec
from rojev.system1 import ActionChunk, Policy, S1Runner
from rojev.system2 import DecisionWorker

SPEC = ActionSpec("stand-in", (ActionDim("v", -100, 100, 0.0),
                               ActionDim("gripper", -1, 1, KEEP, threshold=0.5, irreversible=True)), control_hz=50)
OBS = Observation({}, np.zeros(1, np.float32), "put the cube in A")
A = Operation(OpKind.SWITCH_TO, subtask="pick the cube")
B = Operation(OpKind.SWITCH_TO, subtask="place the cube in A")


class Robot:
    def observe(self):
        return OBS

    def step(self, action):
        return OBS


class StandInPolicy(Policy):
    """Records what it was asked (instruction, subtask); returns 20 actions of value 1 after `latency_s`."""

    def __init__(self, latency_s=0.02):
        self.latency_s, self.calls = latency_s, []

    def act(self, obs, subtask, timeout_s=None):
        self.calls.append((obs.task, subtask))
        time.sleep(self.latency_s)
        return ActionChunk(np.tile([1.0, -1.0], (20, 1)).astype(np.float32))

    def reset(self):
        pass


class RecordingWorker:
    busy = False

    def __init__(self):
        self.requests = []

    def request(self, snapshot):
        self.requests.append(snapshot)


def unit_owner(interaction=False, **kw):
    runtime = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, -1.0]), blend_s=0.0)
    clock = {"t": 0.0}
    captures = itertools.count(1)
    worker = RecordingWorker()
    owner = StateOwner(runtime, Guard(runtime), worker, lambda: Capture(next(captures), clock["t"], OBS),
                       lambda: clock["t"], "put the cube in A", interaction=RecordingWorker() if interaction else None,
                       **kw)
    return owner, runtime, worker, clock


def ask(owner, trigger="watchdog"):
    owner.request_decision(trigger)
    return owner.workers[PLANNING].requests[-1].request_id


def proposal(owner, op, request_id=None, rev=None, base="desired", pre=(), expires=10.0, epoch=1, decider="test",
             task=None):
    """A proposal answering `request_id` (default: a fresh planning request); `task` = new task text it adopts."""
    request_id = ask(owner) if request_id is None else request_id
    base_cmd = (owner.desired.cmd_id if owner.desired else None) if base == "desired" else base
    return DecisionProposal(epoch, request_id, 1, 0.0, owner.instruction_rev if rev is None else rev, base_cmd, 0, op,
                            {}, pre, expires, {"decider": decider, "reason": "test"}, task)


def run_command(owner, runtime, clock, cmd, value=1.0, n=50):
    """Feed command `cmd` a chunk at the current time and apply its first action (it becomes active)."""
    runtime.submit(TimedChunk(1, cmd, 1, clock["t"], SPEC.dt, np.tile([value, -1.0], (n, 1))), clock["t"])
    runtime.tick(clock["t"])
    owner.tick()


def test_valid_proposal_becomes_a_command_with_the_current_instruction():
    owner, runtime, _, _ = unit_owner()
    owner.handle(proposal(owner, A))
    assert owner.desired.cmd_id == 1 and owner.desired.instruction == "put the cube in A"
    assert runtime.pending.operation == A


def test_a_driven_owner_handles_every_posted_event_on_step():
    owner, runtime, _, _ = unit_owner()
    owner.post(proposal(owner, A))
    assert owner.has_events and owner.desired is None
    owner.step()
    assert not owner.has_events and owner.desired.operation == A


def test_a_worker_posts_its_answer_before_it_stops_being_busy():
    seen = []
    clock = {"t": 0.0}
    worker = DecisionWorker(AlwaysContinue(), lambda p: seen.append(worker.busy), lambda: clock["t"])
    owner, _, _, _ = unit_owner()
    ask(owner)
    worker.start()
    try:
        worker.request(owner.workers[PLANNING].requests[-1])
        deadline = time.perf_counter() + 2.0
        while worker.busy and time.perf_counter() < deadline:
            time.sleep(0.001)
    finally:
        worker.stop()
    assert seen == [True] and not worker.busy


def test_stale_proposals_are_refused_with_a_reason():
    owner, runtime, _, _ = unit_owner()
    old, new = ask(owner), ask(owner)
    owner.handle(proposal(owner, A, new))
    owner.handle(proposal(owner, B, old))
    owner.handle(proposal(owner, B, 999))
    owner.handle(proposal(owner, B, rev=7))
    owner.handle(proposal(owner, B, base=None))
    owner.handle(proposal(owner, B, expires=-1.0))
    owner.handle(proposal(owner, B, epoch=0))
    owner.handle(proposal(owner, B, pre=(Precondition("holding", "cube_1"),)))
    whys = [e["why"] for e in owner.record.events if e["kind"] == "refused"]
    assert len(whys) == 7 and owner.desired.operation == A
    for reason in ("not newer", "not made by this owner", "instruction rev", "based on cmd", "expired", "epoch",
                   "preconditions"):
        assert any(reason in w for w in whys), reason


def test_revoke_mode_stops_at_once_and_asks_for_a_decision():
    owner, runtime, worker, clock = unit_owner(on_instruction_change="revoke")
    owner.handle(proposal(owner, A))
    runtime.submit(TimedChunk(1, 1, 1, 0.0, SPEC.dt, np.tile([1.0, -1.0], (10, 1))), 0.0)
    runtime.tick(0.0)
    clock["t"] = 0.05
    owner.handle(UserInstruction("actually, put it in B"))
    assert runtime.tick(0.06).source == "hold" and owner.instruction_rev == 1 and owner.desired is None
    assert worker.requests[-1].trigger == "instruction"
    assert worker.requests[-1].latest_utterance == "actually, put it in B"
    assert worker.requests[-1].instruction == "put the cube in A"
    owner.handle(proposal(owner, B, worker.requests[-1].request_id - 1, rev=0))
    assert owner.desired is None


def test_continue_renews_the_lease_and_same_operation_is_not_a_new_command():
    owner, runtime, _, clock = unit_owner(lease_s=1.0)
    owner.handle(proposal(owner, A))
    clock["t"] = 0.8
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE)))
    owner.handle(proposal(owner, A))
    assert owner.desired.cmd_id == 1 and runtime.pending.expires_t == 1.8


def test_a_rejected_chunk_keeps_the_command_a_rejected_command_asks_again():
    owner, runtime, worker, clock = unit_owner()
    owner.handle(proposal(owner, A))
    runtime.submit(TimedChunk(1, 1, 1, -5.0, SPEC.dt, np.tile([1.0, -1.0], (3, 1))), 0.0)
    owner.tick()
    assert owner.desired is not None and owner.desired.cmd_id == 1
    runtime.stop(0.0, "test")
    owner.handle(proposal(owner, B))
    n = len(worker.requests)
    owner.tick()
    assert owner.desired is None and len(worker.requests) > n


def test_complete_holds_and_stops_asking():
    owner, runtime, worker, clock = unit_owner(watchdog_s=0.1)
    owner.handle(proposal(owner, Operation(OpKind.COMPLETE)))
    n = len(worker.requests)
    clock["t"] = 1.0
    owner.tick()
    assert owner.complete and runtime.mode == Mode.HOLD and len(worker.requests) == n


def test_a_deviation_of_the_active_command_asks_for_a_decision_with_it():
    owner, runtime, worker, clock = unit_owner()
    owner.handle(proposal(owner, A))
    runtime.submit(TimedChunk(1, 1, 1, 0.0, SPEC.dt, np.tile([1.0, -1.0], (10, 1))), 0.0)
    runtime.tick(0.0)
    owner.tick()
    d = Deviation(1, 1, 0.02, score=5.0, threshold=2.0)
    owner.handle(d)
    assert worker.requests[-1].trigger == "deviation" and worker.requests[-1].deviation == d
    owner.handle(proposal(owner, B, worker.requests[-1].request_id))
    runtime.submit(TimedChunk(1, 2, 2, 0.04, SPEC.dt, np.tile([2.0, -1.0], (10, 1))), 0.04)
    runtime.tick(0.04)
    owner.tick()
    assert owner.active_cmd == 2 and owner.deviation is None


def test_stale_deviations_are_logged_not_acted_on():
    owner, runtime, worker, clock = unit_owner()
    owner.handle(proposal(owner, A))
    n = len(worker.requests)
    owner.handle(Deviation(1, 1, 0.0, 5.0, 2.0))
    owner.handle(Deviation(0, None, 0.0, 5.0, 2.0))
    assert len(worker.requests) == n and owner.deviation is None
    assert sum(e["kind"] == "stale deviation" for e in owner.record.events) == 2


def live(decider, policy, seconds, instruction="put the cube in A", events=(), interaction=None, **owner_kw):
    runtime = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, -1.0]), blend_s=0.0)
    guard = Guard(runtime)
    ticks = []
    loop = ControlLoop(Robot(), guard, SPEC, on_tick=lambda r: ticks.append(r.tick))
    runner = S1Runner(runtime, policy, loop.latest, loop.now)
    owner = None
    worker = DecisionWorker(decider, lambda p: owner.post(p), loop.now, name="planning")
    responder = (DecisionWorker(interaction, lambda p: owner.post(p), loop.now, name="interaction")
                 if interaction is not None else None)
    owner = StateOwner(runtime, guard, worker, loop.latest, loop.now, instruction, interaction=responder, **owner_kw)
    threads = (loop, runner, worker, owner) + ((responder,) if responder is not None else ())
    for th in threads:
        th.start()
    w0 = time.perf_counter()
    for at, event in events:
        time.sleep(max(0.0, at - (time.perf_counter() - w0)))
        owner.post(event)
    time.sleep(max(0.0, seconds - (time.perf_counter() - w0)))
    for th in reversed(threads):
        th.stop()
    return owner, runtime, ticks, worker


def test_scripted_arm_runs_its_sequence_live_then_completes():
    policy = StandInPolicy()
    owner, runtime, ticks, worker = live(ScriptedSequence(["pick the cube", "place the cube in A"], 0.3), policy, 1.5,
                                         watchdog_s=0.05)
    commands = [e["subtask"] for e in owner.record.events if e["kind"] == "command"]
    assert commands[:2] == ["pick the cube", "place the cube in A"] and owner.complete
    assert {s for _, s in policy.calls} == {"pick the cube", "place the cube in A"}
    assert runtime.mode == Mode.HOLD and not any(t.hold_reason == "owner lost" for t in ticks)
    assert worker.stats.errors == 0


def test_revoke_mode_live_holds_then_conditions_s1_on_the_new_instruction():
    policy = StandInPolicy()
    owner, runtime, ticks, _ = live(NativePlanner(), policy, 1.2,
                                    events=[(0.5, UserInstruction("actually, put it in B"))], watchdog_s=0.2,
                                    on_instruction_change="revoke")
    kinds = [e["kind"] for e in owner.record.events]
    assert "revoke" in kinds and owner.instruction_rev == 1
    revoke = next(e for e in owner.record.events if e["kind"] == "revoke")
    after = [t for t in ticks if t.t >= revoke["t"]]
    assert after[0].source == "hold"
    assert all(t.cmd_id != revoke["cmd"] for t in after if t.source == "chunk")
    assert any(t.source == "chunk" for t in after)
    tasks = [task for task, _ in policy.calls]
    assert tasks[0] == "put the cube in A" and tasks[-1] == "actually, put it in B"
    assert all(sub == "" for _, sub in policy.calls)
    statuses = [e["status"] for e in owner.record.events if e["kind"] == "ack"]
    assert statuses.count("activated") == 2 and "revoked" in statuses


def test_oracle_arm_follows_truth_live():
    truth = {"picked": False}
    policy = StandInPolicy()

    def flip():
        time.sleep(0.4)
        truth["picked"] = True

    threading.Thread(target=flip, daemon=True).start()
    owner, _, _, _ = live(TruthSequence(["pick the cube", "place the cube in A"],
                                        lambda i: truth["picked"] if i == 0 else False), policy, 1.0,
                          watchdog_s=0.05)
    commands = [e["subtask"] for e in owner.record.events if e["kind"] == "command"]
    assert commands == ["pick the cube", "place the cube in A"]


def test_the_owner_keeps_a_long_task_alive_past_the_lease_without_asking_anyone():
    policy = StandInPolicy()
    owner, runtime, ticks, worker = live(NativePlanner(), policy, 1.5, lease_s=0.5, watchdog_s=100.0)
    assert worker.stats.decisions == 1
    statuses = [e["status"] for e in owner.record.events if e["kind"] == "ack"]
    assert "expired" not in statuses and runtime.mode == Mode.RUNNING
    assert sum(t.source == "chunk" for t in ticks) > 0.8 * len(ticks)


def test_review_keeps_the_command_running_and_freezes_irreversible_switches():
    owner, runtime, worker, clock = unit_owner(review_s=0.5)
    owner.handle(proposal(owner, A))
    runtime.submit(TimedChunk(1, 1, 1, 0.0, SPEC.dt, np.tile([1.0, -1.0], (10, 1))), 0.0)
    runtime.tick(0.0)
    clock["t"] = 0.05
    owner.handle(UserInstruction("actually, put it in B"))
    assert [e["kind"] for e in owner.record.events].count("revoke") == 0
    assert owner.reviewing == 1 and runtime.under_review() and runtime.active.expires_t == 0.55
    assert worker.requests[-1].review and worker.requests[-1].trigger == "instruction"
    r = runtime.tick(0.06)
    assert r.source == "chunk" and r.review


def test_an_utterance_never_becomes_the_task_text_and_a_plain_continue_only_renews():
    owner, runtime, worker, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("wait, nice work"))
    assert owner.instruction == "put the cube in A" and worker.requests[-1].latest_utterance == "wait, nice work"
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), worker.requests[-1].request_id))
    assert owner.desired.cmd_id == 1 and owner.desired.instruction == "put the cube in A"
    assert not any(e["kind"] == "reaffirm" for e in owner.record.events)


def test_a_decision_adopting_new_task_text_reaffirms_without_a_pause():
    owner, runtime, worker, clock = unit_owner()
    owner.handle(proposal(owner, A))
    runtime.submit(TimedChunk(1, 1, 1, 0.0, SPEC.dt, np.tile([1.0, -1.0], (50, 1))), 0.0)
    runtime.tick(0.0)
    clock["t"] = 0.05
    owner.handle(UserInstruction("actually, put it in B"))
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), worker.requests[-1].request_id,
                          task="actually, put it in B"))
    assert owner.desired.cmd_id == 2 and owner.desired.operation == A
    assert owner.desired.instruction == "actually, put it in B" and owner.desired.instruction_rev == 1
    assert runtime.tick(0.08).source == "chunk"
    assert any(e["kind"] == "reaffirm" for e in owner.record.events)


def test_review_without_a_decision_expires_into_a_hold():
    owner, runtime, worker, clock = unit_owner(review_s=0.2)
    owner.handle(proposal(owner, A))
    runtime.submit(TimedChunk(1, 1, 1, 0.0, SPEC.dt, np.tile([1.0, -1.0], (50, 1))), 0.0)
    runtime.tick(0.0)
    owner.handle(UserInstruction("actually, put it in B"))
    rs = [runtime.tick(0.02 * k) for k in range(1, 15)]
    assert rs[-1].source == "hold" and rs[-1].hold_reason == "lease expired"
    owner.tick()
    assert owner.desired is None and worker.requests[-1].trigger == "cmd 1 expired"


def test_unknown_instruction_change_mode_is_refused():
    import pytest
    with pytest.raises(ValueError, match="on_instruction_change"):
        unit_owner(on_instruction_change="ignore")


def test_review_mode_live_never_pauses_and_conditions_s1_on_the_new_instruction():
    policy = StandInPolicy()
    owner, runtime, ticks, _ = live(NativePlanner(), policy, 1.2,
                                    events=[(0.5, UserInstruction("actually, put it in B"))], watchdog_s=0.2)
    kinds = [e["kind"] for e in owner.record.events]
    assert "review" in kinds and "reaffirm" in kinds and "revoke" not in kinds
    review_t = next(e["t"] for e in owner.record.events if e["kind"] == "review")
    after = [t for t in ticks if t.t >= review_t]
    assert all(t.source == "chunk" for t in after[:-2])
    assert policy.calls[-1][0] == "actually, put it in B"


def interaction_request(owner):
    return owner.workers[INTERACTION].requests[-1]


def test_change_triggers_go_to_the_interaction_lane_routine_ones_to_planning():
    owner, runtime, planning, clock = unit_owner(interaction=True)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("actually, put it in B"))
    assert interaction_request(owner).trigger == "instruction" and owner.interaction_pending
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), interaction_request(owner).request_id))
    owner.handle(Deviation(1, owner.active_cmd, 0.1, 5.0, 2.0))
    assert interaction_request(owner).trigger == "deviation"
    assert all(r.trigger not in ("instruction", "deviation") for r in planning.requests)


def test_while_an_interaction_decision_is_pending_planning_waits():
    owner, runtime, planning, clock = unit_owner(interaction=True, watchdog_s=0.1)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    stale_plan = ask(owner)
    owner.handle(UserInstruction("actually, put it in B"))
    n = len(planning.requests)
    clock["t"] = 0.5
    owner.tick()
    owner.request_decision("cmd 9 expired")
    assert len(planning.requests) == n
    assert any(e["kind"] == "planning deferred" for e in owner.record.events)
    owner.handle(proposal(owner, B, stale_plan))
    assert owner.desired.operation == A
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), interaction_request(owner).request_id,
                          task="actually, put it in B"))
    assert not owner.interaction_pending and owner.desired.instruction == "actually, put it in B"
    assert len(planning.requests) == n + 1 and planning.requests[-1].trigger == "cmd 9 expired"


def test_a_newer_change_supersedes_an_older_interaction_request():
    owner, runtime, planning, clock = unit_owner(interaction=True)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(Deviation(1, 1, 0.1, 5.0, 2.0))
    first = interaction_request(owner).request_id
    owner.handle(Deviation(1, 1, 0.2, 6.0, 2.0))
    second = interaction_request(owner).request_id
    owner.handle(proposal(owner, B, first))
    assert owner.desired.operation == A and owner.interaction_pending
    owner.handle(proposal(owner, B, second))
    assert owner.desired.operation == B and not owner.interaction_pending


@pytest.mark.parametrize("release", ["failure", "timeout", "refused"])
def test_an_unanswered_change_holds_the_robot_and_never_reaches_planning(release):
    said = []
    owner, runtime, planning, clock = unit_owner(interaction=True, interaction_timeout_s=0.3, say=said.append)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    n = len(planning.requests)
    owner.handle(UserInstruction("stop"))
    request = interaction_request(owner)
    if release == "failure":
        owner.handle(DecisionFailure(1, request.request_id, "RuntimeError: model down"))
    elif release == "timeout":
        clock["t"] = 0.29
        owner.tick()
        assert owner.interaction_pending
        clock["t"] = 0.3
        owner.tick()
    else:
        owner.handle(proposal(owner, B, request.request_id, expires=-1.0))
    assert not owner.interaction_pending and len(planning.requests) == n
    assert owner.desired.operation == Operation(OpKind.GUARDED_WAIT, template=UNANSWERED)
    assert owner.hold is not None and owner.hold.interrupted == A and said[-1].template == UNANSWERED
    assert runtime.active.operation.kind == OpKind.GUARDED_WAIT
    owner.tick()
    assert owner.desired.operation.template == UNANSWERED


def test_an_unanswered_change_is_asked_again_on_the_hold_recheck_and_then_answered():
    owner, runtime, planning, clock = unit_owner(interaction=True, hold_recheck_s=1.0)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("go to B"))
    owner.handle(DecisionFailure(1, interaction_request(owner).request_id, "boom"))
    clock["t"] = 1.0
    owner.tick()
    retry = interaction_request(owner)
    assert retry.trigger == "hold" and retry.hold.operation.template == UNANSWERED
    owner.handle(proposal(owner, B, retry.request_id))
    assert owner.hold is None and owner.desired.operation == B


def timed_out(owner, runtime, clock, text, **kw):
    """Command A running, the user says `text`, the interaction deadline passes: the owner's placeholder hold stands.
    Returns the interaction request that is still awaited."""
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction(text))
    request = interaction_request(owner)
    clock["t"] = 0.3
    owner.tick()
    assert owner.desired.operation == Operation(OpKind.GUARDED_WAIT, template=UNANSWERED)
    return request


def test_a_late_answer_is_applied_to_the_situation_it_was_decided_on():
    """Live 2026-10-23: OneJev answered in 1.1–1.5 s against a 1.0 s deadline; the answer was refused (its base was the
    owner's own placeholder hold) and the change decided again. Now a late "wait" stands, and a late "keep going"
    resumes the command the placeholder stopped."""
    for op, expected in ((Operation(OpKind.GUARDED_WAIT), Operation(OpKind.GUARDED_WAIT)),
                         (Operation(OpKind.CONTINUE), A)):
        owner, runtime, planning, clock = unit_owner(interaction=True, interaction_timeout_s=0.3)
        request = timed_out(owner, runtime, clock, "slow down a bit")
        owner.handle(proposal(owner, op, request.request_id, base=request.active_cmd))
        assert owner.desired.operation == expected and any(e["kind"] == "late answer" for e in owner.record.events)
        assert (owner.hold is None) == expected.moves_robot()
        assert owner.hold is None or owner.hold.interrupted == A


def test_an_awaited_answer_is_not_asked_for_again_and_a_newer_change_ends_the_wait():
    owner, runtime, planning, clock = unit_owner(interaction=True, interaction_timeout_s=0.3, hold_recheck_s=1.0)
    request = timed_out(owner, runtime, clock, "wait")
    clock["t"] = 2.0
    owner.tick()
    assert interaction_request(owner) is request
    owner.handle(UserInstruction("no, put it in B"))
    newer = interaction_request(owner)
    owner.handle(proposal(owner, Operation(OpKind.GUARDED_WAIT), request.request_id, rev=request.instruction_rev,
                          base=request.active_cmd))
    assert owner.desired.operation.template == UNANSWERED
    owner.handle(proposal(owner, B, newer.request_id))
    assert owner.desired.operation == B and owner.hold is None


def test_a_repeated_question_answering_the_user_is_said_again():
    """The robot is already asking; the user answers and the interaction lane asks again: the same operation, but a
    question is an event, so it is said again (live 2026-10-23: the second question was silent)."""
    said = []
    owner, runtime, planning, clock = unit_owner(interaction=True, say=said.append)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    question = Operation(OpKind.ASK, template="clarify")
    owner.handle(UserInstruction("wait on the microwave"))
    owner.handle(proposal(owner, question, interaction_request(owner).request_id))
    owner.handle(UserInstruction("the apple on the microwave"))
    owner.handle(proposal(owner, question, interaction_request(owner).request_id))
    assert [u.template for u in said if u.template == "clarify"] == ["clarify", "clarify"]
    assert owner.desired.operation == question


def test_without_an_interaction_worker_every_trigger_goes_to_planning():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("actually, put it in B"))
    assert planning.requests[-1].trigger == "instruction" and not owner.interaction_pending


def test_moving_past_a_running_step_records_it_done_in_memory_with_the_decisions_observation():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    clock["t"] = 0.5
    request = ask(owner)
    seen = planning.requests[-1].capture
    owner.handle(proposal(owner, B, request, decider="oracle"))
    assert [(k.cmd_id, k.step) for k in owner.memory.records] == [(1, "pick the cube")]
    assert owner.memory.records[0].capture is seen and owner.memory.records[0].evidence.startswith("decider oracle")
    assert planning.requests[-1].memory.records == ()
    ask(owner)
    assert planning.requests[-1].memory.records == tuple(owner.memory.records)
    run_command(owner, runtime, clock, 2, value=2.0)
    owner.handle(proposal(owner, Operation(OpKind.COMPLETE)))
    assert [k.step for k in owner.memory.records] == ["pick the cube", "place the cube in A"]


def test_aborts_waits_retries_and_never_started_steps_are_not_progress():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    owner.handle(proposal(owner, B))
    assert owner.memory.records == ()
    run_command(owner, runtime, clock, 2)
    for op in (Operation(OpKind.RETRY, subtask="place the cube in A"), Operation(OpKind.GUARDED_WAIT),
               Operation(OpKind.ABORT)):
        owner.handle(proposal(owner, op))
    assert owner.memory.records == ()


def test_memory_survives_a_deviation_and_an_instruction_change():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(proposal(owner, B))
    run_command(owner, runtime, clock, 2, value=2.0)
    owner.handle(Deviation(1, 2, 0.1, 5.0, 2.0))
    assert [k.step for k in planning.requests[-1].memory.records] == ["pick the cube"]
    owner.handle(UserInstruction("actually, put it in B"))
    assert [k.step for k in planning.requests[-1].memory.records] == ["pick the cube"]
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), planning.requests[-1].request_id,
                          task="actually, put it in B"))
    run_command(owner, runtime, clock, 3, value=2.0)
    owner.handle(proposal(owner, Operation(OpKind.COMPLETE)))
    assert [(k.step, k.instruction_rev, k.cmd_id) for k in owner.memory.records] == [
        ("pick the cube", 0, 1), ("place the cube in A", 1, 3)]


def test_a_failing_decider_posts_a_decision_failure():
    class Broken:
        name = "broken"

        def decide(self, snapshot):
            raise RuntimeError("model down")

    owner, runtime, planning, clock = unit_owner()
    owner.request_decision("watchdog")
    out = DecisionWorker(Broken(), lambda p: None, lambda: 0.0).decide(planning.requests[-1])
    assert isinstance(out, DecisionFailure) and out.request_id == planning.requests[-1].request_id
    assert "model down" in out.error


def test_two_lanes_live_answer_a_change_of_mind_from_the_interaction_lane_without_a_pause():
    policy = StandInPolicy()
    owner, runtime, ticks, _ = live(NativePlanner(), policy, 1.2, interaction=AlwaysContinue(),
                                    events=[(0.5, UserInstruction("actually, put it in B"))], watchdog_s=0.1)
    events = owner.record.events
    change = next(i for i, e in enumerate(events) if e["kind"] == "instruction")
    reaffirm = next(i for i, e in enumerate(events) if e["kind"] == "reaffirm")
    assert events[reaffirm]["lane"] == INTERACTION and events[reaffirm]["decider"] == "always-continue"
    assert not [e for e in events[change:reaffirm] if e["kind"] in ("command", "continue") and e["lane"] == PLANNING]
    review_t = next(e["t"] for e in events if e["kind"] == "review")
    assert all(t.source == "chunk" for t in [t for t in ticks if t.t >= review_t][:-2])
    assert policy.calls[-1][0] == "actually, put it in B" and not owner.interaction_pending


def interaction_answer(owner, op):
    """The interaction lane's proposal for its pending request."""
    return proposal(owner, op, owner.workers[INTERACTION].requests[-1].request_id)


def test_a_stop_holds_until_a_change_and_the_planner_cannot_restart_the_task():
    owner, runtime, planning, clock = unit_owner(interaction=True, watchdog_s=0.1)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("stop"))
    owner.handle(interaction_answer(owner, Operation(OpKind.ABORT)))
    owner.tick()
    assert owner.hold.operation.kind == OpKind.ABORT and owner.hold.interrupted == A
    assert runtime.active.operation.kind == OpKind.ABORT
    n = len(planning.requests)
    for t in (1.0, 2.0, 5.0, 9.0):
        clock["t"] = t
        owner.tick()
        runtime.tick(t)
    assert len(planning.requests) == n and runtime.active is not None and runtime.active.operation.kind == OpKind.ABORT
    owner.handle(UserInstruction("go on"))
    snap = owner.workers[INTERACTION].requests[-1]
    assert snap.hold.interrupted == A and snap.trigger == "instruction"
    owner.handle(interaction_answer(owner, snap.hold.interrupted))
    assert owner.hold is None and owner.desired.operation == A
    assert any(e["kind"] == "hold released" for e in owner.record.events)


def test_a_hold_is_rechecked_by_the_interaction_lane_on_its_wait_budget():
    owner, runtime, _, clock = unit_owner(interaction=True, hold_recheck_s=2.0)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("wait"))
    owner.handle(interaction_answer(owner, Operation(OpKind.GUARDED_WAIT)))
    clock["t"] = 1.0
    owner.tick()
    assert owner.workers[INTERACTION].requests[-1].trigger == "instruction"
    clock["t"] = 2.5
    owner.tick()
    snap = owner.workers[INTERACTION].requests[-1]
    assert snap.trigger == "hold" and snap.hold.operation.kind == OpKind.GUARDED_WAIT


def test_instructions_are_acknowledged_at_once_and_questions_join_the_dialogue():
    said = []
    owner, runtime, _, clock = unit_owner(interaction=True, say=said.append, dialogue_turns=3)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("put it over there"))
    assert said[-1].template == "ack" and said[-1].cmd_id is None
    question = Operation(OpKind.ASK, template="which_destination", objects=("plate", "bowl"))
    owner.handle(interaction_answer(owner, question))
    assert said[-1].template == "which_destination" and said[-1].objects == ("plate", "bowl")
    owner.handle(UserInstruction("the plate"))
    snap = owner.workers[INTERACTION].requests[-1]
    assert [(t.speaker, t.text or t.template) for t in snap.dialogue] == [
        ("user", "put it over there"), ("robot", "which_destination"), ("user", "the plate")]
    owner.handle(UserInstruction("no, the bowl"))
    assert len(owner.workers[INTERACTION].requests[-1].dialogue) == 3


def test_a_step_event_asks_planning_at_once_and_becomes_the_records_evidence():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(StepEvent(1, 7, 0.1, "mjev", "pick the cube", {"done": 0.9}))
    assert any(e["kind"] == "stale step event" for e in owner.record.events)
    owner.handle(StepEvent(1, 1, 0.1, "mjev", "pick the cube", {"done": 0.9}))
    snap = planning.requests[-1]
    assert snap.trigger == "step" and snap.step_event.source == "mjev"
    owner.handle(proposal(owner, B, snap.request_id))
    assert owner.memory.records[-1].evidence == "mjev: {'done': 0.9}"


def test_a_dropped_command_is_replaced_by_a_new_decision():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    runtime.hold(clock["t"], "owner lost", drop_commands=True)
    owner.tick()
    assert owner.active_cmd is None and owner.desired is None
    assert planning.requests[-1].trigger == "cmd 1 dropped"
    owner.handle(proposal(owner, A, planning.requests[-1].request_id))
    run_command(owner, runtime, clock, 2)
    assert owner.active_cmd == 2 and runtime.mode == Mode.RUNNING


def test_a_reaffirm_never_pauses_even_when_switches_pause():
    owner, runtime, _, clock = unit_owner(interaction=True, transition=Transition.PAUSE_HOLD)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("carry on, carefully"))
    request = owner.workers[INTERACTION].requests[-1].request_id
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), request, task="carry on, carefully"))
    reaffirm = next(e for e in owner.record.events if e["kind"] == "reaffirm")
    assert reaffirm["transition"] == "immediate"


def test_a_continue_with_nothing_to_renew_says_so():
    owner, _, _, _ = unit_owner()
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE)))
    event = next(e for e in owner.record.events if e["kind"] == "continue")
    assert event["renewed"] is False and event["cmd"] is None


def test_one_lane_a_newer_routine_proposal_cannot_undo_a_stop():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("stop"))
    stop_request = planning.requests[-1].request_id
    routine = ask(owner)
    owner.handle(proposal(owner, Operation(OpKind.ABORT), stop_request))
    owner.handle(proposal(owner, Operation(OpKind.SWITCH_TO), routine))
    assert owner.hold is not None and owner.desired.operation.kind == OpKind.ABORT
    assert any("held by abort" in e.get("why", "") for e in owner.record.events)


def test_a_step_finishing_while_a_change_is_pending_is_planned_after_the_change():
    owner, runtime, planning, clock = unit_owner(interaction=True)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("put it in B instead"))
    n = len(planning.requests)
    owner.handle(StepEvent(1, 1, 0.1, "mjev", "pick the cube", {"done": 0.9}))
    assert len(planning.requests) == n and any(e["kind"] == "planning deferred" for e in owner.record.events)
    owner.handle(interaction_answer(owner, Operation(OpKind.CONTINUE)))
    after = planning.requests[-1]
    assert after.trigger == "step" and after.instruction_rev == 1 and after.step_event.source == "mjev"


def test_when_s1_plans_its_own_steps_only_a_checked_step_event_is_recorded_done():
    owner, runtime, planning, clock = unit_owner()
    native = Operation(OpKind.SWITCH_TO)
    owner.handle(proposal(owner, native))
    run_command(owner, runtime, clock, 1)
    owner.handle(StepEvent(1, 1, 0.5, "s1-planner", "pick the cube", {"next": "place the cube"}))
    assert owner.memory.records == () and planning.requests[-1].trigger == "step"
    owner.handle(StepEvent(1, 1, 0.6, "mjev", "pick the cube", {"p_done": 0.9, "claimed_by": "s1-planner"},
                           checked=True))
    assert [(k.step, k.evidence) for k in owner.memory.records] == [
        ("pick the cube", "mjev: {'p_done': 0.9, 'claimed_by': 's1-planner'}")]



def test_the_review_window_lasts_until_the_interaction_answer_is_due():
    owner, runtime, _, clock = unit_owner(interaction=True, interaction_timeout_s=2.2, review_s=1.0)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(UserInstruction("wait"))
    assert owner.review_window_s == 2.2 and owner.desired.expires_t == pytest.approx(2.2)
    clock["t"] = 1.5
    runtime.tick(1.5)
    owner.tick()
    owner.handle(interaction_answer(owner, Operation(OpKind.GUARDED_WAIT)))
    assert owner.desired.operation.kind == OpKind.GUARDED_WAIT and owner.hold is not None


def test_planning_is_asked_at_the_start_on_an_overdue_step_and_by_the_watchdog_not_on_a_beat():
    owner, runtime, planning, clock = unit_owner(watchdog_s=10.0, step_budget_s=3.0)
    owner.tick()
    assert [r.trigger for r in planning.requests] == ["start"]
    owner.handle(proposal(owner, A, planning.requests[-1].request_id))
    run_command(owner, runtime, clock, 1)
    for t in (0.5, 1.0, 2.0, 2.9):
        clock["t"] = t
        owner.tick()
    assert [r.trigger for r in planning.requests] == ["start"]
    clock["t"] = 3.1
    owner.tick()
    clock["t"] = 3.2
    owner.tick()
    assert [r.trigger for r in planning.requests] == ["start", "overdue"]
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), planning.requests[-1].request_id))
    clock["t"] = 6.0
    owner.tick()
    assert planning.requests[-1].trigger == "overdue" and len(planning.requests) == 2
    clock["t"] = 6.2
    owner.tick()
    assert len(planning.requests) == 3
    owner, _, planning, clock = unit_owner(watchdog_s=10.0)
    owner.tick()
    clock["t"] = 9.9
    owner.tick()
    assert len(planning.requests) == 1
    clock["t"] = 10.1
    owner.tick()
    assert [r.trigger for r in planning.requests] == ["start", "watchdog"]


def test_the_owner_renews_the_lease_but_not_under_review_and_not_once_the_planner_is_silent():
    owner, runtime, planning, clock = unit_owner(lease_s=1.0, liveness_s=2.0, watchdog_s=100.0, review_s=0.3)
    owner.tick()
    owner.handle(proposal(owner, A, planning.requests[-1].request_id))
    run_command(owner, runtime, clock, 1, n=500)
    clock["t"] = 0.8
    owner.tick()
    assert owner.desired.expires_t == 1.8
    owner.handle(UserInstruction("hmm"))
    review_end = owner.desired.expires_t
    clock["t"] = 1.0
    owner.tick()
    assert owner.desired.expires_t == review_end
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), planning.requests[-1].request_id))
    ask(owner)
    for t in (1.5, 2.5, 3.5):
        clock["t"] = t
        owner.tick()
    assert any(e["kind"] == "liveness lapsed" for e in owner.record.events)
    lapsed_at = owner.desired.expires_t
    clock["t"] = lapsed_at + 0.1
    runtime.tick(clock["t"])
    owner.tick()
    assert "expired" in [e["status"] for e in owner.record.events if e["kind"] == "ack"]


def test_choosing_a_step_recorded_done_records_it_undone_and_new_task_text_becomes_the_goal():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(proposal(owner, B))
    assert owner.memory.done() == ["pick the cube"]
    run_command(owner, runtime, clock, 2, value=2.0)
    owner.handle(proposal(owner, A, decider="planner"))
    assert [(r.step, r.status) for r in owner.memory.records][-2:] == [
        ("pick the cube", StepStatus.UNDONE), ("place the cube in A", StepStatus.DONE)]
    assert owner.memory.done() == ["place the cube in A"]
    owner.handle(proposal(owner, Operation(OpKind.CONTINUE), task="put the cube in B"))
    assert owner.memory.goal == "put the cube in B"


def test_a_finished_task_is_appended_to_the_episodic_log_once(tmp_path):
    log = EpisodicLog(tmp_path / "episodes.jsonl")
    owner, runtime, planning, clock = unit_owner(episodic=log)
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(proposal(owner, Operation(OpKind.COMPLETE)))
    owner.stop()
    [episode] = log.read()
    assert episode["outcome"] == "complete" and episode["goal"] == "put the cube in A"
    assert [(s["step"], s["status"]) for s in episode["steps"]] == [("pick the cube", "done")]
    stopped_log = EpisodicLog(tmp_path / "stopped.jsonl")
    owner, _, _, _ = unit_owner(episodic=stopped_log)
    owner.handle(proposal(owner, A))
    owner.stop()
    assert [e["outcome"] for e in stopped_log.read()] == ["stopped"]


def test_a_checked_step_event_records_the_step_done_at_once_and_moving_past_adds_no_duplicate():
    owner, runtime, planning, clock = unit_owner()
    owner.handle(proposal(owner, A))
    run_command(owner, runtime, clock, 1)
    owner.handle(StepEvent(1, 1, 0.2, "checker", "pick the cube", {"p_done": 0.9}, checked=True))
    assert planning.requests[-1].memory.done() == ["pick the cube"]
    owner.handle(proposal(owner, B, planning.requests[-1].request_id))
    assert [(r.step, r.checked) for r in owner.memory.records] == [("pick the cube", True)]
