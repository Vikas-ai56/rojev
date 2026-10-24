"""Execution runtime + guard on a generic robot (no simulator, no models): the authority and timing rules.

The generic robot has two velocity-like arm dimensions (hold = 0) and a gripper switch (>= 0.5 = closed) that cannot be
undone. Times are in seconds at 10 Hz.
"""
import numpy as np
import pytest

from rojev.control import ExecutionRuntime, Guard, Mode
from rojev.protocol import AcceptedGoal, AckStatus, Operation, OpKind, Revoke, TimedChunk, Transition
from rojev.robot import KEEP, ActionDim, ActionSpec

SPEC = ActionSpec("test-arm", (ActionDim("vx", -1, 1, 0.0), ActionDim("vy", -1, 1, 0.0),
                               ActionDim("gripper", -1, 1, KEEP, threshold=0.5, irreversible=True)), control_hz=10)
DT = SPEC.dt
OPEN, CLOSED = -1.0, 1.0
PICK = Operation(OpKind.SWITCH_TO, subtask="pick the cube", target="cube_1")
PLACE_B = Operation(OpKind.SWITCH_TO, subtask="place the cube in B", target="cube_1", dest="box_b")


def runtime(last=(0.0, 0.0, OPEN), **kw):
    return ExecutionRuntime(SPEC, epoch=1, last=np.array(last, dtype=float), blend_s=0.0, **kw)


def goal(cmd, op=PICK, transition=Transition.PAUSE_HOLD, expires=100.0, epoch=1):
    return AcceptedGoal(epoch, cmd, instruction_rev=0, operation=op, transition=transition, expires_t=expires)


def chunk(cmd, t0, actions, epoch=1):
    return TimedChunk(epoch, cmd, obs_id=0, t_start=t0, dt=DT, actions=np.array(actions, dtype=float))


def run(rt, t0, n):
    return [rt.tick(t0 + k * DT) for k in range(n)]


def statuses(rt):
    return [(a.cmd_id, a.status) for a in rt.drain_acks()]


def test_command_activates_only_when_its_first_action_is_applied():
    rt = runtime()
    assert rt.accept(goal(1), 0.0)
    assert rt.tick(0.0).source == "hold" and rt.mode == Mode.TRANSITION and rt.active is None
    assert rt.submit(chunk(1, 0.1, [[0.1, 0, OPEN], [0.2, 0, OPEN]]), 0.05)
    r = rt.tick(0.1)
    assert r.source == "chunk" and r.cmd_id == 1 and r.mode == Mode.RUNNING and r.action[0] == pytest.approx(0.1)
    assert statuses(rt) == [(1, AckStatus.ACTIVATED)]
    assert rt.tick(0.2).action[0] == pytest.approx(0.2)
    assert rt.tick(0.3).source == "hold"


def test_elapsed_actions_are_skipped_never_replayed():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN], [0.2, 0, OPEN], [0.3, 0, OPEN], [0.4, 0, OPEN]]), 0.21)
    r = rt.tick(0.3)
    assert r.action[0] == pytest.approx(0.4) and r.skipped == 1
    assert rt.tick(0.4).source == "hold"


def test_revoke_drops_queued_irreversible_release_immediately():
    rt = runtime(last=np.array([0.0, 0.0, CLOSED]))
    rt.accept(goal(1, PLACE_B), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, CLOSED], [0.1, 0, CLOSED], [0, 0, OPEN], [0, 0, OPEN]]), 0.0)
    run(rt, 0.0, 2)
    assert rt.revoke(Revoke(1, 1, "user: actually A"), 0.15)
    for r in run(rt, 0.2, 3):
        assert r.source == "hold" and r.action[2] == CLOSED
    assert rt.mode == Mode.HOLD and not rt.submit(chunk(1, 0.5, [[0, 0, OPEN]]), 0.45)
    assert (1, AckStatus.REVOKED) in statuses(rt)


def test_irreversible_switch_only_changes_while_running_a_chunk():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0, 0, CLOSED]]), 0.0)
    assert rt.tick(0.0).action[2] == CLOSED
    rt.accept(goal(2, Operation(OpKind.GUARDED_WAIT)), 0.05)
    r = rt.tick(0.1)
    assert r.mode == Mode.HOLD and r.action[2] == CLOSED


def test_stale_or_invalid_messages_are_rejected_with_a_reason():
    rt = runtime()
    assert not rt.accept(goal(1, epoch=0), 0.0)
    assert rt.accept(goal(5), 0.0)
    assert not rt.accept(goal(3), 0.0)
    assert not rt.accept(goal(6, expires=0.0), 0.0)
    assert not rt.submit(chunk(9, 0.0, [[0, 0, OPEN]]), 0.0)
    assert not rt.submit(chunk(5, 0.0, [[0, 0]]), 0.0)
    acks = rt.drain_acks()
    commands = [a for a in acks if a.status == AckStatus.REJECTED]
    chunks = [a for a in acks if a.status == AckStatus.CHUNK_REJECTED]
    assert len(commands) == 3 and len(chunks) == 2 and all(a.detail for a in commands + chunks)


def test_ack_sequence_is_strictly_increasing():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0, 0, OPEN]]), 0.0)
    rt.tick(0.0)
    rt.revoke(Revoke(1, 1, "x"), 0.1)
    rt.accept(goal(1, epoch=9), 0.1)
    seqs = [a.seq for a in rt.drain_acks()]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs) == 3


def test_lease_expiry_holds_and_rejects_later_chunks():
    rt = runtime()
    rt.accept(goal(1, expires=0.25), 0.0)
    rt.submit(chunk(1, 0.0, [[0.5, 0, OPEN]] * 10), 0.0)
    rs = run(rt, 0.0, 4)
    assert [r.source for r in rs] == ["chunk", "chunk", "chunk", "hold"] and rt.mode == Mode.HOLD
    assert (1, AckStatus.EXPIRED) in statuses(rt)
    assert not rt.submit(chunk(1, 0.4, [[0.5, 0, OPEN]]), 0.35)


def test_pending_command_that_never_starts_expires():
    rt = runtime()
    rt.accept(goal(1, expires=0.2), 0.0)
    run(rt, 0.0, 3)
    assert rt.mode == Mode.HOLD and rt.pending is None and (1, AckStatus.EXPIRED) in statuses(rt)


def test_immediate_transition_keeps_old_motion_until_the_new_chunk_is_due():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 10), 0.0)
    run(rt, 0.0, 2)
    rt.accept(goal(2, PLACE_B, Transition.IMMEDIATE), 0.15)
    assert rt.tick(0.2).cmd_id == 1
    rt.submit(chunk(2, 0.4, [[-0.1, 0, OPEN]] * 3), 0.25)
    assert [r.cmd_id for r in run(rt, 0.3, 3)] == [1, 2, 2]
    assert rt.tick(0.6).action[0] == pytest.approx(-0.1)


def test_pause_hold_transition_holds_now_and_refuses_old_chunks():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 10), 0.0)
    rt.tick(0.0)
    rt.accept(goal(2, PLACE_B, Transition.PAUSE_HOLD), 0.05)
    assert rt.tick(0.1).source == "hold" and rt.mode == Mode.TRANSITION
    assert not rt.submit(chunk(1, 0.2, [[0.1, 0, OPEN]]), 0.15)
    rt.submit(chunk(2, 0.2, [[-0.1, 0, OPEN]]), 0.15)
    assert rt.tick(0.2).cmd_id == 2


def test_newer_goal_supersedes_a_pending_one():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.accept(goal(2, PLACE_B), 0.05)
    assert (1, AckStatus.REVOKED) in statuses(rt)
    assert not rt.submit(chunk(1, 0.1, [[0.1, 0, OPEN]]), 0.1)


def test_new_segment_is_blended_in_continuous_dims_only():
    rt = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, 0.0, OPEN]), blend_s=0.3)
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.6, 0, CLOSED]] * 5), 0.0)
    xs = [r.action[0] for r in run(rt, 0.0, 4)]
    assert 0.0 < xs[0] < xs[1] < xs[2] == pytest.approx(0.6) == xs[3]
    assert rt.last[2] == CLOSED


def test_bounds_are_clamped_and_reported():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[2.0, 0, OPEN]]), 0.0)
    r = rt.tick(0.0)
    assert r.action[0] == 1.0 and r.violations and "vx=2.0000" in r.violations[0]


def test_hold_without_a_seed_is_refused_for_keep_dims():
    with pytest.raises(ValueError, match="seed the runtime"):
        ExecutionRuntime(SPEC, epoch=1).tick(0.0)


def test_stop_is_final_for_the_epoch():
    rt = runtime(last=np.array([0.0, 0.0, CLOSED]))
    Guard(rt).stop(0.0)
    assert not rt.accept(goal(1), 0.0)
    r = rt.tick(0.1)
    assert r.mode == Mode.STOPPED and r.source == "hold" and r.action[2] == CLOSED


def test_guard_holds_a_starved_command_and_resumes_when_chunks_return():
    rt = runtime()
    g = Guard(rt, watchdog_s=0.25)
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]]), 0.0)
    g.step(0.0)
    rs = [g.step(0.1 * k) for k in range(1, 5)]
    assert rs[-1].mode == Mode.HOLD and rs[-1].hold_reason == "starved"
    rt.submit(chunk(1, 0.5, [[0.2, 0, OPEN]]), 0.45)
    assert g.step(0.5).source == "chunk"


def test_guard_holds_when_the_owner_is_lost_and_cancels_pending_work():
    rt = runtime()
    g = Guard(rt, owner_timeout_s=0.3)
    g.heartbeat(0.0)
    rt.accept(goal(1), 0.0)
    g.step(0.5)
    assert rt.mode == Mode.HOLD and rt.hold_reason == "owner lost" and rt.pending is None
    assert not rt.submit(chunk(1, 0.6, [[0.1, 0, OPEN]]), 0.55)


def test_an_owner_lost_hold_drops_the_running_command_and_a_new_one_moves_again():
    rt = runtime()
    g = Guard(rt, owner_timeout_s=0.3)
    g.heartbeat(0.0)
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 50), 0.0)
    g.step(0.0)
    g.step(0.5)
    dropped = [a for a in rt.drain_acks() if a.status == AckStatus.DROPPED]
    assert [a.cmd_id for a in dropped] == [1] and rt.active is None and rt.target() is None
    g.heartbeat(0.6)
    assert rt.accept(goal(2), 0.6)
    assert rt.submit(chunk(2, 0.62, [[0.2, 0, OPEN]] * 50), 0.6)
    r = g.step(0.62)
    assert r.source == "chunk" and r.cmd_id == 2 and rt.mode == Mode.RUNNING


def test_guard_survives_a_runtime_error():
    rt = runtime()
    g = Guard(rt)
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[np.nan, 0, OPEN]]), 0.0)
    r = g.step(0.0)
    assert r.source == "hold" and "runtime error" in r.hold_reason and g.errors == 1


def test_robocasa_panda_omron_plugs_in():
    """The first real robot spec: 12-D π0.5² action, holds by zero deltas, keeps gripper and base mode."""
    import importlib.util

    from rojev.robot.sim import REPO
    path = REPO / "executors/robocasa-pi052/embodiment.py"
    mod = importlib.util.module_from_spec(spec := importlib.util.spec_from_file_location("robocasa_embodiment", path))
    spec.loader.exec_module(mod)
    panda = mod.PANDA_OMRON
    assert panda.size == 12 and panda.control_hz == 20
    holding = np.array([0.3] * 6 + [1.0] + [0.2] * 4 + [-1.0])
    hold = panda.hold(holding)
    assert np.all(hold[:6] == 0) and hold[6] == 1.0 and np.all(hold[7:11] == 0) and hold[11] == -1.0
    release = holding.copy()
    release[6] = -1.0
    assert panda.irreversible_change(holding, release) == ["gripper_close"]
    rt = ExecutionRuntime(panda, epoch=1, last=holding)
    rt.accept(goal(1, Operation(OpKind.GUARDED_WAIT)), 0.0)
    assert rt.tick(0.0).action[6] == 1.0


def test_lease_expires_even_while_starved():
    rt = runtime()
    g = Guard(rt, watchdog_s=0.15)
    rt.accept(goal(1, expires=0.5), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]]), 0.0)
    [g.step(0.1 * k) for k in range(4)]
    assert rt.mode == Mode.HOLD and rt.hold_reason == "starved" and rt.target() is not None
    g.step(0.6)
    assert rt.active is None and rt.target() is None and (1, AckStatus.EXPIRED) in statuses(rt)
    assert not rt.submit(chunk(1, 0.7, [[0.1, 0, OPEN]]), 0.65)


def test_chunk_at_another_rate_is_rejected():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    bad = TimedChunk(1, 1, obs_id=0, t_start=0.0, dt=DT / 2, actions=np.array([[0.1, 0, OPEN]] * 4))
    assert not rt.submit(bad, 0.0)
    assert "control dt" in rt.drain_acks()[-1].detail


def test_old_command_chunk_cannot_delay_a_scheduled_takeover():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 3), 0.0)
    rt.tick(0.0)
    rt.accept(goal(2, PLACE_B, Transition.IMMEDIATE), 0.05)
    rt.submit(chunk(2, 0.3, [[-0.1, 0, OPEN]] * 3), 0.05)
    rt.submit(chunk(1, 0.1, [[0.2, 0, OPEN]] * 10), 0.05)
    assert [r.cmd_id for r in run(rt, 0.1, 4)] == [1, 1, 2, 2]


def test_renew_keeps_every_field_but_the_lease():
    rt = runtime()
    g = AcceptedGoal(1, 1, 3, PICK, Transition.PAUSE_HOLD, expires_t=1.0, instruction="put the cube in A")
    rt.accept(g, 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 2), 0.0)
    rt.tick(0.0)
    assert rt.renew(1, 5.0, 0.1)
    assert rt.active.expires_t == 5.0 and rt.active.instruction == "put the cube in A" and rt.active.instruction_rev == 3


def test_review_keeps_moving_but_freezes_the_gripper_until_renewed():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN], [0.1, 0, CLOSED], [0.1, 0, CLOSED], [0.1, 0, CLOSED]]), 0.0)
    rt.tick(0.0)
    assert rt.review(1, until_t=5.0, now=0.05)
    r = rt.tick(0.1)
    assert r.source == "chunk" and r.review and r.action[0] == pytest.approx(0.1)
    assert r.action[2] == OPEN and r.held_switches == ("gripper",)
    assert rt.renew(1, 10.0, 0.15)
    r = rt.tick(0.2)
    assert not r.review and r.action[2] == CLOSED


def test_review_shortens_the_lease_and_cannot_target_unknown_commands():
    rt = runtime()
    rt.accept(goal(1, expires=100.0), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 20), 0.0)
    rt.tick(0.0)
    assert rt.review(1, until_t=0.25, now=0.05) and rt.active.expires_t == 0.25
    assert not rt.review(7, until_t=1.0, now=0.05)
    rs = run(rt, 0.1, 3)
    assert [r.source for r in rs] == ["chunk", "chunk", "hold"] and rs[-1].hold_reason == "lease expired"


def test_a_new_command_ends_the_review():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.1, 0, OPEN]] * 10), 0.0)
    rt.tick(0.0)
    rt.review(1, until_t=5.0, now=0.05)
    rt.accept(goal(2, PLACE_B, Transition.IMMEDIATE), 0.05)
    rt.submit(chunk(2, 0.2, [[0.1, 0, CLOSED]] * 3), 0.1)
    rs = run(rt, 0.1, 3)
    assert rs[0].review and rs[1].cmd_id == 2 and not rs[1].review and rs[1].action[2] == CLOSED


def test_decelerating_hold_ramps_continuous_dims_and_keeps_switches():
    rt = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, 0.0, OPEN]), blend_s=0.0, stop_s=0.3)
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.6, -0.3, CLOSED]] * 2), 0.0)
    run(rt, 0.0, 2)
    rs = run(rt, 0.2, 4)
    xs = [r.action[0] for r in rs]
    assert all(r.source == "hold" for r in rs)
    assert 0.6 > xs[0] > xs[1] > xs[2] == pytest.approx(0.0) == xs[3]
    assert all(r.action[2] == CLOSED for r in rs)


def test_instant_hold_by_default():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.6, 0, OPEN]]), 0.0)
    rt.tick(0.0)
    assert rt.tick(0.1).action[0] == 0.0


def test_starvation_is_counted_only_for_an_active_moving_command_without_a_due_action():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    waiting = run(rt, 0.0, 2)
    assert not any(r.starved for r in waiting) and rt.mode == Mode.TRANSITION
    rt.submit(chunk(1, 0.2, [[0.1, 0, OPEN]] * 2), 0.2)
    running = run(rt, 0.2, 2)
    assert not any(r.starved for r in running)
    gap = run(rt, 0.4, 3)
    assert all(r.starved and r.source == "hold" for r in gap) and rt.starved_ticks == 3
    rt.submit(chunk(1, 0.7, [[0.1, 0, OPEN]] * 2), 0.7)
    assert not rt.tick(0.7).starved and rt.starved_ticks == 3


def test_holding_operations_and_revoked_commands_never_count_as_starved():
    rt = runtime()
    rt.accept(goal(1, op=Operation(OpKind.GUARDED_WAIT)), 0.0)
    assert not any(r.starved for r in run(rt, 0.0, 3))
    rt.accept(goal(2), 0.3)
    rt.submit(chunk(2, 0.3, [[0.1, 0, OPEN]]), 0.3)
    rt.tick(0.3)
    rt.revoke(Revoke(1, 2, "user"), 0.35)
    assert not any(r.starved for r in run(rt, 0.4, 3)) and rt.starved_ticks == 0


def test_blending_stays_inside_the_bounds_and_is_not_a_violation():
    rt = ExecutionRuntime(SPEC, epoch=1, last=np.array([-0.5, 0.0, OPEN]), blend_s=0.5)
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[0.0, 0, OPEN], [-1.0, 0, OPEN], [-1.0, 0, OPEN]]), 0.0)
    rs = run(rt, 0.0, 3)
    assert all(-1.0 <= r.action[0] <= 1.0 for r in rs) and not any(r.violations for r in rs)
    assert rs[1].action[0] == -1.0


def test_an_out_of_range_incoming_action_is_still_reported():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    rt.submit(chunk(1, 0.0, [[1.5, 0, OPEN]]), 0.0)
    r = rt.tick(0.0)
    assert r.action[0] == 1.0 and r.violations and "vx" in r.violations[0]


def test_queued_from_returns_the_commands_actions_due_from_a_time_as_the_chunk_gave_them():
    rt = runtime()
    rt.accept(goal(1), 0.0)
    acts = [[0.1 * k, 0.0, OPEN] for k in range(1, 9)]
    rt.submit(chunk(1, 0.0, acts), 0.0)
    run(rt, 0.0, 3)
    left = rt.queued_from(1, 3 * DT)
    assert np.allclose(left, acts[3:]) and rt.queued_from(2, 0.0) is None
    assert rt.queued_from(1, 100.0) is None
