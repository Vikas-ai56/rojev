"""Control loop + S1 runner with a stand-in robot and policy (no simulator, no models).

Logic tests are deterministic: one thread, simulated time (`Sim`), where the policy's latency runs the control ticks
that would happen while it thinks. The stand-in policy returns a chunk whose action k carries k + 1 in its first
dimension (0 = the robot was holding), so a test sees exactly which actions reached the robot. Wall-clock behaviour
(pacing, real threads) is in the tests marked `timing`, run alone: `uv run pytest -m timing`.
"""
import math
import time

import numpy as np
import pytest

from rojev.control import ControlLoop, ExecutionRuntime, Guard, Mode, RobotBackend
from rojev.protocol import AckStatus, AcceptedGoal, Observation, Operation, OpKind, Revoke, Transition
from rojev.robot import KEEP, ActionDim, ActionSpec, CameraSpec, ManualClock, Rate, joint_target_spec, so101
from rojev.system1 import ActionChunk, Policy, S1Runner

SPEC = ActionSpec("stand-in", (ActionDim("v", -100, 100, 0.0),
                               ActionDim("gripper", -1, 1, KEEP, threshold=0.5, irreversible=True)), control_hz=50)
DT = SPEC.dt
GO = Operation(OpKind.SWITCH_TO, subtask="go")


class Robot:
    def __init__(self, clock=None):
        self.applied: list[np.ndarray] = []
        if clock is not None:
            self.clock = clock

    def observe(self) -> Observation:
        return Observation({}, np.zeros(1, dtype=np.float32), "task")

    def step(self, action) -> Observation:
        self.applied.append(np.array(action))
        return self.observe()


class StandInPolicy(Policy):
    """`think(seconds)` stands for inference: wall sleep in timing tests, simulated control ticks in logic tests."""

    def __init__(self, latency_s: float, horizon: int = 20, fail: bool = False, think=time.sleep):
        self.latency_s, self.horizon, self.fail, self.think, self.calls = latency_s, horizon, fail, think, 0

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        self.calls += 1
        self.think(self.latency_s)
        if self.fail:
            raise ConnectionError("service unreachable")
        return ActionChunk(np.array([[k + 1, -1.0] for k in range(self.horizon)], dtype=np.float32))

    def reset(self) -> None:
        pass


class Sim:
    """Deterministic: simulated clock; the policy's latency runs the ticks that fall inside it."""

    def __init__(self, latency_s, horizon=20, refill_s=0.15, fail=False, watchdog_s=0.3, policy=None):
        self.clock = ManualClock()
        self.robot = Robot(self.clock)
        self.runtime = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, -1.0]), blend_s=0.0)
        self.loop = ControlLoop(self.robot, Guard(self.runtime, watchdog_s=watchdog_s), SPEC, realtime=False)
        self.policy = policy or StandInPolicy(latency_s, horizon, fail, think=self.think)
        self.runner = S1Runner(self.runtime, self.policy, self.loop.latest, self.loop.now, refill_s=refill_s,
                               error_backoff_s=0.0)
        self.next_tick = 0.0

    def think(self, seconds: float) -> None:
        end = self.clock.t + seconds
        while self.next_tick <= end + 1e-9:
            self.clock.t = self.next_tick
            self.loop.run_ticks(1)
            self.next_tick += DT
        self.clock.t = end

    def run(self, seconds: float) -> None:
        end = self.clock.t + seconds
        while self.clock.t < end - 1e-9:
            if not self.runner.step():
                self.think(DT)

    def go(self, instruction: str = "") -> None:
        self.runtime.accept(AcceptedGoal(1, 1, 0, GO, Transition.PAUSE_HOLD, expires_t=100.0,
                                         instruction=instruction), self.clock.t)


def values(sim):
    return [float(a[0]) for a in sim.robot.applied]


def test_runner_keeps_a_command_supplied_without_gaps():
    sim = Sim(latency_s=0.05, horizon=20, refill_s=0.15)
    sim.go()
    sim.run(1.5)
    applied = values(sim)
    first = next(i for i, v in enumerate(applied) if v != 0.0)
    assert first == round(0.05 / DT) + 1
    assert all(v != 0.0 for v in applied[first:]) and sim.runtime.starved_ticks == 0
    assert sim.runtime.mode == Mode.RUNNING and sim.policy.calls >= 3


def test_actions_due_during_inference_are_skipped_and_latency_is_measured():
    sim = Sim(latency_s=0.1, horizon=20, refill_s=0.0)
    sim.go()
    sim.run(0.4)
    first_value = next(v for v in values(sim) if v != 0.0)
    assert first_value == round(0.1 / DT) + 2
    assert sim.runner.stats.obs_age_s[0] == pytest.approx(0.1)
    assert sim.runner.latency_s() == pytest.approx(0.1)


def test_after_the_first_request_the_lead_time_covers_the_latency():
    sim = Sim(latency_s=0.1, horizon=30, refill_s=0.0)
    sim.go()
    sim.run(0.25)
    starved_after_first = sim.runtime.starved_ticks
    sim.run(2.0)
    assert sim.runtime.starved_ticks == starved_after_first


def test_policy_slower_than_its_chunk_never_moves_the_robot():
    sim = Sim(latency_s=0.6, horizon=10, refill_s=0.0)
    sim.go()
    sim.run(1.3)
    assert sim.runner.stats.requests >= 2 and sim.runner.stats.accepted == 0
    assert sim.runtime.mode == Mode.TRANSITION and all(v == 0.0 for v in values(sim))
    assert any(a.status == AckStatus.CHUNK_REJECTED for a in sim.runtime.drain_acks())


def test_revoke_stops_supply_and_late_chunks_are_rejected():
    sim = Sim(latency_s=0.05, refill_s=0.3)
    sim.go()
    sim.run(0.4)
    sim.runtime.revoke(Revoke(1, 1, "user changed mind"), sim.clock.t)
    n_revoke, calls = len(sim.robot.applied), sim.policy.calls
    sim.run(0.3)
    assert all(v == 0.0 for v in values(sim)[n_revoke:])
    assert sim.policy.calls == calls and sim.runtime.target() is None


def test_a_chunk_in_flight_when_revoked_is_rejected():
    sim = Sim(latency_s=0.05, refill_s=0.3)

    class RevokesWhileThinking(StandInPolicy):
        def act(self, obs, subtask):
            if self.calls == 1:
                sim.runtime.revoke(Revoke(1, 1, "user changed mind"), sim.clock.t)
            return super().act(obs, subtask)

    sim.policy = sim.runner.policy = RevokesWhileThinking(0.05, think=sim.think)
    sim.go()
    sim.run(0.6)
    acks = [(a.cmd_id, a.status) for a in sim.runtime.drain_acks()]
    assert (1, AckStatus.REVOKED) in acks and (1, AckStatus.CHUNK_REJECTED) in acks
    assert sim.runner.stats.accepted == 1


def test_policy_errors_are_counted_and_the_robot_holds():
    sim = Sim(latency_s=0.01, fail=True)
    sim.go()
    sim.run(0.4)
    assert sim.runner.stats.errors >= 2 and "unreachable" in sim.runner.stats.last_error
    assert all(v == 0.0 for v in values(sim)) and sim.loop.error is None


def test_policy_is_conditioned_on_the_commands_instruction_not_the_stale_task():
    seen = []

    class Recorder(StandInPolicy):
        def act(self, obs, subtask):
            seen.append((obs.task, subtask))
            return super().act(obs, subtask)

    sim = Sim(latency_s=0.0)
    sim.policy = sim.runner.policy = Recorder(0.0, think=sim.think)
    sim.go(instruction="actually, put it in B")
    sim.runner.step()
    assert seen == [("actually, put it in B", "go")]


def test_refill_now_requests_a_chunk_even_with_a_full_queue():
    sim = Sim(latency_s=0.0, horizon=50, refill_s=0.1)
    sim.go()
    assert sim.runner.step() and not sim.runner.step()
    assert sim.runner.refill_now(1)
    assert sim.runner.step() and sim.runner.stats.forced == 1 and not sim.runner.step()


def test_a_refill_for_another_command_is_ignored_and_counted():
    sim = Sim(latency_s=0.0, horizon=50, refill_s=0.1)
    sim.go()
    assert sim.runner.step()
    assert not sim.runner.refill_now(7) and not sim.runner.refill_now(None)
    assert not sim.runner.step() and sim.runner.stats.forced == 0 and sim.runner.stats.refill_ignored == 2
    assert sim.runner.refill_now(1)
    sim.runtime.accept(AcceptedGoal(1, 2, 0, GO, Transition.PAUSE_HOLD, expires_t=100.0), sim.clock.t)
    assert sim.runtime.target().cmd_id == 2 and sim.runner.step()
    assert sim.runner.stats.forced == 0 and sim.runner.stats.refill_ignored == 3


def test_a_floor_longer_than_the_chunk_is_counted():
    slow = Sim(latency_s=0.3, horizon=20, refill_s=0.0)
    slow.go()
    slow.run(1.5)
    assert slow.runner.stats.floors[0] == 20 and max(slow.runner.stats.floors) == math.ceil(0.6 / DT) + 1
    assert slow.runner.stats.floor_over_chunk == slow.runner.stats.requests - 1
    fast = Sim(latency_s=0.05, horizon=20, refill_s=0.15)
    fast.go()
    fast.run(1.5)
    assert fast.runner.stats.requests > 2 and fast.runner.stats.floor_over_chunk == 0


class Ramp(Policy):
    """Ramps shoulder_pan to q0 + 0.3 rad over a 30-action chunk from the newest pose (the remaining distance roughly
    halves per refill)."""

    def __init__(self, q0):
        self.q0 = q0

    def act(self, obs, subtask):
        q = np.asarray(obs.state, dtype=float)
        steps = np.linspace(0, 1, 30)[:, None]
        target = np.where(np.arange(len(q)) == 0, self.q0[0] + 0.3, self.q0)
        return ActionChunk((q + (target - q) * steps).astype(np.float32))

    def reset(self):
        pass


def so101_system(time_mode):
    robot = so101.simulated(cameras=(CameraSpec("overhead", width=96, height=72, fps=10.0),), time_mode=time_mode)
    return robot


def joint_spec(robot):
    g = robot.limits
    return joint_target_spec(robot, switches={"gripper": float((g.lower[-1] + g.upper[-1]) / 2)},
                             irreversible=("gripper",))


def test_any_rojev_robot_plugs_in_so101_lockstep():
    """The SO-101 MuJoCo backend (own clock, joint-position targets) under the same runtime, S1 runner and guard,
    deterministic: lockstep physics, one thread, the loop paced on the robot's clock. The arm ramps by 0.3 rad, then a
    revoke holds it where it is."""
    with so101_system("lockstep") as robot:
        backend = RobotBackend(robot, task="pan right")
        spec = joint_spec(robot)
        q0 = backend.current_command()
        runtime = ExecutionRuntime(spec, epoch=1, last=q0.copy(), blend_s=0.0)
        loop = ControlLoop(backend, Guard(runtime), spec)
        runner = S1Runner(runtime, Ramp(q0), loop.latest, loop.now, refill_s=0.5)
        runtime.accept(AcceptedGoal(1, 1, 0, GO, Transition.PAUSE_HOLD, expires_t=100.0), robot.clock.now())
        rate = Rate(robot.clock, spec.control_hz)

        def ticks(n):
            for _ in range(n):
                runner.step()
                loop.run_ticks(1)
                rate.sleep()

        ticks(round(2.5 * spec.control_hz))
        moved = np.array(loop.latest().observation.state, dtype=float)
        runtime.revoke(Revoke(1, 1, "stop"), loop.now())
        ticks(round(0.5 * spec.control_hz))
        held = robot.read().joints.position
        assert moved[0] - q0[0] == pytest.approx(0.3, abs=0.03)
        assert np.abs(held - moved).max() < 0.02
        assert backend.last_result is not None and not backend.last_result.violations
        assert runner.stats.accepted >= 3 and runtime.starved_ticks == 0


@pytest.mark.timing
def test_loop_keeps_real_time_with_nothing_to_do():
    """Budget: the loop never starts a tick more than one control period late."""
    robot = Robot()
    runtime = ExecutionRuntime(SPEC, epoch=1, last=np.array([0.0, -1.0]), blend_s=0.0)
    loop = ControlLoop(robot, Guard(runtime), SPEC, realtime=True)
    t0 = time.perf_counter()
    loop.start()
    time.sleep(0.5)
    loop.stop()
    elapsed = time.perf_counter() - t0
    assert loop.ticks == pytest.approx(elapsed * SPEC.control_hz, abs=3)
    assert loop.max_lag_s < DT and all(a[0] == 0.0 for a in robot.applied)


@pytest.mark.timing
def test_so101_realtime_keeps_its_clock_and_lag_budget():
    """Real threads on the SO-101 real-time backend. Budget: robot clock tracks the wall clock within one control
    period over the run, and no tick starts more than one control period late."""
    with so101_system("realtime") as robot:
        backend = RobotBackend(robot, task="pan right")
        spec = joint_spec(robot)
        q0 = backend.current_command()
        runtime = ExecutionRuntime(spec, epoch=1, last=q0.copy(), blend_s=0.0)
        loop = ControlLoop(backend, Guard(runtime), spec)
        runner = S1Runner(runtime, Ramp(q0), loop.latest, loop.now, refill_s=0.5)
        runtime.accept(AcceptedGoal(1, 1, 0, GO, Transition.PAUSE_HOLD, expires_t=100.0), robot.clock.now())
        w0, t0 = time.perf_counter(), robot.clock.now()
        loop.start()
        runner.start()
        time.sleep(2.5)
        runner.stop()
        loop.stop()
        period = 1.0 / spec.control_hz
        assert (robot.clock.now() - t0) == pytest.approx(time.perf_counter() - w0, abs=period)
        assert loop.max_lag_s < period and runner.stats.accepted >= 1
