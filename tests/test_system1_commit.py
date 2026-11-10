"""Commitment policies (rojev/system1/commit.py) and the latency-derived floor (rojev/system1/runner.py): unit rules,
randomized invariants, then a deterministic closed-loop reach-and-grasp through the real runtime and S1 runner, where
the robot keeps executing while the policy thinks.

The simulated world is one axis: the arm moves at the commanded velocity until the gripper closes; closing within
`GRASP_TOL` of the object is a grasp, anywhere else grasps air. Starvation is read from the runtime
(`starved_ticks`), never inferred by the test. Real-policy validation (π0.5² on RoboCasa) is roadmap T8a.
"""
import math

import numpy as np
import pytest

from rojev.control import ExecutionRuntime
from rojev.protocol import AcceptedGoal, Capture, Observation, Operation, OpKind, Transition
from rojev.robot import KEEP, ActionDim, ActionSpec
from rojev.system1 import (ActionChunk, CommitPolicy, Commitment, ConsensusCommit, FixedCommit, PhaseCommit, Policy,
                           S1Runner, make_commit_policy, speed_profile)
from rojev.system1.commit import first_switch_disagreement, irreversible_states

HZ = 50
DT = 1.0 / HZ
VEL = ActionSpec("velocity arm", (ActionDim("v", -1, 1, 0.0),
                                  ActionDim("gripper", -1, 1, KEEP, threshold=0.0, irreversible=True)), HZ)
POS = ActionSpec("position arm", (ActionDim("q", -5, 5, KEEP),
                                  ActionDim("gripper", -1, 1, KEEP, threshold=0.0, irreversible=True)), HZ)
OPEN, CLOSED = -1.0, 1.0
OBJECT, GRASP_TOL = 1.0, 0.05


def chunk(v, gripper=OPEN):
    v = np.asarray(v, dtype=float)
    g = np.broadcast_to(np.asarray(gripper, dtype=float), v.shape)
    return np.stack([v, g], axis=1)


def closes_at(k, horizon=30, v=0.5):
    return chunk(np.full(horizon, v), [OPEN] * k + [CLOSED] * (horizon - k))


def test_fixed_commits_the_whole_chunk_and_a_cap_never_goes_below_the_floor():
    c = chunk(np.full(30, 0.5))
    assert FixedCommit().plan([c], VEL, 5, None).steps == 30
    assert FixedCommit(max_commit=12).plan([c], VEL, 5, None).steps == 12
    assert FixedCommit(max_commit=5).plan([c], VEL, 12, None).steps == 12
    assert FixedCommit(max_commit=5).plan([c], VEL, 40, None).steps == 30


def test_phase_commits_through_the_first_slow_point_after_the_floor():
    v = np.concatenate([np.linspace(0.1, 0.8, 10), np.linspace(0.8, 0.05, 10), np.linspace(0.05, 0.8, 10)])
    out = PhaseCommit().plan([chunk(v)], VEL, 5, None)
    assert out.steps == 20 and out.reason == "phase boundary"
    assert PhaseCommit().plan([chunk(v)], VEL, 22, None).reason == "no slow point"
    assert PhaseCommit().plan([chunk(np.full(30, 0.5))], VEL, 5, None).reason == "no slow point"


def test_phase_reads_speed_from_steps_on_a_position_robot():
    q = np.concatenate([np.cumsum(np.full(15, 0.05)), np.full(3, 0.75), 0.75 + np.cumsum(np.full(12, 0.05))])
    speed = speed_profile(POS, chunk(q))
    assert speed[16] == 0.0 and speed[5] == pytest.approx(0.05)
    assert PhaseCommit().plan([chunk(q)], POS, 5, None).steps == 17


def test_identical_candidates_commit_everything():
    c = closes_at(20)
    out = ConsensusCommit().plan([c, c, c], VEL, 10, None)
    assert out.steps == 30 and out.reason == "candidates agree"


def test_still_candidates_agree_and_any_motion_against_a_still_majority_disagrees():
    still = chunk(np.zeros(30))
    assert ConsensusCommit().plan([still] * 3, VEL, 5, None).steps == 30
    mover = chunk(np.concatenate([np.zeros(10), np.full(20, 0.5)]))
    out = ConsensusCommit().plan([still, still, mover], VEL, 5, None)
    assert out.chosen in (0, 1) and out.steps == 30
    out = ConsensusCommit().plan([still, mover, mover + 0.3], VEL, 5, None)
    assert 10 <= out.steps <= 12 and out.reason == "candidates diverge"


def test_diverging_candidates_commit_their_agreed_prefix():
    base = np.full(30, 0.5)
    fan = [chunk(np.concatenate([base[:15], np.full(15, s)])) for s in (0.1, 0.5, 0.9)]
    out = ConsensusCommit().plan(fan, VEL, 5, None)
    assert 15 <= out.steps <= 17 and out.reason == "candidates diverge" and out.chosen == 1


def test_an_outlier_is_never_the_committed_candidate():
    good = chunk(np.full(30, 0.5))
    out = ConsensusCommit(tolerance=10).plan([chunk(np.full(30, -0.9)), good, good + 0.01], VEL, 5, None)
    assert out.chosen in (1, 2)


def test_disputed_switch_beyond_the_floor_stops_the_commitment_before_it():
    out = ConsensusCommit().plan([closes_at(12), closes_at(20), closes_at(20)], VEL, 5, None)
    assert out.steps == 12 and out.reason == "stop before a disputed switch"
    assert (out.actions[:, 1] == OPEN).all()


def test_disputed_switch_inside_the_floor_is_decided_by_majority():
    out = ConsensusCommit().plan([closes_at(0), closes_at(25), closes_at(25)], VEL, 10, None)
    assert out.steps == 10 and (out.actions[:, 1] == OPEN).all() and out.detail["votes"] == "2/3"
    out = ConsensusCommit().plan([closes_at(0), closes_at(0), closes_at(25)], VEL, 10, None)
    assert out.steps == 10 and (out.actions[:, 1] == CLOSED).all()
    assert out.reason == "switch decided by majority"


def test_even_k_tie_keeps_the_switch_value_in_effect_until_a_majority_changes_it():
    out = ConsensusCommit(samples=2).plan([closes_at(3), closes_at(6)], VEL, 10, None)
    assert out.chosen == 1 and out.reason == "switch tie keeps the value in effect" and out.detail["votes"] == "1/2"
    assert (out.actions[:6, 1] == OPEN).all() and (out.actions[6:, 1] == CLOSED).all()
    out = ConsensusCommit(samples=4).plan([closes_at(2), closes_at(8), closes_at(4), closes_at(6)], VEL, 10, None)
    assert out.chosen == 3 and out.reason == "switch tie keeps the value in effect" and out.detail["votes"] == "2/4"


def test_a_tie_at_the_first_step_has_no_value_in_effect_and_says_so():
    out = ConsensusCommit(samples=2).plan([closes_at(0), closes_at(25)], VEL, 10, None)
    assert out.steps == 10 and out.reason == "switch without a majority" and out.detail["votes"] == "0/2"
    out = ConsensusCommit(samples=4).plan([closes_at(0), closes_at(0), closes_at(25), closes_at(5)], VEL, 10, None)
    assert out.chosen != 2 and (out.actions[5:, 1] == CLOSED).all() and out.reason == "switch without a majority"


def test_a_tie_at_the_first_step_keeps_the_robots_current_switch():
    still_open = ConsensusCommit(samples=2).plan([closes_at(0), closes_at(25)], VEL, 10, np.array([0.0, OPEN]))
    assert still_open.chosen == 1 and (still_open.actions[:, 1] == OPEN).all()
    assert still_open.reason == "switch tie keeps the value in effect" and still_open.detail["votes"] == "1/2"
    already_closed = ConsensusCommit(samples=2).plan([closes_at(0), closes_at(25)], VEL, 10, np.array([0.0, CLOSED]))
    assert already_closed.chosen == 0 and (already_closed.actions[:, 1] == CLOSED).all()


def test_the_largest_group_is_not_a_majority():
    """Two of five close at step 2 (the largest identical group), but a majority has closed only from step 5."""
    cands = [closes_at(2), closes_at(2), closes_at(5), closes_at(9), closes_at(9)]
    out = ConsensusCommit(samples=5).plan(cands, VEL, 12, None)
    assert out.chosen == 2 and out.reason == "switch decided by majority" and out.detail["votes"] == "3/5"


def test_then_caps_the_consensus():
    v = np.concatenate([np.full(10, 0.8), np.full(5, 0.05), np.full(15, 0.8)])
    out = ConsensusCommit(then=PhaseCommit()).plan([chunk(v)] * 3, VEL, 5, None)
    assert out.steps < 30 and out.reason == "phase boundary"


def majority_switch(closed: np.ndarray) -> list[bool | None]:
    """Reference rule, one switch: per step the strict majority's value, else the previous step's (None = unknown)."""
    k, out = closed.shape[0], []
    for t in range(closed.shape[1]):
        n = int(closed[:, t].sum())
        out.append(True if 2 * n > k else False if 2 * (k - n) > k else (out[-1] if out else None))
    return out


def random_candidates(rng, spec, k, horizon):
    out = []
    for _ in range(k):
        if spec is VEL:
            motion = np.cumsum(rng.normal(0, 0.1, horizon)) + rng.normal(0, 0.3)
        else:
            motion = np.cumsum(np.cumsum(rng.normal(0, 0.01, horizon)))
        switch = rng.integers(0, horizon + 1)
        out.append(chunk(motion, [OPEN] * switch + [CLOSED] * (horizon - switch)))
    return out


@pytest.mark.parametrize("spec", [VEL, POS], ids=["velocity", "position"])
def test_commitment_invariants_hold_on_random_candidates(spec):
    """Never empty, never below the floor, never longer than the chunk, always a prefix of the chosen candidate;
    consensus: on every committed step the chosen candidate's switch takes the majority value (or the value in effect
    on a tie), and a disputed switch at or beyond the floor is never committed."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        k, horizon, floor = int(rng.integers(2, 6)), int(rng.integers(5, 40)), int(rng.integers(0, 45))
        cands = random_candidates(rng, spec, k, horizon)
        policies = [FixedCommit(), FixedCommit(max_commit=int(rng.integers(1, 40))), PhaseCommit(),
                    ConsensusCommit(samples=k), ConsensusCommit(samples=k, then=PhaseCommit())]
        for policy in policies:
            out = policy.plan(cands if isinstance(policy, ConsensusCommit) else cands[:1], spec, floor, None)
            assert min(max(floor, 1), horizon) <= out.steps <= horizon, (policy, floor, horizon, out.steps)
            assert np.array_equal(out.actions, np.asarray(cands[out.chosen])[:out.steps])
            if isinstance(policy, ConsensusCommit):
                closed = np.array([irreversible_states(spec, c[:out.steps])[:, 0] for c in cands])
                majority = majority_switch(closed)
                assert all(m is None or m == closed[out.chosen, t] for t, m in enumerate(majority))
                switch = first_switch_disagreement(spec, np.asarray(cands))
                if switch is not None and switch >= max(1, min(floor, horizon)):
                    assert out.steps <= switch


def test_factory_is_opt_in_and_strict():
    assert make_commit_policy() == FixedCommit()
    assert make_commit_policy("phase", slow_fraction=0.2) == PhaseCommit(slow_fraction=0.2)
    spec = make_commit_policy("phase", speculative_predict=True, samples=5)
    assert isinstance(spec, ConsensusCommit) and spec.samples == 5 and spec.then == PhaseCommit()
    for bad in ({"commit_policy": "adaptive"}, {"samples": 3}, {"tolerance": 0.2}, {"min_commit": 4},
                {"commit_policy": "phase", "min_commit": 4}):
        with pytest.raises(ValueError):
            make_commit_policy(**bad)
    with pytest.raises(ValueError):
        make_commit_policy(speculative_predict=True, samples=1)


class Counter(Policy):
    def __init__(self):
        self.calls = 0

    def act(self, obs, subtask, timeout_s=None):
        self.calls += 1
        return ActionChunk(chunk(np.full(5, float(self.calls))))

    def reset(self):
        pass


def test_default_act_samples_draws_independent_chunks():
    policy = Counter()
    chunks = policy.act_samples(Observation({}, np.zeros(2), "t"), "", 3)
    assert policy.calls == 3 and [c.actions[0, 0] for c in chunks] == [1.0, 2.0, 3.0]


class Clock:
    """Simulated time; advancing it runs every control tick crossed, so the robot moves while the policy thinks."""

    def __init__(self, runtime: ExecutionRuntime, world: "ReachWorld"):
        self.t, self.next_tick, self.runtime, self.world, self.trace = 0.0, 0.0, runtime, world, []

    def advance(self, seconds: float) -> None:
        end = self.t + seconds
        while self.next_tick <= end + 1e-9:
            self.t = self.next_tick
            r = self.runtime.tick(self.t)
            self.world.apply(r.action)
            self.trace.append(r)
            self.next_tick += DT
        self.t = end


class ReachWorld:
    def __init__(self):
        self.x, self.closed_at = 0.0, None

    def apply(self, action) -> None:
        if self.closed_at is not None:
            return
        if VEL.switches(action)[1]:
            self.closed_at = self.x
        else:
            self.x += float(action[0]) * DT

    @property
    def grasped(self) -> bool:
        return self.closed_at is not None and abs(self.closed_at - OBJECT) < GRASP_TOL


class ReachPolicy(Policy):
    """Plans a velocity ramp, from the position it observed (the request's capture, as a real policy plans from its
    observation), to where it believes the object is, and closes the gripper on arrival. Each sample sees the object
    at `OBJECT + noise`; `bad` samples believe it is right here (a perception failure). `batched`: K
    samples cost one latency (a batching service); otherwise K (the default `act_samples`, e.g. today's remote π0.5²)."""

    def __init__(self, clock: Clock, world: ReachWorld, latency_s=0.1, horizon=25, noise=0.0, bad=(), seed=0,
                 batched=True):
        self.clock, self.world, self.latency_s, self.horizon = clock, world, latency_s, horizon
        self.noise, self.bad, self.rng, self.batched = noise, set(bad), np.random.default_rng(seed), batched

    def sample(self, i: int, observed_x: float) -> ActionChunk:
        believed = observed_x if i in self.bad else OBJECT + self.rng.normal(0.0, self.noise)
        x, rows = observed_x, []
        for _ in range(self.horizon):
            arrived = abs(believed - x) < 0.01
            v = 0.0 if arrived else float(np.clip(3.0 * (believed - x), -0.5, 0.5))
            x += v * DT
            rows.append([v, CLOSED if arrived else OPEN])
        return ActionChunk(np.array(rows))

    def act_samples(self, obs, subtask, k, timeout_s=None):
        observed_x = self.world.x
        self.clock.advance(self.latency_s * (1 if self.batched else k))
        return [self.sample(i, observed_x) for i in range(k)]

    def act(self, obs, subtask, timeout_s=None):
        return self.act_samples(obs, subtask, 1)[0]

    def reset(self):
        pass


class IgnoresFloor(CommitPolicy):
    """A broken policy that commits `n` actions whatever the floor (to show what the floor prevents)."""

    def __init__(self, n):
        self.n = n

    def plan(self, candidates, spec, floor, last):
        return Commitment(np.asarray(candidates[0])[:self.n], 0, "ignores floor")


def reach(commit, seconds=4.0, **policy_kw):
    world = ReachWorld()
    runtime = ExecutionRuntime(VEL, epoch=1, last=np.array([0.0, OPEN]), blend_s=0.0)
    clock = Clock(runtime, world)
    policy = ReachPolicy(clock, world, **policy_kw)
    captures = iter(range(10**9))
    runner = S1Runner(runtime, policy, lambda: Capture(next(captures), clock.t, Observation({}, np.zeros(2), "pick")),
                      lambda: clock.t, commit=commit)
    runtime.accept(AcceptedGoal(1, 1, 0, Operation(OpKind.SWITCH_TO, subtask="pick"), Transition.PAUSE_HOLD,
                                expires_t=100.0), 0.0)
    grasp_t = None
    while clock.t < seconds and (grasp_t is None or clock.t < grasp_t + 0.2):
        if not runner.step():
            clock.advance(DT)
        if grasp_t is None and world.closed_at is not None:
            grasp_t = clock.t
    return world, runner, runtime


def test_runner_derives_floor_and_lead_time_from_measured_latency():
    world, runner, runtime = reach(FixedCommit(), latency_s=0.1)
    assert runner.stats.floors[0] == 25
    assert runner.latency_s() == pytest.approx(0.1, abs=DT)
    assert runner.floor(25) == math.ceil(2 * runner.latency_s() / DT) + 1
    assert runner.lead_s() == pytest.approx(runner.latency_s() + DT)


def test_closed_loop_fixed_commitment_grasps_with_a_clean_policy():
    world, runner, runtime = reach(FixedCommit())
    assert world.grasped and runner.stats.accepted == runner.stats.requests and runtime.starved_ticks == 0


def test_only_the_first_request_can_starve_and_later_ones_lead_by_the_measured_latency():
    """Before the first answer no latency is known (lead one tick), so the first chunk can land at most one latency
    of ticks late; every later request leads by the measured latency."""
    world, runner, runtime = reach(FixedCommit(), latency_s=0.2)
    assert world.grasped and runtime.starved_ticks <= round(0.2 / DT) + 1


def test_closed_loop_consensus_rejects_a_minority_perception_failure():
    bad = {"bad": (0,), "noise": 0.0}
    world, _, _ = reach(FixedCommit(), **bad)
    assert world.closed_at is not None and not world.grasped
    world, runner, runtime = reach(ConsensusCommit(samples=3), **bad)
    assert world.grasped and "switch decided by majority" in runner.stats.commit_reasons
    assert runtime.starved_ticks == 0


@pytest.mark.parametrize("seed", range(10))
def test_closed_loop_disputed_grasp_neither_livelocks_nor_starves_nor_grasps_early(seed):
    world, runner, runtime = reach(ConsensusCommit(samples=3), noise=0.01, seed=seed)
    assert world.grasped and runner.stats.accepted == runner.stats.requests
    assert runtime.starved_ticks == 0


@pytest.mark.parametrize("latency_s", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("policy", ["phase", "consensus", "consensus+phase"])
def test_closed_loop_derived_floor_never_starves(latency_s, policy):
    commit = {"phase": PhaseCommit(), "consensus": ConsensusCommit(samples=3),
              "consensus+phase": ConsensusCommit(samples=3, then=PhaseCommit())}[policy]
    world, runner, runtime = reach(commit, seconds=6.0, noise=0.005, seed=1, latency_s=latency_s, horizon=40)
    assert world.grasped and runtime.starved_ticks == 0


def test_a_policy_ignoring_the_floor_starves_the_robot():
    _, _, runtime = reach(IgnoresFloor(8), bad=())
    assert runtime.starved_ticks > 20


def test_unbatched_samples_raise_the_floor_and_starve_only_when_the_chunk_is_too_short():
    world, runner, runtime = reach(ConsensusCommit(samples=3), batched=False, latency_s=0.1, horizon=40, noise=0.005)
    assert max(runner.stats.floors[1:]) >= math.ceil(2 * 0.3 / DT) + 1
    assert world.grasped and runtime.starved_ticks == 0
    _, runner, runtime = reach(ConsensusCommit(samples=3), batched=False, latency_s=0.1, horizon=25, noise=0.005)
    assert max(runner.stats.floors[1:]) > 25 and runtime.starved_ticks > 0
