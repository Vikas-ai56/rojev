"""Execution monitor (rojev/system1/monitor.py): calibration, the conformal false-alarm guarantee, and faults in a
simulated arm driven through the real runtime, guard and control loop (lockstep).

The simulated arm tracks a commanded planar velocity through a first-order lag with sensor noise, like an impedance
controller. Faults: blocked (commands applied, the arm does not move) and pushed (the arm moves without a command).
Bounds come from the method, not from observed results: with n calibration segments and rank r = ⌈(n+1)(1−α)⌉ the
expected per-segment false-alarm rate is (n+1−r)/(n+1) ≤ α. Real-robot calibration (RoboCasa with π0.5²) is a separate
run; see docs/features/execution-monitor.md.
"""
import math

import numpy as np
import pytest

from rojev.control import ControlLoop, ExecutionRuntime, Guard
from rojev.protocol import AcceptedGoal, Observation, Operation, OpKind, TimedChunk, Transition
from rojev.robot import ActionDim, ActionSpec, EffectModel, JointTargetEffect
from rojev.system1 import ExecutionMonitor, MonitorModel, Recording, fit_monitor
from rojev.system1.monitor import regressors

HZ = 50
SPEC = ActionSpec("planar arm", (ActionDim("vx", -1, 1, 0.0), ActionDim("vy", -1, 1, 0.0)), HZ)
WINDOW, SEGMENT, ALPHA = 10, 50, 0.05
TICKS = 400
NOISE = 2e-4


class VelocityEffect(EffectModel):
    def tracked(self, state):
        return np.asarray(state, dtype=float)

    def command(self, state, action):
        return np.asarray(action, dtype=float)

    @property
    def resolution(self):
        return NOISE


class PlanarArm:
    """Position integrates a lagged velocity (time constant `tau`) plus sensor noise; `blocked` / `push` are tick
    ranges where the arm stops / is displaced."""

    def __init__(self, seed, tau=0.06, blocked=range(0), push=range(0)):
        self.rng, self.tau, self.blocked, self.push = np.random.default_rng(seed), tau, blocked, push
        self.x, self.v, self.k = np.zeros(2), np.zeros(2), 0

    def observe(self):
        return Observation({}, self.x + self.rng.normal(0, NOISE, 2), "reach")

    def step(self, action):
        self.v += (np.asarray(action, dtype=float) - self.v) * min(1.0, (1 / HZ) / self.tau)
        if self.k in self.blocked:
            self.v[:] = 0.0
        self.x = self.x + self.v / HZ + (np.array([0.004, 0.0]) if self.k in self.push else 0.0)
        self.k += 1
        return self.observe()


def motion(seed):
    rng = np.random.default_rng(1000 + seed)
    t = np.arange(TICKS) / HZ
    f, phase, amp = rng.uniform(0.2, 0.8, 2), rng.uniform(0, 2 * np.pi, 2), rng.uniform(0.2, 0.6, 2)
    return np.stack([amp[i] * np.sin(2 * np.pi * f[i] * t + phase[i]) for i in range(2)], axis=1)


def run(seed, model=None, record=False, actions=None, **faults):
    arm = PlanarArm(seed, **faults)
    runtime = ExecutionRuntime(SPEC, epoch=1, last=np.zeros(2), blend_s=0.0)
    loop = ControlLoop(arm, Guard(runtime, watchdog_s=100.0), SPEC, realtime=False)
    found, scores = [], []
    monitor = ExecutionMonitor(VelocityEffect(), 1, model, on_deviation=found.append, record=record)

    def on_tick(r):
        monitor.observe(r)
        scores.append(monitor.score)

    loop.on_tick = on_tick
    runtime.accept(AcceptedGoal(1, 1, 0, Operation(OpKind.SWITCH_TO, subtask="reach"), Transition.PAUSE_HOLD, 1e9), 0.0)
    runtime.submit(TimedChunk(1, 1, 0, 0.0, SPEC.dt, motion(seed) if actions is None else actions), 0.0)
    loop.run_ticks(TICKS)
    return monitor, found, scores


@pytest.fixture(scope="module")
def recordings():
    return {s: run(s, record=True)[0].recording for s in range(120)}


@pytest.fixture(scope="module")
def model(recordings):
    return fit_monitor([recordings[s] for s in range(8)], WINDOW, SEGMENT, ALPHA, NOISE)


def segment_maxima(model, recording):
    tracked, command = recording.arrays()
    change, commanded, ends = regressors(tracked, command, model.window)
    scores = np.where(model.covers(commanded), model.scores(change, commanded), -np.inf)
    seg = ends // model.segment
    return [scores[seg == s].max() for s in np.unique(seg) if (seg == s).sum() == model.segment]


def test_recording_covers_every_tick_from_the_state_before_the_first_action(recordings):
    tracked, command = recordings[0].arrays()
    assert command.shape == (TICKS, 2) and tracked.shape == (TICKS + 1, 2)
    assert np.allclose(command, motion(0))


def test_fit_learns_the_tracking_gain(model):
    """Four calibration runs × 7 complete segments; the current and previous window gains add up to the plant's
    velocity-to-position gain (1/HZ per tick), the lag splitting it between them."""
    assert model.gain.shape == (4, 2) and model.calibration_segments == 4 * 7
    assert np.allclose(np.diag(model.gain[:2]) + np.diag(model.gain[2:]), 1 / HZ, rtol=0.05)
    assert np.all(np.diag(model.gain[2:]) > 0) and np.abs(model.gain[[0, 2], 1]).max() < 0.02 / HZ
    assert math.isfinite(model.threshold) and np.all(model.scale >= NOISE)


def test_online_scores_equal_offline_scores_on_the_same_run(model):
    monitor, _, online = run(130, model, record=True)
    tracked, command = monitor.recording.arrays()
    change, commanded, _ = regressors(tracked, command, WINDOW)
    first = 2 * WINDOW - 1
    assert online[:first] == [None] * first
    assert np.allclose(np.array(online[first:], dtype=float), model.scores(change, commanded))


def test_false_alarm_rate_meets_the_conformal_guarantee_on_average(recordings):
    """Ten independent calibrations of 8 runs (28 calibration segments, α = 0.05 → rank 28, expected rate 1/29); each
    is evaluated on the same 40 held-out runs (280 segments). The mean rate stays within three standard errors of α
    (σ ≈ 0.033 per calibration from the Beta spread, σ ≈ 0.011 binomial)."""
    held_out = [recordings[s] for s in range(80, 120)]
    rates = []
    for c in range(10):
        m = fit_monitor([recordings[s] for s in range(8 * c, 8 * c + 8)], WINDOW, SEGMENT, ALPHA, NOISE)
        maxima = [x for rec in held_out for x in segment_maxima(m, rec)]
        rates.append(np.mean(np.array(maxima) > m.threshold))
    assert np.mean(rates) <= ALPHA + 3 * math.hypot(0.033 / math.sqrt(10), 0.011)


def fastest_tick(seed, lo=100, hi=300):
    return lo + int(np.argmax(np.linalg.norm(motion(seed)[lo:hi], axis=1)))


@pytest.mark.parametrize("seed", range(200, 220))
def test_a_blocked_arm_is_detected_within_the_window(model, seed):
    start = fastest_tick(seed)
    _, found, _ = run(seed, model, blocked=range(start, start + 60))
    hits = [round(d.t * HZ) for d in found if round(d.t * HZ) >= start]
    assert hits and hits[0] - start <= WINDOW and found[-1].cmd_id == 1


def test_a_blocked_arm_that_was_not_asked_to_move_raises_nothing(model):
    actions = motion(240)
    actions[150:260] = 0.0
    _, found, _ = run(240, model, actions=actions, blocked=range(170, 250))
    assert not [d for d in found if 170 <= round(d.t * HZ) <= 250]


@pytest.mark.parametrize("seed", range(300, 320))
def test_an_arm_pushed_without_a_command_is_detected(model, seed):
    _, found, _ = run(seed, model, push=range(200, 205))
    assert [d for d in found if 200 <= round(d.t * HZ) <= 200 + WINDOW]


def test_one_deviation_per_excursion(model):
    start = fastest_tick(202)
    monitor, found, _ = run(202, model, blocked=range(start, start + 40), push=range(350, 355))
    inside = [d for d in found if start <= round(d.t * HZ) < 350]
    assert len(inside) >= 1 and [d for d in found if round(d.t * HZ) >= 350]
    assert monitor.max_score > model.threshold


def test_commands_in_a_direction_never_exercised_are_not_scored(recordings, model):
    y_only = np.array([False, True, False, True])
    narrow = MonitorModel(model.gain, model.scale, y_only, model.threshold, WINDOW, SEGMENT, ALPHA, 1)
    monitor, found, _ = run(5, narrow)
    assert not found and monitor.score is None and monitor.uncovered == TICKS - 2 * WINDOW + 1
    actions = motion(5)
    actions[:, 0] = 0.0
    monitor, _, scores = run(5, narrow, actions=actions)
    assert monitor.uncovered == 0 and scores[-1] is not None


def test_never_commanded_dimensions_are_marked_unexercised():
    actions = {s: motion(s) for s in range(4)}
    for a in actions.values():
        a[:, 0] = 0.0
    recs = [run(s, record=True, actions=actions[s])[0].recording for s in range(4)]
    m = fit_monitor(recs, WINDOW, SEGMENT, 0.2, NOISE)
    assert m.exercised.tolist() == [False, True, False, True]
    assert np.all(m.gain[[0, 2]] == 0.0)


def test_resolution_bounds_the_residual_scale(recordings):
    m = fit_monitor([recordings[s] for s in range(8)], WINDOW, SEGMENT, ALPHA, 1.0)
    assert np.all(m.scale == 1.0)
    for bad in (0.0, -1.0, np.array([NOISE, 0.0])):
        with pytest.raises(ValueError, match="resolution"):
            fit_monitor([recordings[0], recordings[1]], WINDOW, SEGMENT, ALPHA, bad)


def test_too_few_runs_or_segments_or_a_short_segment_is_refused(recordings):
    with pytest.raises(ValueError, match="two runs"):
        fit_monitor([recordings[0]], WINDOW, SEGMENT, ALPHA, NOISE)
    with pytest.raises(ValueError, match="calibration segments"):
        fit_monitor([recordings[0], recordings[1]], WINDOW, SEGMENT, 0.01, NOISE)
    with pytest.raises(ValueError, match="two windows"):
        fit_monitor([recordings[0], recordings[1]], WINDOW, 2 * WINDOW - 1, ALPHA, NOISE)


def test_model_and_recording_round_trip(tmp_path, model, recordings):
    assert MonitorModel.from_json(model.to_json()).threshold == model.threshold
    recordings[3].save(tmp_path / "r.npz")
    back = Recording.load(tmp_path / "r.npz")
    assert all(np.allclose(a, b) for a, b in zip(back.arrays(), recordings[3].arrays()))


def test_joint_target_effect_commands_the_tracking_error():
    effect = JointTargetEffect(resolution=0.0015)
    assert np.allclose(effect.command(np.array([0.1, 0.2]), np.array([0.3, 0.2])), [0.2, 0.0])
    assert np.allclose(effect.tracked(np.array([0.1, 0.2])), [0.1, 0.2]) and effect.resolution == 0.0015
