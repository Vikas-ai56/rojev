"""Execution monitor: did the world respond to the applied actions the way it does on nominal runs? (spec step 4.3)

Kinematic and training-free (CheckVLA's check without its world model). Per control tick it takes the applied action
and the observation that followed (`TickRecord`), asks the robot's `EffectModel` what the action commanded of the
tracked quantities and what they measure, and compares, over the last `window` ticks, the measured change with the
change the commands predict. The prediction is a linear map learned on nominal runs (`fit_monitor`) from the commands
summed over this window and over the window before it, so it absorbs controller gains and the tracking lag that carries
earlier commands into this window; the residual is scaled per tracked quantity and summarised as one score. A score
above the calibrated threshold raises one `Deviation` per excursion (blocked arm, slip, collision, being pushed).
The map is trusted only in the command directions the calibration exercised: a command dimension that was exactly
zero throughout the fitting runs has no fitted effect, so a window that uses it is not scored but counted as
`uncovered`. Magnitude is not boxed: the map is linear, and a larger command in an exercised direction is still
predicted (saturation beyond the calibrated range is a distribution shift, like any other, and shows as alarms).

Calibration is split conformal, split by run: even-numbered nominal runs fit the map and residual scale; the others are
cut into `segment`-tick segments, each giving one score (its maximum). The threshold is the ⌈(n+1)(1−α)⌉-th smallest of
those n maxima, so a segment of a new nominal run raises a false alarm with probability at most α. Splitting by run, not
by segment, matters: each run moves differently, so calibrating on segments of the fitting runs underestimates the
residual of a new run (measured: 0.17 false alarms per segment for α = 0.05). Segments of one run are correlated, so
the bound is exact for runs exchangeable with the calibration runs and approximate per segment; the test suite checks
it on held-out runs. α is per segment, not per run: `alarms_per_minute` states what it means in
time. A deviation is advisory (a fresh chunk and a decision, never a stop), so a moderate rate is cheap. Recalibrate
when the robot, controller, policy or task family changes.
"""
import json
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rojev.control import TickRecord
from rojev.protocol import Deviation
from rojev.robot import EffectModel


@dataclass
class Recording:
    """One run: `tracked[k]` measured after tick k, `command[k]` what tick k's action commanded (from the state before
    it). Row k of both belongs to the same tick; the state before the first recorded tick is `start`. `conditions`
    records how the run was made (e.g. real-time factor, policy, seed) so calibration can refuse unlike runs."""

    start: np.ndarray
    tracked: list[np.ndarray] = field(default_factory=list)
    command: list[np.ndarray] = field(default_factory=list)
    conditions: dict = field(default_factory=dict)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(T + 1, m) tracked including `start`, (T, n) commands."""
        return np.vstack([self.start, *self.tracked]), np.array(self.command, dtype=float)

    def save(self, path: Path) -> None:
        tracked, command = self.arrays()
        np.savez(path, tracked=tracked, command=command, conditions=json.dumps(self.conditions))

    @staticmethod
    def load(path: Path) -> "Recording":
        data = np.load(path)
        conditions = json.loads(str(data["conditions"])) if "conditions" in data else {}
        return Recording(data["tracked"][0], list(data["tracked"][1:]), list(data["command"]), conditions)


def regressors(tracked: np.ndarray, command: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per tick k ≥ 2·window − 1: measured change over the last `window` ticks, and the commands summed over that window
    and over the window before it (n + n columns); also the tick index k of each row."""
    ends = np.arange(2 * window - 1, len(command))
    csum = np.vstack([np.zeros((1, command.shape[1])), np.cumsum(command, axis=0)])
    change = tracked[ends + 1] - tracked[ends + 1 - window]
    current = csum[ends + 1] - csum[ends + 1 - window]
    previous = csum[ends + 1 - window] - csum[ends + 1 - 2 * window]
    return change, np.hstack([current, previous]), ends


@dataclass(frozen=True)
class MonitorModel:
    """`gain` (2n × m) maps the summed commands of this window and the one before to measured change; `scale` (m) is
    the nominal residual spread; `threshold` the conformal score bound for false-alarm rate `alpha` per `segment`-tick
    segment; `exercised` (2n) marks the regressors that were ever non-zero while fitting."""

    gain: np.ndarray
    scale: np.ndarray
    exercised: np.ndarray
    threshold: float
    window: int
    segment: int
    alpha: float
    calibration_segments: int

    def alarms_per_minute(self, control_hz: float) -> float:
        """Upper bound on nominal false alarms per minute implied by `alpha` per segment."""
        return self.alpha * 60.0 * control_hz / self.segment

    def covers(self, commanded: np.ndarray) -> np.ndarray:
        """Whether a window uses only exercised command directions."""
        return np.all((commanded == 0) | self.exercised, axis=-1)

    def scores(self, change: np.ndarray, commanded: np.ndarray) -> np.ndarray:
        residual = (change - commanded @ self.gain) / self.scale
        return np.sqrt(np.mean(residual ** 2, axis=-1))

    def to_json(self) -> str:
        return json.dumps({"gain": self.gain.tolist(), "scale": self.scale.tolist(),
                           "exercised": self.exercised.tolist(),
                           "threshold": self.threshold,
                           "window": self.window, "segment": self.segment, "alpha": self.alpha,
                           "calibration_segments": self.calibration_segments}, indent=2)

    @staticmethod
    def from_json(text: str) -> "MonitorModel":
        d = json.loads(text)
        return MonitorModel(np.array(d["gain"]), np.array(d["scale"]), np.array(d["exercised"], dtype=bool),
                            d["threshold"], d["window"], d["segment"], d["alpha"], d["calibration_segments"])


def fit_monitor(runs: list[Recording], window: int, segment: int, alpha: float,
                resolution: np.ndarray | float) -> MonitorModel:
    """Fit on runs 0, 2, 4, …; calibrate on runs 1, 3, 5, … (module docstring); the residual scale is at least the
    effect model's `resolution`. Raises when there are too few calibration segments for `alpha` (at least
    ⌈1/α⌉ − 1)."""
    if np.any(np.asarray(resolution) <= 0):
        raise ValueError("resolution must be positive")
    if segment < 2 * window:
        raise ValueError(f"segment ({segment}) must be at least two windows ({2 * window})")
    pairs = [regressors(*run.arrays(), window) for run in runs]
    fitting = [p for p in pairs[0::2] if len(p[0])]
    if not fitting or len(pairs) < 2:
        raise ValueError("need at least two runs: one to fit, one to calibrate")
    change, commanded = np.vstack([p[0] for p in fitting]), np.vstack([p[1] for p in fitting])
    gain, *_ = np.linalg.lstsq(commanded, change, rcond=None)
    scale = np.maximum((change - commanded @ gain).std(axis=0), resolution)
    exercised = np.any(commanded != 0, axis=0)
    model = MonitorModel(gain, scale, exercised, math.inf, window, segment, alpha, 0)
    maxima = []
    for run_change, run_cmd, ends in pairs[1::2]:
        scores = np.where(model.covers(run_cmd), model.scores(run_change, run_cmd), -np.inf)
        seg = ends // segment
        for s in np.unique(seg):
            inside = seg == s
            if inside.sum() == segment and np.isfinite(scores[inside]).any():
                maxima.append(float(scores[inside].max()))
    n = len(maxima)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        raise ValueError(f"alpha {alpha} needs at least {math.ceil(1 / alpha) - 1} full calibration segments, got {n}")
    return MonitorModel(gain, scale, exercised, sorted(maxima)[rank - 1], window, segment, alpha, n)


class ExecutionMonitor:
    """Runs in the control-loop thread (`observe` per `TickRecord`; a few vector operations). With a `model` it scores
    every tick and calls `on_deviation` once per excursion above the threshold; with `record` it keeps the run as a
    `Recording` for calibration. `score` is the latest score (None until two windows of ticks exist, or while they use
    a command direction the calibration never exercised: `uncovered` counts those ticks)."""

    def __init__(self, effect: EffectModel, epoch: int, model: MonitorModel | None = None,
                 on_deviation: Callable[[Deviation], None] | None = None, record: bool = False):
        self.effect, self.epoch, self.model, self.on_deviation = effect, epoch, model, on_deviation
        self.recording: Recording | None = None
        self.record = record
        self.score: float | None = None
        self.max_score = 0.0
        self.uncovered = 0
        self.deviations: list[Deviation] = []
        size = model.window if model is not None else 1
        self._tracked: deque[np.ndarray] = deque(maxlen=size + 1)
        self._command: deque[np.ndarray] = deque(maxlen=2 * size)
        self._alarm = False

    def observe(self, r: TickRecord) -> None:
        before = np.asarray(r.before.observation.state, dtype=float)
        command = self.effect.command(before, np.asarray(r.tick.action, dtype=float))
        tracked = self.effect.tracked(np.asarray(r.after.observation.state, dtype=float))
        if not self._tracked:
            start = self.effect.tracked(before)
            self._tracked.append(start)
            if self.record:
                self.recording = Recording(start)
        if self.recording is not None:
            self.recording.tracked.append(tracked)
            self.recording.command.append(command)
        if self.model is None:
            return
        self._tracked.append(tracked)
        self._command.append(command)
        w = self.model.window
        if len(self._command) < 2 * w:
            return
        history = np.array(self._command)
        change = self._tracked[-1] - self._tracked[0]
        commanded = np.concatenate([history[w:].sum(axis=0), history[:w].sum(axis=0)])
        if not self.model.covers(commanded):
            self.score = None
            self.uncovered += 1
            return
        self.score = float(self.model.scores(change, commanded))
        self.max_score = max(self.max_score, self.score)
        if self.score > self.model.threshold and not self._alarm:
            self._alarm = True
            d = Deviation(self.epoch, r.tick.cmd_id, r.tick.t, round(self.score, 3), round(self.model.threshold, 3))
            self.deviations.append(d)
            if self.on_deviation is not None:
                self.on_deviation(d)
        elif self.score <= self.model.threshold:
            self._alarm = False
