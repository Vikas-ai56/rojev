"""Joint limits from the robot model plus explicitly sourced extras, and the safety filter that enforces them.

These are hard checks, independent of any planner's costs or soft constraints. They are applied to every command a
backend receives, and planners reuse `JointLimits` to validate goals and trajectories.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

from .types import JointCommand


@dataclass(frozen=True)
class JointLimits:
    names: tuple[str, ...]
    lower: np.ndarray  # rad
    upper: np.ndarray  # rad
    max_velocity: np.ndarray  # rad/s; inf where unknown
    max_effort: np.ndarray  # actuator force limit; inf where unlimited
    sources: dict[str, str]  # where each kind of limit came from

    @classmethod
    def from_model(cls, m: mujoco.MjModel, names: tuple[str, ...], max_velocity: dict[str, float] | None = None,
                   velocity_source: str = "unspecified") -> "JointLimits":
        """Position limit = joint range ∩ ctrlrange of the actuator driving the joint; effort from forcerange."""
        lower, upper, effort = [], [], []
        for name in names:
            j = m.joint(name).id
            lo, hi = (m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf))
            acts = [a for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and m.actuator_trnid[a, 0] == j]
            if len(acts) != 1:
                raise ValueError(f"joint {name!r} must be driven by exactly one actuator, found {len(acts)}")
            a = acts[0]
            if m.actuator_ctrllimited[a]:
                lo, hi = max(lo, m.actuator_ctrlrange[a, 0]), min(hi, m.actuator_ctrlrange[a, 1])
            if not lo < hi:
                raise ValueError(f"joint {name!r} has an empty position range [{lo}, {hi}]")
            lower.append(lo)
            upper.append(hi)
            effort.append(max(abs(m.actuator_forcerange[a, 0]), abs(m.actuator_forcerange[a, 1]))
                          if m.actuator_forcelimited[a] else np.inf)
        vel = np.array([(max_velocity or {}).get(n, np.inf) for n in names], dtype=float)
        return cls(tuple(names), np.array(lower), np.array(upper), vel, np.array(effort),
                   {"position": "model: joint range ∩ actuator ctrlrange", "effort": "model: actuator forcerange",
                    "velocity": velocity_source})

    def position_violations(self, q: np.ndarray, tol: float = 0.0) -> list[str]:
        return [f"{n}={q[i]:.4f} outside [{self.lower[i]:.4f}, {self.upper[i]:.4f}]"
                for i, n in enumerate(self.names) if not self.lower[i] - tol <= q[i] <= self.upper[i] + tol]

    def trajectory_violations(self, t: np.ndarray, q: np.ndarray, tol: float = 1e-9) -> list[str]:
        """Position limits at every sample and finite-difference velocity limits between samples."""
        out = []
        for k, qk in enumerate(q):
            out += [f"t={t[k]:.3f}s {v}" for v in self.position_violations(qk, tol)]
        if len(t) > 1:
            v = np.diff(q, axis=0) / np.diff(t)[:, None]
            for k, i in zip(*np.nonzero(np.abs(v) > self.max_velocity + tol)):
                out.append(f"t={t[k]:.3f}s {self.names[i]} speed {abs(v[k, i]):.3f} > {self.max_velocity[i]:.3f} rad/s")
        return out


class SafetyFilter:
    """Last line before a backend: clamps each target into its position range and to at most `max_velocity * dt`
    away from the measured position, and reports every clamp. A correct upstream pipeline produces no reports."""

    def __init__(self, limits: JointLimits, dt: float):
        self.limits, self.dt = limits, dt

    def apply(self, cmd: JointCommand, measured: np.ndarray) -> tuple[JointCommand, list[str]]:
        if cmd.names != self.limits.names:
            raise ValueError(f"command joints {cmd.names} != robot joints {self.limits.names}")
        lim, q = self.limits, cmd.position
        violations = lim.position_violations(q)
        out = np.clip(q, lim.lower, lim.upper)
        step = lim.max_velocity * self.dt
        lo, hi = measured - step, measured + step
        for i in np.nonzero((out < lo) | (out > hi))[0]:
            violations.append(f"{lim.names[i]} step {abs(out[i] - measured[i]):.4f} rad > {step[i]:.4f} rad "
                              f"({lim.max_velocity[i]:.2f} rad/s × {self.dt:.4f} s)")
        out = np.clip(out, lo, hi)
        return JointCommand(cmd.names, out), violations
