"""Control loop: drives any robot backend at its control rate on its own thread, never waiting for a model.

Every tick the guard produces the action, the backend applies it and returns the next observation, which replaces the
previous one in a latest-value slot (`latest()`); slow consumers (S1, S2) read the newest observation, never a backlog.
Time is the robot's. A backend that owns a clock (`backend.clock`: a `rojev.robot` robot, simulated in any time mode or
real) paces the loop with `Rate` on that clock. A backend without one (an environment that advances only when stepped,
e.g. RoboCasa) is stepped by the loop: tick k is due at t = k · dt, paced to the wall clock with `realtime`, back to
back without it. Either way the loop reports how late ticks start (`lag_s`, `max_lag_s`) instead of skipping time.
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rojev.protocol import Capture, Observation
from rojev.robot import ActionSpec, Rate

from .execution import TickResult
from .guard import Guard


class Backend(Protocol):
    """A robot or environment: `observe` returns the current observation, `step` applies one action and returns the
    next. May expose `clock` (a `rojev.robot.Clock`) when it keeps its own time."""

    def observe(self) -> Observation: ...

    def step(self, action) -> Observation: ...


@dataclass(frozen=True)
class TickRecord:
    """One tick: the guard's result, the observation the action was applied to (`before`) and the one the backend
    returned after it (`after`), the wall time the backend step took, and how late the tick started."""

    tick: TickResult
    before: Capture
    after: Capture
    step_ms: float
    lag_s: float


class ControlLoop:
    def __init__(self, backend: Backend, guard: Guard, spec: ActionSpec, realtime: bool = True,
                 on_tick: Callable[[TickRecord], None] | None = None):
        self.backend, self.guard, self.spec, self.realtime, self.on_tick = backend, guard, spec, realtime, on_tick
        self.clock = getattr(backend, "clock", None)
        self.lag_s = self.max_lag_s = 0.0
        self.ticks = 0
        self.error: BaseException | None = None
        self._latest = Capture(0, self.now(), backend.observe())
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._wall0 = 0.0

    def now(self) -> float:
        """Robot-clock time: the backend's own clock, else the due time of the next tick."""
        return self.clock.now() if self.clock is not None else self.ticks * self.spec.dt

    def latest(self) -> Capture:
        with self._lock:
            return self._latest

    def start(self) -> None:
        self._running = True
        self._wall0 = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="control-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self.error is not None:
            raise self.error

    def run_ticks(self, n: int) -> None:
        """Run `n` ticks on the caller's thread (lockstep use)."""
        for _ in range(n):
            self._tick()

    def _run(self) -> None:
        try:
            if self.clock is not None:
                rate = Rate(self.clock, self.spec.control_hz)
                while self._running:
                    self._tick()
                    self.lag_s = rate.sleep()
                    self.max_lag_s = max(self.max_lag_s, self.lag_s)
                return
            while self._running:
                if self.realtime:
                    due = self._wall0 + self.ticks * self.spec.dt
                    wait = due - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                    self.lag_s = max(0.0, time.perf_counter() - due)
                    self.max_lag_s = max(self.max_lag_s, self.lag_s)
                self._tick()
        except BaseException as e:
            self.error = e
            self._running = False

    def _tick(self) -> None:
        before = self.latest()
        t = self.now()
        result = self.guard.step(t)
        w0 = time.perf_counter()
        obs = self.backend.step(result.action)
        step_ms = (time.perf_counter() - w0) * 1000
        self.ticks += 1
        with self._lock:
            self._latest = after = Capture(self._latest.obs_id + 1, self.now(), obs)
        if self.on_tick is not None:
            self.on_tick(TickRecord(result, before, after, step_ms, self.lag_s))
