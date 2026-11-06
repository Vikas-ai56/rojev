"""Lockstep with each model's measured delay added back (`realtime.py --lockstep --model-delay`).

Plain lockstep stops simulated time while any model thinks, so S1 and S2 latency cost nothing and a slow or late
answer can never hurt the robot. Here every model call costs simulated time equal to its own wall time, while simulator
stepping costs none: the loop ticks freely while no call is open, never ticks past `start + wall time so far` of an
open call, and a finished call returns to its caller only once simulated time has reached `start + its wall time`.
The answer therefore lands as late, in simulated time, as it took, and the robot runs whatever it had queued meanwhile
(or starves), as it would on hardware, however slow or fast the simulator steps. Answers are applied at the first
control tick at or after their due time, as in real time.

The S1 runner and the S2 lanes run on their own threads, as in real time; `realtime.py` owns which lanes are waiting
(a request about to start, a call open, or an answer not yet handed over) and asks `allows` before each tick. A call
that raises (a timeout, an error) also costs its wall time. `log` keeps one record per call: lane, simulated start,
wall time, due time and the simulated time it returned at. `release` frees waiting callers when the run ends.
"""
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class OpenCall:
    start: float
    wall_start: float
    due: float | None = None


class ModelDelay:
    def __init__(self, now: Callable[[], float]):
        self.now = now
        self.log: list[dict] = []
        self._open: dict[str, OpenCall] = {}
        self._changed = threading.Condition()
        self._released = False

    @contextmanager
    def call(self, lane: str) -> Iterator[None]:
        with self._changed:
            opened = OpenCall(self.now(), time.perf_counter())
            self._open[lane] = opened
        try:
            yield
        finally:
            with self._changed:
                wall = time.perf_counter() - opened.wall_start
                opened.due = opened.start + wall
                self._changed.notify_all()
                while not self._released and self.now() < opened.due:
                    self._changed.wait(0.05)
                del self._open[lane]
                self.log.append({"lane": lane, "t_start": round(opened.start, 3), "wall_s": round(wall, 3),
                                 "t_due": round(opened.due, 3), "t_returned": round(self.now(), 3)})

    def bound(self, lane: str) -> float | None:
        """How far simulated time may run while `lane` has a call open; None when it has none."""
        with self._changed:
            opened = self._open.get(lane)
            if opened is None:
                return None
            return opened.due if opened.due is not None else opened.start + time.perf_counter() - opened.wall_start

    def allows(self, lane: str, waiting: bool) -> bool:
        """May the loop tick now, given whether `lane` has a request or an answer in flight?"""
        if not waiting:
            return True
        bound = self.bound(lane)
        return bound is not None and self.now() < bound

    def ticked(self) -> None:
        with self._changed:
            self._changed.notify_all()

    def release(self) -> None:
        with self._changed:
            self._released = True
            self._changed.notify_all()
