"""Step watcher: turns end-state checks into step-completion events for the owner (spec §4, §9).

On its own thread, every `every_s`: take the running command (`running()`, the runtime's active command) and the
newest capture; if the command moves with a subtask, check it (`Verifier`). When P(done) ≥ `done_threshold` on
`confirmations` consecutive checks of the same command, post one `StepEvent` (source = the verifier's name, detail =
P(done), the observation it was judged on and the checks behind it); then stay quiet for that command. A check that
fails is counted and skipped, never posted. Commands without a subtask (S1's own planner) are not checked: their step
is not known here. `done_threshold` and `confirmations` are declared tuned constants until the verifier is measured
against simulator truth (roadmap T6).
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rojev.protocol import AcceptedGoal, Capture, StepEvent

from .checks import Verifier


@dataclass
class WatchStats:
    checks: int = 0
    errors: int = 0
    last_error: str = ""
    events: int = 0
    check_ms: list[float] = field(default_factory=list)


class StepWatcher:
    def __init__(self, verifier: Verifier, running: Callable[[], AcceptedGoal | None],
                 latest: Callable[[], Capture], post: Callable[[StepEvent], None], now: Callable[[], float],
                 cameras: list[str], every_s: float = 1.0, done_threshold: float = 0.7, confirmations: int = 2):
        self.verifier, self.running, self.latest, self.post, self.now = verifier, running, latest, post, now
        self.cameras, self.every_s = cameras, every_s
        self.done_threshold, self.confirmations = done_threshold, confirmations
        self.stats = WatchStats()
        self._streak: tuple[int, int] = (-1, 0)
        self._reported: set[int] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="step-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def check_once(self) -> StepEvent | None:
        """One check of the running step; returns (and posts) the event if this check completes the evidence."""
        goal = self.running()
        if goal is None or not goal.operation.moves_robot() or goal.operation.subtask is None \
                or goal.cmd_id in self._reported:
            return None
        capture = self.latest()
        step = goal.operation.subtask
        w0 = time.perf_counter()
        try:
            states = self.verifier.check([capture.observation.images[c] for c in self.cameras], [step])[step]
        except Exception as e:
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {e}"
            return None
        self.stats.checks += 1
        self.stats.check_ms.append((time.perf_counter() - w0) * 1000)
        p_done = states.get("done", 0.0)
        cmd, streak = self._streak
        streak = (streak + 1 if cmd == goal.cmd_id else 1) if p_done >= self.done_threshold else 0
        self._streak = (goal.cmd_id, streak)
        if streak < self.confirmations:
            return None
        self._reported.add(goal.cmd_id)
        event = StepEvent(goal.epoch, goal.cmd_id, capture.t, self.verifier.name,
                          {"step": step, "p_done": round(p_done, 3), "obs_id": capture.obs_id, "checks": streak})
        self.stats.events += 1
        self.post(event)
        return event

    def _run(self) -> None:
        while not self._stop.wait(self.every_s):
            self.check_once()
