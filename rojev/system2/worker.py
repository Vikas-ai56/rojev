"""S2 decision worker: runs a decider (Jev, or a baseline arm) on its own thread and posts its proposal to the owner.

The owner asks with an `ObservationSnapshot`; only the newest request is kept (an older unstarted one is replaced, so a
slow decider never works through a backlog). The decider returns a `Verdict`; the worker turns it into a
`DecisionProposal` stamped with the snapshot's epoch, request id, instruction revision, base command and observation,
plus an expiry, so the owner can refuse it if the world or the instruction moved on meanwhile. A decider error is
counted and posted as a `DecisionFailure` for that request, so whatever waits on it is released at once; the robot
keeps doing what it was doing. `name` labels the worker's lane (thread name, logs).
"""
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from rojev.protocol import DecisionFailure, DecisionProposal, ObservationSnapshot

from .decider import Decider


@dataclass
class S2Stats:
    decisions: int = 0
    errors: int = 0
    last_error: str = ""
    decide_ms: list[float] = field(default_factory=list)
    replaced: int = 0


class DecisionWorker:
    def __init__(self, decider: Decider, post: Callable[[DecisionProposal | DecisionFailure], None],
                 now: Callable[[], float], proposal_ttl_s: float = 2.0, name: str = "s2"):
        self.decider, self.post, self.now, self.proposal_ttl_s = decider, post, now, proposal_ttl_s
        self.name = name
        self.stats = S2Stats()
        self._next: ObservationSnapshot | None = None
        self._busy = False
        self._cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def busy(self) -> bool:
        with self._cond:
            return self._busy or self._next is not None

    def request(self, snapshot: ObservationSnapshot) -> None:
        with self._cond:
            if self._next is not None:
                self.stats.replaced += 1
            self._next = snapshot
            self._cond.notify()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def decide(self, snap: ObservationSnapshot) -> DecisionProposal | DecisionFailure:
        """Run the decider on one snapshot (also usable synchronously)."""
        w0 = time.perf_counter()
        try:
            v = self.decider.decide(snap)
        except Exception as e:
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {e}"
            return DecisionFailure(snap.epoch, snap.request_id, self.stats.last_error)
        ms = (time.perf_counter() - w0) * 1000
        self.stats.decisions += 1
        self.stats.decide_ms.append(ms)
        return DecisionProposal(snap.epoch, snap.request_id, snap.capture.obs_id, snap.capture.t, snap.instruction_rev,
                                snap.active_cmd, 0, v.operation, v.scores, v.preconditions,
                                self.now() + self.proposal_ttl_s,
                                {**v.evidence, "reason": v.reason, "trigger": snap.trigger, "decide_ms": round(ms, 1),
                                 "decider": self.decider.name}, v.task)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._running and self._next is None:
                    self._cond.wait()
                if not self._running:
                    return
                snap, self._next, self._busy = self._next, None, True
            outcome = self.decide(snap)
            with self._cond:
                self._busy = False
            self.post(outcome)
