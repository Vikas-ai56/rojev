"""Guard: the last layer before the robot, independent of whether anything above it is correct or alive.

Runs inside the control loop only (docs/plan/v3-v1-spec.md §6). Every tick it asks the execution runtime for the
action; if the runtime raises, if no valid chunk action arrived for `watchdog_s` while running ("starved"), or if the
owner stopped sending heartbeats for `owner_timeout_s` ("owner lost"), the robot holds still. A starvation hold keeps
the command (it resumes with its next chunk); an owner-lost or runtime-error hold drops it, so authority returns only
through a new command from the owner. `stop()` needs nothing but the control loop. Bound clamps are already applied
and reported by the runtime (`ActionSpec.limit`).
"""
from dataclasses import dataclass

from .execution import ExecutionRuntime, Mode, TickResult


@dataclass
class Guard:
    runtime: ExecutionRuntime
    watchdog_s: float = 0.5
    owner_timeout_s: float = 2.0
    last_heartbeat: float | None = None
    errors: int = 0

    def heartbeat(self, now: float) -> None:
        self.last_heartbeat = now

    def stop(self, now: float, reason: str = "stop requested") -> None:
        self.runtime.stop(now, reason)

    def step(self, now: float) -> TickResult:
        rt = self.runtime
        if self.last_heartbeat is not None and now - self.last_heartbeat > self.owner_timeout_s:
            rt.hold(now, "owner lost", drop_commands=True)
        elif rt.mode == Mode.RUNNING and now - (rt.last_valid_t if rt.last_valid_t is not None else now) \
                > self.watchdog_s:
            rt.hold(now, "starved", drop_commands=False)
        try:
            return rt.tick(now)
        except Exception as e:
            self.errors += 1
            rt.hold(now, f"runtime error: {type(e).__name__}: {e}", drop_commands=True)
            return rt.tick(now)
