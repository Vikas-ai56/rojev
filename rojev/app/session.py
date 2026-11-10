"""The one assembly of the runtime for any robot: a robot plug-in (embodiment + backend) and model plug-ins (S1 policy,
planning decider, optional interaction decider, optional done checker) in, one supervised run out.

`Session` builds the execution runtime (seeded with the embodiment's start command), the guard, the control loop, the
S1 runner, a decision worker per lane, the state owner (event-driven planning, owner-owned lease, task memory, episodic
log under `out`), and, with a verifier, the step watcher fed by `SwitchCues` from the applied actions. It knows no
robot, simulator or model by name: only `Embodiment`, `Backend`, `Policy`, `Decider` and `Verifier`.

`run(seconds, say, stop_when)` runs until `seconds` of robot time pass, the owner records the task complete,
`stop_when()` turns true, or the loop fails. Time modes (`SessionConfig.lockstep`):
  lockstep  one thread alternates the owner (every posted event, then its tick), one S1 supply step and one control
            tick, and does not tick while a decision lane or the done checker is busy or an answer is unhandled: model
            computation takes no robot time (how benchmarks are scored).
  real time every part on its own thread against the robot's clock; nothing waits for a model.
Threads start loop → runner → lanes → owner → watcher and stop in reverse; objects built before the run are frozen
for the garbage collector (`gc.freeze`) so a full collection never walks them mid-run. Records are written when the
run ends (`records.write_records`); the video, if asked for, is encoded only then (`video.VideoRecorder`).
"""
import gc
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rojev.control import Backend, ControlLoop, ExecutionRuntime, Guard, SwitchCues, TickRecord
from rojev.owner import EpisodicLog, StateOwner
from rojev.protocol import StepEvent, TaskMemory, UserInstruction, Utterance
from rojev.system1 import Policy, S1Runner
from rojev.system2 import Decider, DecisionWorker
from rojev.system2.verify import StepWatcher, Verifier

from .config import SessionConfig
from .embodiment import Embodiment
from .records import RecordedDecider, RecordedPolicy, write_records
from .video import VideoRecorder

EPOCH = 1
VIDEO_FPS = 10
LOCKSTEP_WAIT_S = 0.002
REALTIME_POLL_S = 0.01


@dataclass(frozen=True)
class RunResult:
    ended: str
    sim_s: float
    wall_s: float
    ticks: int
    complete: bool
    memory: TaskMemory
    out: Path


class Session:
    def __init__(self, embodiment: Embodiment, backend: Backend, policy: Policy, planner: Decider, instruction: str,
                 config: SessionConfig, out: Path, interaction: Decider | None = None,
                 verifier: Verifier | None = None, video: bool = False,
                 observers: Sequence[Callable[[TickRecord], None]] = ()):
        self.embodiment, self.config, self.instruction = embodiment, config, instruction
        self.error: str | None = None
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        spec = embodiment.spec
        start = None if embodiment.start is None else np.asarray(embodiment.start, dtype=np.float64).copy()
        self.runtime = ExecutionRuntime(spec, epoch=EPOCH, last=start, blend_s=config.blend_s, stop_s=config.stop_s)
        self.guard = Guard(self.runtime)
        self.observers = tuple(observers)
        self.ticks: list[dict] = []
        self.utterances: list[dict] = []
        self.video = VideoRecorder(self.out / "view.mp4", VIDEO_FPS) if video else None
        self._video_every = max(1, round(spec.control_hz / VIDEO_FPS))
        self.loop = ControlLoop(backend, self.guard, spec, realtime=not config.lockstep, on_tick=self._on_tick)
        self.policy = RecordedPolicy(policy, self.loop.now)
        self.runner = S1Runner(self.runtime, self.policy, self.loop.latest, self.loop.now, on_step=self._on_s1_step,
                               on_plan=self._post, clock_runs_while_thinking=not config.lockstep)
        self.planning = RecordedDecider(planner, self.loop.now, "planning")
        self.planning_worker = DecisionWorker(self.planning, self._post, self.loop.now, name="planning")
        self.interaction = RecordedDecider(interaction, self.loop.now, "interaction") if interaction else None
        self.interaction_worker = (DecisionWorker(self.interaction, self._post, self.loop.now, name="interaction")
                                   if self.interaction is not None else None)
        self.owner = StateOwner(self.runtime, self.guard, self.planning_worker, self.loop.latest, self.loop.now,
                                instruction, lease_s=config.lease_s, watchdog_s=config.watchdog_s,
                                step_budget_s=config.step_budget_s, liveness_s=config.liveness_s,
                                on_instruction_change=config.on_change, review_s=config.review_s,
                                interaction=self.interaction_worker,
                                interaction_timeout_s=config.interaction_timeout_s, say=self._say,
                                dialogue_turns=config.dialogue_turns, hold_recheck_s=config.hold_recheck_s,
                                episodic=EpisodicLog(self.out / "episodes.jsonl"))
        self.watcher = (StepWatcher(verifier, self.runtime.active_goal, self.loop.latest, self.owner.post,
                                    self.loop.now, list(embodiment.cameras), every_s=config.verify_every_s,
                                    done_threshold=config.done_threshold, confirmations=config.confirmations,
                                    recheck_s=config.recheck_s) if verifier is not None else None)
        self.cues = SwitchCues(spec, self.watcher.cue) if self.watcher is not None else None

    def _post(self, event) -> None:
        self.owner.post(event)

    def _on_s1_step(self, event: StepEvent) -> None:
        if self.watcher is not None:
            self.watcher.claim(event)
        else:
            self.owner.post(event)

    def _say(self, u: Utterance) -> None:
        self.utterances.append({"t": round(u.t, 3), "template": u.template, "objects": list(u.objects),
                                "cmd": u.cmd_id})

    def _on_tick(self, r: TickRecord) -> None:
        t = r.tick
        self.ticks.append({"t": round(t.t, 3), "source": t.source, "mode": t.mode.value, "cmd": t.cmd_id,
                           "skipped": t.skipped, "held": list(t.held_switches), "violations": list(t.violations),
                           "step_ms": round(r.step_ms, 1), "action": np.round(t.action, 4).tolist()})
        if self.cues is not None:
            self.cues(r)
        for observer in self.observers:
            observer(r)
        if self.video is not None and t.index % self._video_every == 0:
            images = r.after.observation.images
            self.video.add([images[c] for c in self.embodiment.cameras], self._panel(t.t))

    def _panel(self, t: float) -> list[str]:
        goal = self.owner.desired
        decisions = self.planning.log + (self.interaction.log if self.interaction else [])
        latest = max(decisions, key=lambda d: d["t_start"]) if decisions else None
        return [f"t {t:6.2f} s  {self.embodiment.name}  task: {self.owner.instruction[:90]}",
                "command: " + ("none" if goal is None else
                               f"{goal.cmd_id} {goal.operation.kind.value} {goal.operation.subtask or ''}"),
                "decision: " + ("none" if latest is None else
                                f"[{latest['lane']}/{latest['trigger']}] {latest.get('op', 'error')} "
                                f"{latest.get('subtask') or ''} {latest.get('reason', latest.get('error', ''))}"),
                "done: " + (", ".join(self.owner.memory.done()) or "-")]

    def _lanes(self) -> tuple:
        lanes = tuple(w for w in (self.planning_worker, self.interaction_worker) if w is not None)
        return lanes + ((self.watcher,) if self.watcher is not None else ())

    def run(self, seconds: float, say: Sequence[tuple[float, str]] = (),
            stop_when: Callable[[], bool] | None = None) -> RunResult:
        lockstep = self.config.lockstep
        lanes = tuple(w for w in (self.planning_worker, self.interaction_worker) if w is not None)
        watcher = (self.watcher,) if self.watcher is not None else ()
        threads = lanes + watcher if lockstep else (self.loop, self.runner) + lanes + (self.owner,) + watcher
        pending = sorted(say)
        ended, self.error = "seconds", None
        self._write_config(seconds, say)
        gc.collect()
        gc.freeze()
        w0 = time.perf_counter()
        for thread in threads:
            thread.start()
        try:
            while (self.loop.now() if lockstep else time.perf_counter() - w0) < seconds:
                if self.loop.error is not None:
                    ended, self.error = "error", repr(self.loop.error)
                    break
                while pending and self.loop.now() >= pending[0][0]:
                    self.owner.post(UserInstruction(pending.pop(0)[1]))
                if self.owner.complete:
                    ended = "complete"
                    break
                if stop_when is not None and stop_when():
                    ended = "stop_when"
                    break
                if not lockstep:
                    time.sleep(REALTIME_POLL_S)
                    continue
                self.owner.step()
                if any(lane.busy for lane in self._lanes()) or self.owner.has_events:
                    time.sleep(LOCKSTEP_WAIT_S)
                    continue
                self.runner.step()
                self.loop.run_ticks(1)
        except Exception as e:
            ended, self.error = "error", f"{type(e).__name__}: {e}"
        finally:
            for thread in reversed(threads):
                try:
                    thread.stop()
                except Exception as e:
                    ended, self.error = "error", self.error or f"{type(e).__name__}: {e}"
            self.owner.stop()
            gc.unfreeze()
        wall = time.perf_counter() - w0
        result = RunResult(ended, round(self.loop.now(), 3), round(wall, 3), self.loop.ticks, self.owner.complete,
                           self.owner.memory, self.out)
        self._write(result)
        return result

    def _write_config(self, seconds: float, say: Sequence[tuple[float, str]]) -> None:
        about = {"embodiment": self.embodiment.name, "cameras": list(self.embodiment.cameras),
                 "action_dims": [d.name for d in self.embodiment.spec.dims], "control_hz": self.embodiment.spec.control_hz,
                 "instruction": self.instruction, "seconds": seconds, "say": [list(s) for s in say],
                 "planner": self.planning.name, "interaction": self.interaction.name if self.interaction else None,
                 "verifier": self.watcher.verifier.name if self.watcher else None, "runtime": asdict(self.config)}
        (self.out / "config.json").write_text(json.dumps(about, indent=1, default=str))

    def _write(self, result: RunResult) -> None:
        events = self.owner.record.events
        stats = self.runner.stats
        summary = {"ended": result.ended, "sim_s": result.sim_s, "wall_s": result.wall_s, "ticks": result.ticks,
                   "realtime_factor": round(result.sim_s / result.wall_s, 3) if result.wall_s else None,
                   "complete": result.complete, "starved_ticks": self.runtime.starved_ticks,
                   "violations": sum(len(t["violations"]) for t in self.ticks),
                   "error": self.error,
                   "memory": [{"step": r.step, "status": r.status.value, "cmd": r.cmd_id, "t": round(r.t, 3),
                               "evidence": r.evidence, "checked": r.checked} for r in result.memory.records],
                   "commands": [{"t": e["t"], "cmd": e["cmd"], "op": e["op"], "subtask": e["subtask"],
                                 "lane": e.get("lane"), "reason": e.get("reason")}
                                for e in events if e["kind"] in ("command", "reaffirm")],
                   "ack_counts": {s: sum(e["kind"] == "ack" and e["status"] == s for e in events)
                                  for s in ("activated", "revoked", "rejected", "chunk_rejected", "expired",
                                            "dropped")},
                   "liveness_lapses": sum(e["kind"] == "liveness lapsed" for e in events),
                   "s1": {"requests": stats.requests, "accepted": stats.accepted, "errors": stats.errors,
                          "last_error": stats.last_error},
                   "lanes": {w.name: {"decisions": w.stats.decisions, "errors": w.stats.errors}
                             for w in (self.planning_worker, self.interaction_worker) if w is not None},
                   "verify": None if self.watcher is None else {
                       "checks": self.watcher.stats.checks, "events": self.watcher.stats.events,
                       "cues": self.watcher.stats.cues, "errors": self.watcher.stats.errors}}
        decisions = self.planning.log + (self.interaction.log if self.interaction else [])
        write_records(self.out, self.owner.record.events, self.policy.log, decisions, self.utterances, summary)
        with (self.out / "ticks.jsonl").open("w") as f:
            for row in self.ticks:
                f.write(json.dumps(row) + "\n")
        if self.video is not None:
            self.video.close()
