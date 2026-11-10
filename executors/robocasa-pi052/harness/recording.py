"""Per-call records of S1 requests, commitments and S2 decisions, taken by wrapping the core's plug-in interfaces.

The runtime keeps aggregate stats; tracing needs one record per call. Each wrapper implements the same interface as
what it wraps (`Policy`, `CommitPolicy`, `Decider`) and only appends a dict to its `log`, so the core runs unchanged.
Times are the robot clock (`now`); S1 records carry the request's stage times (`timings_ms`). With a `ModelDelay`
(`--model-delay`) each S1 request and S2 decision is one delayed call, so it costs its wall time in simulated time.

`InteractionCases` saves every question a Jev decision asked the model as a calibration case (`<dir>/<case>/case.json`
+ one PNG per camera: the exact model inputs, the question and its options; `question` is `act`, or `task` for the
second question of a revision) and appends a line to `<dir>/labels.jsonl` with the predicted option and
`"label": null` for a person to fill in; `calibrate_jev.py` fits one `Calibration` per question from them.
"""
import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from rojev.protocol import ObservationSnapshot
from rojev.robot import ActionSpec
from rojev.system1 import ActionChunk, CommitPolicy, Commitment, Continuation, Policy
from rojev.system2 import Verdict
from rojev.system2.jev import JevInteraction
from rojev.system2.jev.chooser import situation

from .model_delay import ModelDelay


class RecordingPolicy(Policy):
    """Logs each request of `inner` (and whether it carried a continuation); accepts a continuation exactly when
    `inner` does and forwards it."""

    def __init__(self, inner: Policy, now: Callable[[], float], delay: ModelDelay | None = None):
        self.inner, self.now, self.log, self.delay = inner, now, [], delay
        self.accepts_continuation = inner.accepts_continuation

    def act(self, obs, subtask: str, timeout_s: float | None = None,
            continuation: Continuation | None = None) -> ActionChunk:
        return self.act_samples(obs, subtask, 1, timeout_s, continuation)[0]

    def act_samples(self, obs, subtask: str, k: int, timeout_s: float | None = None,
                    continuation: Continuation | None = None) -> list[ActionChunk]:
        t0, w0 = self.now(), time.perf_counter()
        extra = (continuation,) if continuation is not None else ()
        try:
            with self.delay.call("s1") if self.delay is not None else nullcontext():
                chunks = (self.inner.act_samples(obs, subtask, k, timeout_s, *extra) if k > 1
                          else [self.inner.act(obs, subtask, timeout_s, *extra)])
        except Exception as e:
            self.log.append({"t_start": t0, "t_end": self.now(), "subtask": subtask, "samples": k,
                             "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"t_start": t0, "t_end": self.now(), "wall_ms": round((time.perf_counter() - w0) * 1000, 1),
                         "subtask": subtask, "samples": k, "instruction": obs.task,
                         "continuation": None if continuation is None else {
                             "steps": len(continuation.actions), "delay_steps": continuation.delay_steps},
                         "timings_ms": chunks[0].timings_ms,
                         "info": {key: v for key, v in chunks[0].info.items() if key != "sample"}})
        return chunks

    def reset(self) -> None:
        self.inner.reset()


class RecordingCommit(CommitPolicy):
    def __init__(self, inner: CommitPolicy, now: Callable[[], float]):
        self.inner, self.now, self.log = inner, now, []
        self.samples = inner.samples

    def plan(self, candidates: list[np.ndarray], spec: ActionSpec, floor: int,
             last: np.ndarray | None) -> Commitment:
        out = self.inner.plan(candidates, spec, floor, last)
        self.log.append({"t": self.now(), "floor": floor, "steps": out.steps, "horizon": int(np.shape(candidates[0])[0]),
                         "reason": out.reason, "chosen": out.chosen, "detail": {k: v for k, v in out.detail.items()},
                         "gripper_close_at": _first_close(out.actions)})
        return out

    def __repr__(self) -> str:
        return repr(self.inner)


class RecordingDecider:
    def __init__(self, inner, now: Callable[[], float], lane: str, delay: ModelDelay | None = None):
        self.inner, self.now, self.lane, self.log, self.delay = inner, now, lane, [], delay
        self.name = inner.name

    def decide(self, snapshot: ObservationSnapshot) -> Verdict:
        t0, w0 = self.now(), time.perf_counter()
        try:
            with self.delay.call(self.lane) if self.delay is not None else nullcontext():
                v = self.inner.decide(snapshot)
        except Exception as e:
            self.log.append({"t_start": t0, "t_end": self.now(), "lane": self.lane, "request": snapshot.request_id,
                             "trigger": snapshot.trigger, "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"t_start": t0, "t_end": self.now(), "lane": self.lane, "request": snapshot.request_id,
                         "trigger": snapshot.trigger, "op": v.operation.kind.value, "subtask": v.operation.subtask,
                         "reason": v.reason, "instruction_rev": snapshot.instruction_rev,
                         "steps_done": len(snapshot.memory.done()), "review": snapshot.review,
                         "deviation": snapshot.deviation is not None, "scores": v.scores,
                         "decide_ms": round((time.perf_counter() - w0) * 1000, 1), "task": v.task})
        return v


def _first_close(actions: np.ndarray) -> int | None:
    closed = np.flatnonzero(np.asarray(actions)[:, 6] >= 0.5)
    return int(closed[0]) if closed.size else None


class InteractionCases:
    """Wraps a `JevInteraction`; each decision that asked the model becomes a calibration case (module docstring)."""

    def __init__(self, inner: JevInteraction, cameras: list[str], out: Path, run: str):
        self.inner, self.cameras, self.out, self.run = inner, cameras, out, run
        self.name = inner.name
        self.saved = 0
        out.mkdir(parents=True, exist_ok=True)

    def decide(self, snapshot: ObservationSnapshot) -> Verdict:
        v = self.inner.decide(snapshot)
        if v.evidence.get("options"):
            self._save(snapshot, "act", v.evidence, v.scores)
        task = v.evidence.get("task")
        if task and task.get("options"):
            self._save(snapshot, "task", task, task.get("scores") or {})
        return v

    def _save(self, s: ObservationSnapshot, question: str, asked: dict, scores: dict[str, float]) -> None:
        case = f"{self.run}-r{s.request_id}-{question}"
        folder = self.out / case
        folder.mkdir(exist_ok=True)
        for cam in self.cameras:
            Image.fromarray(s.capture.observation.images[cam]).save(folder / f"{cam}.png")
        predicted = max(scores, key=scores.get) if scores else None
        (folder / "case.json").write_text(json.dumps({
            "case": case, "question": question, "question_text": asked.get("question"), "t": s.now,
            "trigger": s.trigger, "instruction": s.instruction, "cameras": self.cameras,
            "situation": situation(s, self.cameras), "options": asked["options"], "raw": asked.get("raw"),
            "effects": asked.get("effects"), "predicted": predicted}, indent=1, default=str))
        with (self.out / "labels.jsonl").open("a") as f:
            f.write(json.dumps({"case": case, "question": question, "instruction": s.instruction, "trigger": s.trigger,
                                "said": s.latest_utterance, "predicted": predicted, "label": None}) + "\n")
        self.saved += 1


class RecordingVerifier:
    """Wraps a `Verifier`; logs every check with the simulator's truth for the same step at the same moment (`truth`
    returns None for a step without an authored completion check), so end-state checks can be scored (roadmap T6)."""

    def __init__(self, inner, truth: Callable[[str], bool | None], now: Callable[[], float]):
        self.inner, self.truth, self.now, self.log = inner, truth, now, []
        self.name = inner.name

    def check(self, images, steps: list[str]) -> dict[str, dict[str, float]]:
        t0 = self.now()
        out = self.inner.check(images, steps)
        for step in steps:
            self.log.append({"t": t0, "step": step, "states": out[step], "truth": self.truth(step),
                             "timings_ms": getattr(self.inner, "last_timings_ms", {})})
        return out
