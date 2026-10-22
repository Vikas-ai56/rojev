"""Per-call records of S1 requests, commitments and S2 decisions, taken by wrapping the core's plug-in interfaces.

The runtime keeps aggregate stats; tracing needs one record per call. Each wrapper implements the same interface as
what it wraps (`Policy`, `CommitPolicy`, `Decider`) and only appends a dict to its `log`, so the core runs unchanged.
Times are the robot clock (`now`).

`InteractionCases` saves every Jev decision that asked the model as a calibration case (`<dir>/<case>/case.json` +
one PNG per camera, the exact model inputs and legal options) and appends a line to `<dir>/labels.jsonl` with the
predicted act and `"label": null` for a person to fill in; `calibrate_jev.py` fits the `Calibration` from them.
"""
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from rojev.protocol import ObservationSnapshot
from rojev.robot import ActionSpec
from rojev.system1 import ActionChunk, CommitPolicy, Commitment, Policy
from rojev.system2 import Verdict
from rojev.system2.jev import JevInteraction
from rojev.system2.jev.chooser import situation


class RecordingPolicy(Policy):
    def __init__(self, inner: Policy, now: Callable[[], float]):
        self.inner, self.now, self.log = inner, now, []

    def act(self, obs, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        return self.act_samples(obs, subtask, 1, timeout_s)[0]

    def act_samples(self, obs, subtask: str, k: int, timeout_s: float | None = None) -> list[ActionChunk]:
        t0, w0 = self.now(), time.perf_counter()
        try:
            chunks = (self.inner.act_samples(obs, subtask, k, timeout_s) if k > 1
                      else [self.inner.act(obs, subtask, timeout_s)])
        except Exception as e:
            self.log.append({"t_start": t0, "t_end": self.now(), "subtask": subtask, "samples": k,
                             "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"t_start": t0, "t_end": self.now(), "wall_ms": round((time.perf_counter() - w0) * 1000, 1),
                         "subtask": subtask, "samples": k, "instruction": obs.task,
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
    def __init__(self, inner, now: Callable[[], float], lane: str):
        self.inner, self.now, self.lane, self.log = inner, now, lane, []
        self.name = inner.name

    def decide(self, snapshot: ObservationSnapshot) -> Verdict:
        t0 = self.now()
        try:
            v = self.inner.decide(snapshot)
        except Exception as e:
            self.log.append({"t_start": t0, "t_end": self.now(), "lane": self.lane, "request": snapshot.request_id,
                             "trigger": snapshot.trigger, "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"t_start": t0, "t_end": self.now(), "lane": self.lane, "request": snapshot.request_id,
                         "trigger": snapshot.trigger, "op": v.operation.kind.value, "subtask": v.operation.subtask,
                         "reason": v.reason, "instruction_rev": snapshot.instruction_rev,
                         "keyframes": len(snapshot.keyframes), "review": snapshot.review,
                         "deviation": snapshot.deviation is not None, "scores": v.scores})
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
        options = v.evidence.get("options")
        if options:
            self._save(snapshot, options, v)
        return v

    def _save(self, s: ObservationSnapshot, options: dict[str, str], v: Verdict) -> None:
        case = f"{self.run}-r{s.request_id}"
        folder = self.out / case
        folder.mkdir(exist_ok=True)
        for cam in self.cameras:
            Image.fromarray(s.capture.observation.images[cam]).save(folder / f"{cam}.png")
        predicted = max(v.scores, key=v.scores.get) if v.scores else None
        (folder / "case.json").write_text(json.dumps({
            "case": case, "t": s.now, "trigger": s.trigger, "instruction": s.instruction, "cameras": self.cameras,
            "situation": situation(s, self.cameras), "options": options, "raw": v.evidence.get("raw"),
            "predicted": predicted}, indent=1, default=str))
        with (self.out / "labels.jsonl").open("a") as f:
            f.write(json.dumps({"case": case, "instruction": s.instruction, "trigger": s.trigger,
                                "predicted": predicted, "label": None}) + "\n")
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
