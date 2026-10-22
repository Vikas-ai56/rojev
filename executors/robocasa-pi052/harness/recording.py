"""Per-call records of S1 requests, commitments and S2 decisions, taken by wrapping the core's plug-in interfaces.

The runtime keeps aggregate stats; tracing needs one record per call. Each wrapper implements the same interface as
what it wraps (`Policy`, `CommitPolicy`, `Decider`) and only appends a dict to its `log`, so the core runs unchanged.
Times are the robot clock (`now`).
"""
import time
from collections.abc import Callable

import numpy as np

from rojev.protocol import ObservationSnapshot
from rojev.robot import ActionSpec
from rojev.system1 import ActionChunk, CommitPolicy, Commitment, Policy
from rojev.system2 import Verdict


class RecordingPolicy(Policy):
    def __init__(self, inner: Policy, now: Callable[[], float]):
        self.inner, self.now, self.log = inner, now, []

    def act(self, obs, subtask: str) -> ActionChunk:
        return self.act_samples(obs, subtask, 1)[0]

    def act_samples(self, obs, subtask: str, k: int) -> list[ActionChunk]:
        t0, w0 = self.now(), time.perf_counter()
        try:
            chunks = self.inner.act_samples(obs, subtask, k) if k > 1 else [self.inner.act(obs, subtask)]
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
                         "deviation": snapshot.deviation is not None})
        return v


def _first_close(actions: np.ndarray) -> int | None:
    closed = np.flatnonzero(np.asarray(actions)[:, 6] >= 0.5)
    return int(closed[0]) if closed.size else None
