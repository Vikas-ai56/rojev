"""What a run leaves behind: every request to S1, every decision of each lane, every control tick, everything the robot
said, the owner's record, and a summary; written under the run's directory by `Session`.

`RecordedPolicy` and `RecordedDecider` wrap the models and log each call (robot-clock start and end, wall time, what
was asked, what came back, or the error) without changing what they return. `write_records` serialises them with the
owner's record and the run's summary. Files: `owner.jsonl`, `s1_requests.jsonl`, `decisions.jsonl`, `ticks.jsonl`,
`utterances.jsonl`, `summary.json` (plus `config.json`, `episodes.jsonl` and `view.mp4` written by the session and the
owner). The field names are what tools that read runs rely on.
"""
import json
import time
from collections.abc import Callable
from pathlib import Path

from rojev.protocol import ObservationSnapshot
from rojev.system1 import ActionChunk, Continuation, Policy
from rojev.system2 import Verdict


class RecordedPolicy(Policy):
    def __init__(self, inner: Policy, now: Callable[[], float]):
        self.inner, self.now, self.log = inner, now, []
        self.accepts_continuation = inner.accepts_continuation

    def act(self, obs, subtask: str, timeout_s: float | None = None,
            continuation: Continuation | None = None) -> ActionChunk:
        return self.act_samples(obs, subtask, 1, timeout_s, continuation)[0]

    def act_samples(self, obs, subtask: str, k: int, timeout_s: float | None = None,
                    continuation: Continuation | None = None) -> list[ActionChunk]:
        t0, w0 = self.now(), time.perf_counter()
        extra = (continuation,) if continuation is not None else ()
        try:
            chunks = (self.inner.act_samples(obs, subtask, k, timeout_s, *extra) if k > 1
                      else [self.inner.act(obs, subtask, timeout_s, *extra)])
        except Exception as e:
            self.log.append({"t_start": t0, "t_end": self.now(), "subtask": subtask, "samples": k,
                             "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"t_start": t0, "t_end": self.now(), "wall_ms": round((time.perf_counter() - w0) * 1000, 1),
                         "subtask": subtask, "instruction": obs.task, "samples": k,
                         "actions": len(chunks[0].actions), "timings_ms": chunks[0].timings_ms})
        return chunks

    def reset(self) -> None:
        self.inner.reset()


class RecordedDecider:
    def __init__(self, inner, now: Callable[[], float], lane: str):
        self.inner, self.now, self.lane, self.log = inner, now, lane, []
        self.name = inner.name

    def decide(self, snapshot: ObservationSnapshot) -> Verdict:
        t0, w0 = self.now(), time.perf_counter()
        try:
            v = self.inner.decide(snapshot)
        except Exception as e:
            self.log.append({"t_start": t0, "t_end": self.now(), "lane": self.lane, "request": snapshot.request_id,
                             "trigger": snapshot.trigger, "error": f"{type(e).__name__}: {e}"})
            raise
        self.log.append({"t_start": t0, "t_end": self.now(), "lane": self.lane, "request": snapshot.request_id,
                         "trigger": snapshot.trigger, "op": v.operation.kind.value, "subtask": v.operation.subtask,
                         "reason": v.reason, "instruction_rev": snapshot.instruction_rev,
                         "steps_done": snapshot.memory.done(), "scores": v.scores, "task": v.task,
                         "decide_ms": round((time.perf_counter() - w0) * 1000, 1)})
        return v


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_records(out: Path, owner_events: list[dict], s1_requests: list[dict], decisions: list[dict],
                  utterances: list[dict], summary: dict) -> None:
    write_jsonl(out / "owner.jsonl", owner_events)
    write_jsonl(out / "s1_requests.jsonl", s1_requests)
    write_jsonl(out / "decisions.jsonl", sorted(decisions, key=lambda d: d["t_start"]))
    write_jsonl(out / "utterances.jsonl", utterances)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
