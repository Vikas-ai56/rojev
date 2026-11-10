"""Task memory: what the robot knows about the task in progress, written only by the state owner and read by every
decider through its snapshot (docs/target-system.md, Memory).

A `TaskMemory` is immutable: the owner replaces it on every write, so a snapshot holds the memory as it was when the
decision was asked. Its step ledger is append-only: each `StepRecord` says what was learnt about one step, when, from
whom and on which observation; the latest record of a step is its current status.
  DONE    the step was confirmed done by a checked step event, or judged done when a decider moved past it;
          `checked` says whether an end-state check confirmed it (a decider moving on by its own judgement is not a
          check).
  UNDONE  a decider chose a step again after it was recorded done: the scene contradicted the memory (a cube dropped,
          a drawer closed again), so the step is owed again.
`cmd_id` is the command the record was learnt under (for UNDONE, the command running when the decider chose the step
again). Failed attempts are not recorded yet: nothing in the loop reports a failure as such.
"""
from dataclasses import dataclass, field
from enum import Enum

from .observation import Capture


class StepStatus(str, Enum):
    DONE = "done"
    UNDONE = "undone"


@dataclass(frozen=True)
class StepRecord:
    step: str
    status: StepStatus
    cmd_id: int | None
    instruction_rev: int
    capture: Capture
    evidence: str
    checked: bool
    t: float


@dataclass(frozen=True)
class TaskMemory:
    goal: str
    records: tuple[StepRecord, ...] = field(default_factory=tuple)

    def with_record(self, record: StepRecord) -> "TaskMemory":
        return TaskMemory(self.goal, self.records + (record,))

    def with_goal(self, goal: str) -> "TaskMemory":
        return TaskMemory(goal, self.records)

    def status(self, step: str) -> StepStatus | None:
        return next((r.status for r in reversed(self.records) if r.step == step), None)

    def done(self) -> list[str]:
        """Steps whose latest record is DONE, in the order they were first done."""
        order = list(dict.fromkeys(r.step for r in self.records))
        return [s for s in order if self.status(s) == StepStatus.DONE]

    def checked_done(self) -> set[str]:
        """Steps whose latest record is DONE and was confirmed by an end-state check."""
        latest = {r.step: r for r in self.records}
        return {s for s, r in latest.items() if r.status == StepStatus.DONE and r.checked}
