"""Evaluation baselines (not the supervisor): arms B and E for comparing against arm D (docs/paper/method.md §6).

Both use an authored subtask sequence. E additionally reads simulator ground truth, so it exists only in evaluation:
it is the reference ceiling ("perfect subtask choice") that isolates the executor. Nothing here ships with the
supervisor; the deployable arm decides from images without per-subtask checks.
  B  FixedSequence   advance after a fixed number of chunks per subtask (open loop)
  E  OracleSequence  next = after the latest subtask whose ground-truth check holds now
"""
from collections.abc import Callable

from rojev.eval.deciders import next_after_latest_done

from .arms import Decision, DecisionContext, SubtaskSource


class FixedSequence(SubtaskSource):
    """Arm B: run each subtask for exactly `chunks_per_subtask` chunks, in order."""
    name = "B-fixed"

    def __init__(self, sequence: list[str], chunks_per_subtask: int):
        self.sequence, self.budget, self.index = list(sequence), chunks_per_subtask, 0

    def decide(self, ctx: DecisionContext) -> Decision:
        if ctx.current is not None and ctx.chunks_on_current >= self.budget:
            self.index += 1
        if self.index >= len(self.sequence):
            return Decision(None, "sequence finished", {"index": self.index})
        reason = "start" if ctx.current is None else ("budget spent → next" if self.sequence[self.index] != ctx.current
                                                       else f"chunk {ctx.chunks_on_current + 1}/{self.budget}")
        return Decision(self.sequence[self.index], reason, {"index": self.index, "budget_chunks": self.budget})


class OracleSequence(SubtaskSource):
    """Arm E: next subtask = the one after the latest subtask whose ground-truth check holds *now*.

    `is_done(i)` reads the world's current state (held, inside, closed, on). Moves forward on progress, skips what is
    already done, and goes back when progress is undone (object dropped → "pick up" again). First live run
    (2026-10-18) showed the forward-only version kept saying "place …" after the steak was dropped.
    """
    name = "E-oracle"

    def __init__(self, sequence: list[str], is_done: Callable[[int], bool]):
        self.sequence, self.is_done, self.index = list(sequence), is_done, 0

    def decide(self, ctx: DecisionContext) -> Decision:
        now = [bool(self.is_done(i)) for i in range(len(self.sequence))]
        previous, self.index = self.index, next_after_latest_done(now)
        if self.index >= len(self.sequence):
            return Decision(None, "all subtasks done (ground truth)", {"index": self.index, "now": now})
        if ctx.current is None:
            reason = "start"
        elif self.index > previous:
            reason = "done (ground truth) → next"
        elif self.index < previous:
            reason = "progress undone (ground truth) → back"
        else:
            reason = "not done yet"
        return Decision(self.sequence[self.index], reason, {"index": self.index, "now": now})
