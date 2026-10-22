"""Baseline deciders for the v3 runtime (evaluation only): what Jev must beat (docs/plan/v3-roadmap.md T8).

  scripted  ScriptedSequence  each authored subtask for a fixed time, in order (open loop, no perception)
  oracle    TruthSequence     next = after the latest subtask whose simulator truth holds now (evaluation ceiling)
  native    NativePlanner     one command with no subtask: S1 (π0.5²) plans its own subtasks from the instruction
  continue  AlwaysContinue    interaction lane: every change (instruction, deviation) is answered with CONTINUE, i.e.
                              keep the current step under the new instruction (the reaffirm baseline for Jev's acts)

Each returns a `Verdict` from an `ObservationSnapshot`; the owner turns it into commands like any decider's.
"""
from collections.abc import Callable

from rojev.protocol import ObservationSnapshot, Operation, OpKind
from rojev.system2 import Verdict


def next_after_latest_done(done_now: list[bool]) -> int:
    """Index of the step after the latest one whose truth holds now: forward on progress, skip what is already done,
    back when progress is undone (object dropped)."""
    return max((i for i, ok in enumerate(done_now) if ok), default=-1) + 1


def _step(subtask: str | None) -> Operation:
    return Operation(OpKind.SWITCH_TO, subtask=subtask)


class ScriptedSequence:
    """Each subtask runs `seconds_per_subtask` after it was activated, then the next; COMPLETE after the last. Ignores
    instruction changes by design: it is the open-loop baseline."""

    name = "scripted"

    def __init__(self, sequence: list[str], seconds_per_subtask: float):
        self.sequence, self.budget, self.index = list(sequence), seconds_per_subtask, 0

    def decide(self, s: ObservationSnapshot) -> Verdict:
        current = s.active_operation.subtask if s.active_operation is not None else None
        if current == self.sequence[min(self.index, len(self.sequence) - 1)] and s.active_since is not None \
                and s.now - s.active_since >= self.budget:
            self.index += 1
        if self.index >= len(self.sequence):
            return Verdict(Operation(OpKind.COMPLETE), "sequence finished", evidence={"index": self.index})
        target = self.sequence[self.index]
        if current == target:
            return Verdict(Operation(OpKind.CONTINUE), f"{self.index + 1}/{len(self.sequence)} within budget",
                           evidence={"index": self.index})
        return Verdict(_step(target), "start" if current is None else "budget spent → next",
                       evidence={"index": self.index})


class TruthSequence:
    """Reads simulator truth (`is_done(i)` for step i, true now): forward on progress, skip what is done, back when
    progress is undone. Never a deployable arm."""

    name = "oracle"

    def __init__(self, sequence: list[str], is_done: Callable[[int], bool]):
        self.sequence, self.is_done = list(sequence), is_done

    def decide(self, s: ObservationSnapshot) -> Verdict:
        now = [bool(self.is_done(i)) for i in range(len(self.sequence))]
        index = next_after_latest_done(now)
        if index >= len(self.sequence):
            return Verdict(Operation(OpKind.COMPLETE), "all steps done (truth)", evidence={"truth": now})
        target = self.sequence[index]
        current = s.active_operation.subtask if s.active_operation is not None else None
        if current == target:
            return Verdict(Operation(OpKind.CONTINUE), "not done yet (truth)", evidence={"truth": now})
        return Verdict(_step(target), "start" if current is None else "truth → step changed",
                       evidence={"truth": now, "index": index})


class AlwaysContinue:
    """Interaction-lane baseline (re-prompting S1): whatever the user said becomes the task text and the current step
    goes on under it (the owner reaffirms it). What Jev's interaction acts must beat on the revision benchmark."""

    name = "always-continue"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        said = s.latest_utterance if s.trigger == "instruction" else None
        return Verdict(Operation(OpKind.CONTINUE), f"{s.trigger}: keep the current step", task=said or None)


class NativePlanner:
    """One moving command without a subtask; S1 plans its own subtasks from the task text (π0.5²'s language head). Asked
    to answer a change itself (one lane, or the revoke baseline), it re-prompts S1: what the user said becomes the task
    text."""

    name = "native"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        said = (s.latest_utterance or None) if s.trigger == "instruction" else None
        if s.active_operation is None or s.active_operation.kind != OpKind.SWITCH_TO:
            return Verdict(_step(None), "hand the task to S1's own planner", task=said)
        return Verdict(Operation(OpKind.CONTINUE), "S1 plans its own subtasks", task=said)
