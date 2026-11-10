"""Planning-lane deciders: `StepPlanner` chooses each step for an S1 that follows step text; `CommitVerifyPlanner`
commits a frozen S1 that plans its own subtasks (π0.5²).

`StepPlanner`: on each planning trigger it asks a `NextStepModel` (the S2 planner VLM, e.g. Embodied-R1.5 behind a
remote client) for the immediate next step, given the goal, the steps the task memory records as done (the ledger the
model reads as its "last steps") and the allowed steps (the robot's skill list plus COMPLETE_STEP), on the newest
images of `cameras`. A step other than the running one becomes `SWITCH_TO` that step; the running one, `CONTINUE`;
COMPLETE_STEP, `COMPLETE`; an answer outside the allowed steps keeps what runs (`CONTINUE`, the reason logged), so a
malformed answer never moves the robot. Choosing a step the memory records as done is how the planner says the scene
contradicts the memory: the owner records it UNDONE and the step is redone.

`CommitVerifyPlanner`: commit and verify. S1's own subtask head is memoryless (trained on task + image only), so it
flips between subtasks when the image is ambiguous (live: freezer ↔ fridge 18 times in 60 s). S1 is also steerable by
subtask text. So:
  1. With no command, or after a release: one command without a subtask; S1 plans (its choice arrives as
     `snapshot.planned`).
  2. S1 planned a subtask not yet verified done: commit, i.e. `SWITCH_TO` that text. S1 now follows it and its
     planner's flips cannot reach the robot; the step watcher checks the committed step.
  3. A checked step event about the committed command (P(done) confirmed): release (`SWITCH_TO` without a subtask),
     which the owner records as a checked DONE step in the task memory; S1 plans the next subtask.
  4. S1 plans a subtask that is already verified done (a checked DONE record; a released-by-bound step is not): not
     committed again (S1 keeps planning); counted in `reentries`, the evidence for training its subtask head with
     memory (v2).
  5. Liveness: a commitment that has not been verified within `max_commit_s` is released (declared bound, not a
     completion signal; counted in `expired`), so a step the verifier cannot see never holds S1 forever.
"""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rojev.protocol import ObservationSnapshot, Operation, OpKind

from .decider import Verdict

COMPLETE_STEP = "the task is complete"


@dataclass(frozen=True)
class NextStep:
    """A next-step model's answer: `step` is one of the allowed steps, or None when the answer was not one."""

    step: str | None
    reason: str


class NextStepModel(Protocol):
    name: str

    def next_step(self, goal: str, record: list[str], options: list[str],
                  images: dict[str, np.ndarray]) -> NextStep: ...


class StepPlanner:
    name = "step-planner"

    def __init__(self, skills: list[str], model: NextStepModel, cameras: list[str]):
        self.skills, self.model, self.cameras = list(skills), model, list(cameras)
        self.options = self.skills + [COMPLETE_STEP]

    def decide(self, s: ObservationSnapshot) -> Verdict:
        record = s.memory.done()
        images = {c: s.capture.observation.images[c] for c in self.cameras}
        answer = self.model.next_step(s.memory.goal, record, self.options, images)
        evidence = {"model": self.model.name, "record": record, "answer": answer.step}
        op = s.active_operation
        running = op.subtask if op is not None and op.kind == OpKind.SWITCH_TO else None
        if answer.step not in self.options:
            return Verdict(Operation(OpKind.CONTINUE), f"no allowed step in the answer: {answer.reason}",
                           evidence=evidence)
        if answer.step == COMPLETE_STEP:
            return Verdict(Operation(OpKind.COMPLETE), answer.reason, evidence=evidence)
        if answer.step == running:
            return Verdict(Operation(OpKind.CONTINUE), f"still {running}: {answer.reason}", evidence=evidence)
        return Verdict(Operation(OpKind.SWITCH_TO, subtask=answer.step), answer.reason, evidence=evidence)


class CommitVerifyPlanner:
    name = "commit-verify"

    def __init__(self, max_commit_s: float):
        self.max_commit_s = max_commit_s
        self.reentries = 0
        self.expired = 0

    def decide(self, s: ObservationSnapshot) -> Verdict:
        op = s.active_operation
        done = s.memory.checked_done()
        if op is None or not op.moves_robot():
            return Verdict(Operation(OpKind.SWITCH_TO), "S1 plans the next subtask")
        if op.subtask is None:
            planned = s.planned.subtask if s.planned is not None and s.planned.cmd_id == s.active_cmd else None
            if planned is None:
                return Verdict(Operation(OpKind.CONTINUE), "S1 is planning")
            if planned in done:
                self.reentries += 1
                return Verdict(Operation(OpKind.CONTINUE), f"S1 re-planned a verified step: {planned}",
                               evidence={"reentry": planned})
            return Verdict(Operation(OpKind.SWITCH_TO, subtask=planned), f"commit to S1's subtask: {planned}")
        checked = s.step_event is not None and s.step_event.checked and s.step_event.cmd_id == s.active_cmd \
            and s.step_event.step == op.subtask
        if checked:
            return Verdict(Operation(OpKind.SWITCH_TO), f"verified done: {op.subtask}; S1 plans the next subtask")
        if s.active_since is not None and s.now - s.active_since >= self.max_commit_s:
            self.expired += 1
            return Verdict(Operation(OpKind.SWITCH_TO), f"commit bound {self.max_commit_s:g} s spent: {op.subtask}",
                           evidence={"expired": op.subtask})
        return Verdict(Operation(OpKind.CONTINUE), f"committed: {op.subtask}")
