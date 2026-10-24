"""Planning-lane deciders deployable with a frozen S1 that plans its own subtasks (π0.5²).

`CommitVerifyPlanner`: commit and verify. S1's own subtask head is memoryless (trained on task + image only), so it
flips between subtasks when the image is ambiguous (live: freezer ↔ fridge 18 times in 60 s). S1 is also steerable by
subtask text. So:
  1. With no command, or after a release: one command without a subtask; S1 plans (its choice arrives as
     `snapshot.planned`).
  2. S1 planned a subtask not yet verified done: commit, i.e. `SWITCH_TO` that text. S1 now follows it and its
     planner's flips cannot reach the robot; the step watcher checks the committed step.
  3. A checked step event about the committed command (P(done) confirmed): release (`SWITCH_TO` without a subtask),
     which the owner records as the verified keyframe; S1 plans the next subtask.
  4. S1 plans a subtask that is already verified done (a checked keyframe; a released-by-bound step is not): not
     committed again (S1 keeps planning); counted in `reentries`, the evidence for training its subtask head with
     memory (v2).
  5. Liveness: a commitment that has not been verified within `max_commit_s` is released (declared bound, not a
     completion signal; counted in `expired`), so a step the verifier cannot see never holds S1 forever.
"""
from rojev.protocol import ObservationSnapshot, Operation, OpKind

from .decider import Verdict


class CommitVerifyPlanner:
    name = "commit-verify"

    def __init__(self, max_commit_s: float):
        self.max_commit_s = max_commit_s
        self.reentries = 0
        self.expired = 0

    def decide(self, s: ObservationSnapshot) -> Verdict:
        op = s.active_operation
        done = {k.subtask for k in s.keyframes if k.checked}
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
