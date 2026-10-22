"""Interaction acts: the legal answers to a change, generated in code from the snapshot, and their operations.

A change (instruction, deviation) is answered with one act. Only acts that make sense in the situation are offered
(plan p. 4: "generate complete, compatible candidates in code"):
  instruction, robot moving     continue · revise · wait · ask · abort
  instruction, robot idle       continue · revise · ask · abort
  instruction, robot held       resume (if a step was interrupted) · continue (keep holding) · revise · ask · abort
                                (abort only if the hold is not already a stop)
  deviation                     continue · retry (if a step is running) · revise · ask
`ask` is offered only while the question budget allows it (`may_ask`). REVISE has no operation of its own: a `Reviser`
turns it into an operation and the new task text S1 is conditioned on (default `S1Replans`: S1's own planner re-plans
under what the user just said, spec §9 / decision D1). Every other act keeps the task text in force.
Each option is worded as what the user wants (live 2026-10-22: wording an option as the robot's action let "go on"
match a "continue" that meant "keep holding"). Robot speech is a template id plus objects; `TEMPLATES` holds the
default wording the executor may render.
"""
from dataclasses import dataclass
from typing import Protocol

from rojev.protocol import ROBOT, UNANSWERED, ObservationSnapshot, Operation, OpKind

CONTINUE, REVISE, WAIT, ASK, ABORT, RETRY, RESUME = "continue", "revise", "wait", "ask", "abort", "retry", "resume"
CLARIFY, WAITING = "clarify", "waiting_for_user"
TEMPLATES = {CLARIFY: "Sorry, I am not sure what you want me to do. Could you say it differently?",
             WAITING: "I am holding still and waiting for you.",
             UNANSWERED: "I could not work out what to do yet, so I am holding still."}


@dataclass(frozen=True)
class Option:
    label: str
    text: str


class Reviser(Protocol):
    """Turns a REVISE act into the operation that follows the user's change and the task text to condition S1 on."""

    def revise(self, snapshot: ObservationSnapshot) -> tuple[Operation, str]: ...


class S1Replans:
    """REVISE = a moving command without a subtask whose task text is what the user just said: S1's own planner
    re-plans from it (π0.5²)."""

    def revise(self, snapshot: ObservationSnapshot) -> tuple[Operation, str]:
        return Operation(OpKind.SWITCH_TO, subtask=None), snapshot.latest_utterance


def may_ask(snapshot: ObservationSnapshot, max_asks: int) -> bool:
    """A question is allowed while fewer than `max_asks` are in the dialogue and the last turn is not already one
    (never ask again without new evidence from the user)."""
    asked = [t for t in snapshot.dialogue if t.speaker == ROBOT and t.template == CLARIFY]
    last = snapshot.dialogue[-1] if snapshot.dialogue else None
    return len(asked) < max_asks and not (last is not None and last.speaker == ROBOT and last.template == CLARIFY)


def _running(snapshot: ObservationSnapshot) -> Operation | None:
    op = snapshot.active_operation
    return op if op is not None and op.moves_robot() else None


def _describe(op: Operation | None) -> str:
    if op is None:
        return "nothing"
    return op.subtask if op.subtask else ("the task, in steps the robot plans itself" if op.moves_robot()
                                          else op.kind.value)


def candidates(snapshot: ObservationSnapshot, max_asks: int) -> list[Option]:
    running, hold = _running(snapshot), snapshot.hold
    out = []
    if snapshot.trigger == "deviation":
        out.append(Option(CONTINUE, f"the motion is fine: keep going with {_describe(running)}"))
        if running is not None:
            out.append(Option(RETRY, f"the motion went wrong: start {_describe(running)} again"))
        out.append(Option(REVISE, "this way cannot work: plan the step differently"))
    elif hold is not None:
        if hold.interrupted is not None:
            out.append(Option(RESUME, "the user wants the robot to go on with what it was doing before it held still: "
                                      f"{_describe(hold.interrupted)}"))
        out.append(Option(CONTINUE, "the user has not asked the robot to move again: keep holding still"))
        out.append(Option(REVISE, "the user gave a new goal or a new detail, such as the answer to the robot's "
                                  "question: plan again to follow it"))
    else:
        out.append(Option(CONTINUE, f"what the user said does not change the task: keep doing {_describe(running)}"))
        out.append(Option(REVISE, "the user changed the goal or how to reach it: plan again to follow what they said"))
        if running is not None:
            out.append(Option(WAIT, "the user wants a pause: hold still until told to go on"))
    if may_ask(snapshot, max_asks):
        out.append(Option(ASK, "it is unclear what the user wants: ask them"))
    if snapshot.trigger != "deviation" and not (hold is not None and hold.operation.kind == OpKind.ABORT):
        out.append(Option(ABORT, "the user wants the robot to stop the task"))
    return out


def operation(label: str, snapshot: ObservationSnapshot) -> Operation:
    """The operation of every act but REVISE (which the `Reviser` decides)."""
    running = _running(snapshot)
    if label == CONTINUE:
        return Operation(OpKind.CONTINUE)
    if label == RESUME:
        return snapshot.hold.interrupted
    if label == RETRY:
        return Operation(OpKind.RETRY, subtask=running.subtask, target=running.target, dest=running.dest)
    if label == WAIT:
        return Operation(OpKind.GUARDED_WAIT)
    if label == ASK:
        return Operation(OpKind.ASK, template=CLARIFY)
    if label == ABORT:
        return Operation(OpKind.ABORT)
    raise ValueError(f"unknown act {label!r}")
