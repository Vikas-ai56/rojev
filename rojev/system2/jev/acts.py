"""Interaction acts: what the user wants, generated in code from the snapshot, and the operation each one becomes.

A change (instruction, deviation) is answered with one act. An act is the user's intent; its effect on the robot is
decided here in code from the act and the situation, never by the model. One act per distinct effect, offered only
where it is legal (plan p. 4: "generate complete, compatible candidates in code"):

  situation   acts                                            effect
  moving      continue · revise · wait · ask · abort          keep · new task · hold · question · stop
  idle        continue · revise · ask · abort                 keep · new task · question · stop
  held        resume · continue · revise · ask · abort        the interrupted operation · keep holding · new task ·
                                                              question · stop (resume only if a step was interrupted,
                                                              abort only if the hold is not already a stop)
  deviation   continue · wait                                 keep going · hold until the user decides

Each act has an effect on the robot (`effect`): it keeps or starts motion (`move`), holds it (`hold`) or starts a
different task (`switch`). Acts with the same effect differ only in wording or in what happens later (stop and wait
both hold now), so the decider hesitates between effects, never between two acts that do the same thing.
`ask` is offered only while the question budget allows it (`may_ask`). There is no retry: an S1 conditioned only on the
task sentence (XR-1) gets the same sentence again, which is `continue`. REVISE has no operation of its own: a `Reviser`
names the new task (default `task_choice.CanonicalTask`: one of the sentences S1 was trained on), or finds none and the
user is asked. Every other act keeps the task text in force.
Each option is worded as what the user wants (live 2026-10-22: wording an option as the robot's action let "go on"
match a "continue" that meant "keep holding"). REVISE means the user stated a complete new task; a partial change or
an answer fragment is a question (live 2026-10-23: "please finish the task fast" was revised into the task text).
Options that keep the task name it. Robot speech is a template id plus objects; `TEMPLATES` holds the default wording
the executor may render.
"""
from dataclasses import dataclass, field
from typing import Protocol

from rojev.protocol import ROBOT, UNANSWERED, ObservationSnapshot, Operation, OpKind

CONTINUE, REVISE, WAIT, ASK, ABORT, RESUME = "continue", "revise", "wait", "ask", "abort", "resume"
CLARIFY, WAITING = "clarify", "waiting_for_user"
TEMPLATES = {CLARIFY: "Sorry, I am not sure what you want me to do. Could you say it differently?",
             WAITING: "I am holding still and waiting for you.",
             UNANSWERED: "I could not work out what to do yet, so I am holding still."}
DEVIATION, HELD, MOVING, IDLE = "deviation", "held", "moving", "idle"
MOVE, HOLD, SWITCH = "move", "hold", "switch"

NEW_TASK = "the user stated a complete new task (what to do and with which objects): switch to that task"
UNCLEAR = "it is unclear what the user wants, or they named a change without the full new task: ask them"


@dataclass(frozen=True)
class Option:
    label: str
    text: str


@dataclass(frozen=True)
class Revision:
    """A `Reviser`'s answer to REVISE: the operation that starts the new task and the task text S1 is conditioned on,
    or `operation` None when no task it can name fits (the user is asked). `evidence` is what it saw and scored."""

    operation: Operation | None
    task: str | None
    reason: str
    evidence: dict = field(default_factory=dict)


class Reviser(Protocol):
    """Names the new task a REVISE act asks for."""

    def revise(self, snapshot: ObservationSnapshot) -> Revision: ...


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


def situation_of(snapshot: ObservationSnapshot) -> str:
    """Which row of the act table applies."""
    if snapshot.trigger == "deviation":
        return DEVIATION
    if snapshot.hold is not None:
        return HELD
    return MOVING if _running(snapshot) is not None else IDLE


def effect(label: str, situation: str) -> str:
    """What an act does to the robot now: continue keeps a motion going (`move`) but keeps a hold standing (`hold`);
    resume moves; revise starts another task (`switch`); wait, ask and abort hold."""
    if label == REVISE:
        return SWITCH
    if label == RESUME or (label == CONTINUE and situation in (MOVING, DEVIATION)):
        return MOVE
    return HOLD


def candidates(snapshot: ObservationSnapshot, max_asks: int) -> list[Option]:
    situation, hold, task = situation_of(snapshot), snapshot.hold, snapshot.instruction
    if situation == DEVIATION:
        running = _describe(_running(snapshot))
        return [Option(CONTINUE, f"the arm's motion is fine: keep going with {running}"),
                Option(WAIT, "the arm's motion went wrong: hold still until the user decides")]
    out = []
    if situation == HELD:
        if hold.interrupted is not None:
            out.append(Option(RESUME, "the user wants the robot to go on with the task it was doing before it held "
                                      f"still: '{task}'"))
        out.append(Option(CONTINUE, "the user has not asked the robot to move again: keep holding still"))
    else:
        out.append(Option(CONTINUE, f"what the user said keeps the task '{task}' (a comment, or a request about how "
                                    "fast or how carefully): keep doing it"))
    out.append(Option(REVISE, NEW_TASK))
    if situation == MOVING:
        out.append(Option(WAIT, "the user wants a pause: hold still until told to go on"))
    if may_ask(snapshot, max_asks):
        out.append(Option(ASK, UNCLEAR))
    if not (hold is not None and hold.operation.kind == OpKind.ABORT):
        out.append(Option(ABORT, "the user wants the robot to stop the task"))
    return out


def operation(label: str, snapshot: ObservationSnapshot) -> Operation:
    """The operation of every act but REVISE (which the `Reviser` decides)."""
    if label == CONTINUE:
        return Operation(OpKind.CONTINUE)
    if label == RESUME:
        return snapshot.hold.interrupted
    if label == WAIT:
        return Operation(OpKind.GUARDED_WAIT)
    if label == ASK:
        return Operation(OpKind.ASK, template=CLARIFY)
    if label == ABORT:
        return Operation(OpKind.ABORT)
    raise ValueError(f"unknown act {label!r}")
