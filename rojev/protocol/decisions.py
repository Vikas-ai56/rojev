"""Decisions between the state owner and System 2 (Jev or a baseline): what a decider sees and what it proposes.

Authority: advice is evidence, never a command; a proposal is not a command (the state owner decides). Every message
carries the session `epoch`; times are on the robot's clock.
"""
from dataclasses import dataclass, field

from .commands import SCHEMA, Operation
from .observation import Capture


@dataclass(frozen=True)
class UserInstruction:
    text: str


@dataclass(frozen=True)
class Deviation:
    epoch: int
    cmd_id: int | None
    t: float
    score: float
    threshold: float
    source: str = "kinematic"


@dataclass(frozen=True)
class Keyframe:
    """Progress: step `subtask` (command `cmd_id`, under instruction revision `instruction_rev`) was judged done on
    `capture` (the observation the judgement used). `evidence` names who judged it; `checked` is True when that was a
    confirmed end-state check (a checked `StepEvent`), False when a decider moved on by its own judgement. Kept across
    repairs and revisions, so a decider never redoes a done step."""
    cmd_id: int
    subtask: str
    instruction_rev: int
    capture: Capture
    evidence: str
    checked: bool = False


@dataclass(frozen=True)
class StepEvent:
    """Evidence that a step of the running command is done, posted to the owner by whoever observes it 
    an end-state check (MJev, a sensor) or S1's own planner moving on to another subtask. `cmd_id` is the 
    command the evidence is about; `step` the step judged done; `source` names the observer; `detail`
    is what it saw; `checked` is True when an end-state check found the step done (a claim, such as S1's planner moving
    on, is not). The owner asks the planning lane at once and, when the step is then left (or, for S1's own steps, when
    the event is checked), records `source` and `detail` as the keyframe's evidence."""

    epoch: int
    cmd_id: int | None
    t: float
    source: str
    step: str
    detail: dict = field(default_factory=dict)
    checked: bool = False


@dataclass(frozen=True)
class PlannedSubtask:
    """S1's own planner chose `subtask` for command `cmd_id` (a command without a subtask) at `t`: a proposal from S1,
    which a planning decider may commit the robot to (`SWITCH_TO` that text)."""

    epoch: int
    cmd_id: int
    t: float
    subtask: str


USER, ROBOT = "user", "robot"


@dataclass(frozen=True)
class Turn:
    """One line of the dialogue between the user and the robot. A `USER` turn carries the instruction `text`; a `ROBOT`
    turn carries the `template` it was rendered from and its `objects` (the executor renders the words), so a decider
    can read an instruction as the answer to its own question."""

    t: float
    speaker: str
    instruction_rev: int
    text: str = ""
    template: str | None = None
    objects: tuple[str, ...] = ()


ACK, UNANSWERED = "ack", "unanswered"


@dataclass(frozen=True)
class Utterance:
    """Something the robot says or shows (owner → executor, which renders it as text, speech or a light). `template`
    is `ACK` the moment an instruction is heard (the fast acknowledgement, before any model answers), `UNANSWERED` when
    no decision answered a change (the robot holds), or the template of an accepted operation (an ASK question, a
    report); `cmd_id` is that operation's command."""

    epoch: int
    t: float
    template: str
    objects: tuple[str, ...] = ()
    cmd_id: int | None = None


@dataclass(frozen=True)
class Hold:
    """The robot holds because the interaction lane answered a change with `operation` (ABORT, GUARDED_WAIT or ASK) at
    `since`. It stands until a later change resolves it: the planning lane cannot move the robot meanwhile.
    `interrupted` is the moving operation it stopped, so a decider can resume it."""

    operation: Operation
    since: float
    interrupted: Operation | None


@dataclass(frozen=True)
class DecisionFailure:
    """A decision request that produced no proposal (the decider raised); the owner releases what waited on it."""

    epoch: int
    request_id: int
    error: str


@dataclass(frozen=True)
class ObservationSnapshot:
    epoch: int
    request_id: int
    now: float
    capture: Capture
    instruction_rev: int
    instruction: str
    active_cmd: int | None
    active_operation: Operation | None
    active_since: float | None
    keyframes: tuple[Keyframe, ...]
    trigger: str
    review: bool = False
    deviation: Deviation | None = None
    step_event: StepEvent | None = None
    planned: PlannedSubtask | None = None
    hold: Hold | None = None
    dialogue: tuple[Turn, ...] = ()

    @property
    def latest_utterance(self) -> str:
        return next((t.text for t in reversed(self.dialogue) if t.speaker == USER), "")


@dataclass(frozen=True)
class Precondition:
    kind: str
    object_id: str | None = None
    tol: float = 0.0


@dataclass(frozen=True)
class Advice:
    epoch: int
    advice_id: int
    source: str
    t_source_obs: float
    instruction_rev: int
    scope_cmd: int | None
    content: dict
    expires_t: float
    schema: int = SCHEMA


@dataclass(frozen=True)
class DecisionProposal:
    epoch: int
    request_id: int
    obs_id: int
    t_capture: float
    instruction_rev: int
    base_active_cmd: int | None
    candidates_rev: int
    operation: Operation
    scores: dict[str, float]
    preconditions: tuple[Precondition, ...]
    expires_t: float
    evidence: dict = field(default_factory=dict)
    task: str | None = None
    schema: int = SCHEMA
