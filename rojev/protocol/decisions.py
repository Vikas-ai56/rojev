"""Decisions between the state owner and System 2 (Jev or a baseline): what a decider sees and what it proposes.

Authority: advice is evidence, never a command; a proposal is not a command (the state owner decides). Every message
carries the session `epoch`; times are on the robot's clock.
"""
from dataclasses import dataclass, field

from .commands import SCHEMA, Operation
from .observation import Capture


@dataclass(frozen=True)
class UserInstruction:
    """A new or revised user instruction; the owner bumps `instruction_rev`."""

    text: str


@dataclass(frozen=True)
class Deviation:
    """The world did not respond to the applied actions as the robot's effect model predicts (execution monitor →
    owner). `score` crossed the calibrated `threshold` at `t` while `cmd_id` was active; `source` names the monitor."""

    epoch: int
    cmd_id: int | None
    t: float
    score: float
    threshold: float
    source: str = "kinematic"


@dataclass(frozen=True)
class Keyframe:
    """Progress: step `subtask` (command `cmd_id`, under instruction revision `instruction_rev`) was judged done on
    `capture` (the observation the judgement used). `evidence` names who judged it (a decider moving on until
    postconditions are checked in step 6). Kept across repairs and revisions, so a decider never redoes a done step."""

    cmd_id: int
    subtask: str
    instruction_rev: int
    capture: Capture
    evidence: str


@dataclass(frozen=True)
class StepEvent:
    """Evidence that the running step may be done, posted to the owner by whoever observes it (spec §4 step-completion
    events): a postcondition watcher (MJev, a sensor), S1's own planner moving on to another subtask, a phase boundary.
    `cmd_id` is the command the evidence is about; `source` names the observer; `detail` is what it saw. The owner asks
    the planning lane at once and, if the step is then left, records `source` and `detail` as the keyframe's evidence."""

    epoch: int
    cmd_id: int | None
    t: float
    source: str
    detail: dict = field(default_factory=dict)


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


ACK = "ack"


@dataclass(frozen=True)
class Utterance:
    """Something the robot says or shows (owner → executor, which renders it as text, speech or a light). `template`
    is `ACK` the moment an instruction is heard (the fast acknowledgement, before any model answers), or the template
    of an accepted operation (an ASK question, a report); `cmd_id` is that operation's command."""

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
    """What a decider sees, frozen when the owner asks: the newest observation (`capture`, a loop `Capture`) plus the
    owner's state at that moment. `active_cmd` / `active_operation` describe the desired command (pending or active);
    `active_since` is when its operation started (kept across a reaffirmation; None before); `keyframes` are the steps
    judged done, in order; `trigger` says why the decision was requested; `review` is True while the desired command is
    under review after an instruction change; `deviation` is the latest execution deviation of the desired command;
    `step_event` the latest step-completion evidence about it; `hold` the interaction hold that stands, if any;
    `dialogue` the latest turns between user and robot, oldest first (bounded by the owner)."""

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
    hold: Hold | None = None
    dialogue: tuple[Turn, ...] = ()


@dataclass(frozen=True)
class Precondition:
    """A fact the decision relied on; `kind` is holding | not_holding | object_visible | object_near."""

    kind: str
    object_id: str | None = None
    tol: float = 0.0


@dataclass(frozen=True)
class Advice:
    """Evidence from a slower worker. `t_source_obs` is the capture time of the observation it was computed from;
    `scope_cmd` is the command it is about (None = task level)."""

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
    """Jev's proposal. `base_active_cmd` is the active command the decision assumed; `scores` maps candidate labels
    to the decision model's probabilities."""

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
    schema: int = SCHEMA
