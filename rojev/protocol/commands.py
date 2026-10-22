"""Commands between the state owner, the S1 runner and the execution runtime (docs/plan/v3-v1-spec.md §3).

Authority: a command is active only after the execution runtime has applied its first action and acknowledged it.
Every message carries the session `epoch`, so nothing from before a restart can act. Times are on the robot's clock.
"""
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

SCHEMA = 1


class OpKind(str, Enum):
    """COMPLETE: the decider judges the task done (the robot holds; the owner records it)."""

    CONTINUE = "continue"
    SWITCH_TO = "switch_to"
    RETRY = "retry"
    ABORT = "abort"
    GUARDED_WAIT = "guarded_wait"
    ASK = "ask"
    OBSERVE = "observe"
    COMPLETE = "complete"


@dataclass(frozen=True)
class Operation:
    """What a command asks for.

    `subtask` is the canonical text the S1 policy was trained on (route A); None asks S1 to plan its own subtasks from
    the instruction (the native baseline). `target` and `dest` are object (track) ids; `template` and `objects` are the
    ASK question template and its arguments.
    """

    kind: OpKind
    subtask: str | None = None
    target: str | None = None
    dest: str | None = None
    template: str | None = None
    objects: tuple[str, ...] = ()

    def moves_robot(self) -> bool:
        return self.kind in (OpKind.SWITCH_TO, OpKind.RETRY, OpKind.OBSERVE)


class Transition(str, Enum):
    """How a new command takes over: IMMEDIATE replaces the old command's future actions when the new chunk is due
    (blended); PAUSE_HOLD holds now and starts the new command when its first chunk is due."""

    IMMEDIATE = "immediate"
    PAUSE_HOLD = "pause_hold"


@dataclass(frozen=True)
class AcceptedGoal:
    """A command published by the state owner. `expires_t` is its lease: the owner must renew or replace it before.
    `instruction` is the user instruction at `instruction_rev`; S1 is conditioned on it (never on a stale task text)."""

    epoch: int
    cmd_id: int
    instruction_rev: int
    operation: Operation
    transition: Transition
    expires_t: float
    instruction: str = ""
    schema: int = SCHEMA


@dataclass(frozen=True)
class Revoke:
    epoch: int
    cmd_id: int
    reason: str
    schema: int = SCHEMA


@dataclass(frozen=True)
class TimedChunk:
    """A policy's `ActionChunk` placed on the timeline of one command: action k of `actions` (H × action_dim) is due
    at `t_start + k * dt`. `obs_id` is the observation the policy acted on."""

    epoch: int
    cmd_id: int
    obs_id: int
    t_start: float
    dt: float
    actions: np.ndarray
    timings_ms: dict = field(default_factory=dict)
    schema: int = SCHEMA


class AckStatus(str, Enum):
    """ACTIVATED: first action of the command applied. REJECTED: a command message refused (stale epoch, revoked or
    older command, expired lease, renewal of a command that is gone). CHUNK_REJECTED: one chunk refused (late, wrong
    rate or shape, paused, holding); the command itself stands. EXPIRED: lease ran out. DROPPED: the runtime ended the
    command itself because a safety hold took authority away (owner lost, runtime error); a new command is needed.
    STOPPED: stop requested."""

    ACTIVATED = "activated"
    REVOKED = "revoked"
    REJECTED = "rejected"
    CHUNK_REJECTED = "chunk_rejected"
    EXPIRED = "expired"
    DROPPED = "dropped"
    STOPPED = "stopped"


@dataclass(frozen=True)
class ExecutionAck:
    """Report from the execution runtime. `seq` is strictly increasing per runtime, so the owner ignores any seq it has
    already seen; `applied_index` counts actions applied since the runtime started."""

    epoch: int
    cmd_id: int | None
    seq: int
    status: AckStatus
    t: float
    applied_index: int
    detail: str = ""
    schema: int = SCHEMA
