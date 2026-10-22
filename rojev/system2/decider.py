"""System 2 decider contract: anything that chooses what to do next (Jev, or a baseline arm) from a snapshot."""
from dataclasses import dataclass, field
from typing import Protocol

from rojev.protocol import ObservationSnapshot, Operation, Precondition


@dataclass(frozen=True)
class Verdict:
    """A decider's answer: the operation, why, and the evidence behind it (scores per option, preconditions it relied
    on, anything else worth logging)."""

    operation: Operation
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    preconditions: tuple[Precondition, ...] = ()
    evidence: dict = field(default_factory=dict)


class Decider(Protocol):
    name: str

    def decide(self, snapshot: ObservationSnapshot) -> Verdict: ...
