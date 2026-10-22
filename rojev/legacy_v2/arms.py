"""v2 arm contract (chunk-synchronous): who decides which subtask the frozen policy runs next, once per chunk.

Called at each chunk boundary with a `DecisionContext`; returns a `Decision`. Arm D (OneJev + MJev) is in
`onejev_arm.py`, baselines B / E in `baselines.py`. Replaced in v3 by `rojev.system2.Decider` (docs/paper/method.md §6).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rojev.protocol import Observation


@dataclass(frozen=True)
class DecisionContext:
    chunk: int  
    step: int  
    current: str | None  
    chunks_on_current: int  
    history: tuple[str, ...]  
    observation: Observation | None = None  

@dataclass(frozen=True)
class Decision:
    subtask: str | None  
    reason: str  
    detail: dict = field(default_factory=dict)  


class SubtaskSource(ABC):
    name: str

    @abstractmethod
    def decide(self, ctx: DecisionContext) -> Decision: ...
