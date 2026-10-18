"""Contract for every subtask source ("arm"): who decides which subtask the frozen executor runs next.

Called once per chunk boundary with a `DecisionContext`; returns a `Decision`. The supervisor (arm D, OneJev + MJev)
implements it from images alone; evaluation baselines (B, E) live in `rojev.eval.baselines` (docs/paper/method.md §6).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rojev.executor.base import Observation


@dataclass(frozen=True)
class DecisionContext:
    chunk: int  # chunk index in the episode (0 = first)
    step: int  # env steps executed so far
    current: str | None  # subtask running in the previous chunk (None before the first)
    chunks_on_current: int  # chunks already spent on `current`
    history: tuple[str, ...]  # subtasks the arm has moved past, in order
    observation: Observation | None = None  # current cameras + state + task (what an image-reading arm decides from)


@dataclass(frozen=True)
class Decision:
    subtask: str | None  # None = the arm considers the task finished (episode ends)
    reason: str  # human-readable why, shown live and logged
    detail: dict = field(default_factory=dict)  # arm-specific evidence (probabilities, budgets, truth, …)


class SubtaskSource(ABC):
    name: str

    @abstractmethod
    def decide(self, ctx: DecisionContext) -> Decision: ...
