"""System 1 policy contract: a visuomotor policy that turns (observation, subtask) into a chunk of actions.

Hardware-shaped: a local model, a simulator stand-in and a remote GPU service all sit behind the same interface. Callers
only ever see `Observation` in and `ActionChunk` out; the policy's own pre/post-processing stays behind `act`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rojev.protocol import Observation


@dataclass(frozen=True)
class ActionChunk:
    """`actions` (n × action_dim, float32) run in order, one per control step. `timings_ms` per stage (preprocess,
    infer, network, …). `info` is what the policy reports about itself, e.g. the subtask its own planner chose."""

    actions: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)
    info: dict = field(default_factory=dict)


class Policy(ABC):
    @abstractmethod
    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        """Actions for the next chunk, conditioned on `subtask` (the text the policy was trained to follow). An empty
        `subtask` asks a policy with its own planner to choose the subtask from `obs.task`."""

    def act_samples(self, obs: Observation, subtask: str, k: int) -> list[ActionChunk]:
        """`k` independent chunks for the same observation (a stochastic policy samples different ones). Default:
        `k` sequential calls; a service that batches samples overrides it."""
        return [self.act(obs, subtask) for _ in range(k)]

    @abstractmethod
    def reset(self) -> None:
        """Clear policy state between episodes."""
