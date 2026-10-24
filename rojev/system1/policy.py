"""System 1 policy contract: a visuomotor policy that turns (observation, subtask) into a chunk of actions.

Hardware-shaped: a local model, a simulator stand-in and a remote GPU service all sit behind the same interface. Callers
only ever see `Observation` in and `ActionChunk` out; the policy's own pre/post-processing stays behind `act`.

Inference delay: a chunk is placed at its observation's time, and the actions already queued from that time run while
the policy thinks. A policy that can inpaint its chunk on them (real-time chunking: the new chunk agrees with what
the robot executes meanwhile instead of jumping to a plan made without it) declares `accepts_continuation`; callers
then pass a `Continuation`. Every other policy is never given one.
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


@dataclass(frozen=True)
class Continuation:
    """The actions already queued for the command from the observation's time on (n × action_dim, robot action
    space, one per control step) and `delay_steps`: how many of them run before the new chunk can arrive (the
    measured latency in control steps). The new chunk's first `delay_steps` actions should equal them."""

    actions: np.ndarray
    delay_steps: int


class Policy(ABC):
    accepts_continuation: bool = False

    @abstractmethod
    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None,
            continuation: Continuation | None = None) -> ActionChunk:
        """Actions for the next chunk, conditioned on `subtask` (the text the policy was trained to follow). An empty
        `subtask` asks a policy with its own planner to choose the subtask from `obs.task`. `timeout_s`: the caller
        gives up on the answer after it; a remote policy abandons the request (`TimeoutError`), a local one may finish
        anyway. `continuation` is passed only to a policy that `accepts_continuation`."""

    def act_samples(self, obs: Observation, subtask: str, k: int, timeout_s: float | None = None,
                    continuation: Continuation | None = None) -> list[ActionChunk]:
        """`k` independent chunks for the same observation (a stochastic policy samples different ones). Default:
        `k` sequential calls; a service that batches samples overrides it."""
        if continuation is None:
            return [self.act(obs, subtask, timeout_s) for _ in range(k)]
        return [self.act(obs, subtask, timeout_s, continuation) for _ in range(k)]

    @abstractmethod
    def reset(self) -> None:
        """Clear policy state between episodes."""
