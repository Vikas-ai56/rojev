"""End-state checks: is a step's goal achieved in the current images? One question per step, answered as a
probability over `STATES`, for any robot and any step text (no per-skill parser). `MJevVerifier` asks MJev-Spatial
the question measured in the arm-D precision check (services/mjev/check_precision.py)."""
from typing import Protocol

import numpy as np

from ..models import RemoteMJev

STATES = {"done": "done: its goal is already achieved",
          "possible": "possible: it can be started now; what it needs is present and reachable",
          "blocked": "blocked: it cannot be done now; something it needs is missing, hidden, or not ready"}


class Verifier(Protocol):
    name: str

    def check(self, images: list[np.ndarray], steps: list[str]) -> dict[str, dict[str, float]]:
        """Per step text, a probability for each of `STATES`."""
        ...


def question(step: str) -> str:
    return f"Robot step: '{step}'. Which is true right now?"


class MJevVerifier:
    """All steps in one MJev request; every question sees every camera image."""

    name = "mjev"

    def __init__(self, mjev: RemoteMJev):
        self.mjev = mjev
        self.last_timings_ms: dict = {}

    def check(self, images: list[np.ndarray], steps: list[str]) -> dict[str, dict[str, float]]:
        space = {"kind": "choice", "options": [{"id": k, "text": v} for k, v in STATES.items()]}
        out = self.mjev.ask(images, [{"id": f"s{i}", "question": question(s), "answer_space": space}
                                     for i, s in enumerate(steps)])
        self.last_timings_ms = out["timings_ms"]
        return {s: dict(out["answers"][f"s{i}"]["probabilities"]) for i, s in enumerate(steps)}
