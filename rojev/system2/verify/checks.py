"""End-state checks: is a step's goal achieved in the current images? One question per step, answered as a
probability over `STATES`, for any robot and any step text (no per-skill parser). `MJevVerifier` asks MJev-Spatial
the question measured in the arm-D precision check (services/mjev/check_precision.py); `OneJevVerifier` asks OneJev
the same question with the same options, so the two are interchangeable and comparable on identical inputs.

With `postconditions=True` either one asks the step's postcondition instead (`postconditions.fact`: "Is the steak
inside the microwave?", yes / no) and reports P(done) = P(yes), P(possible) = P(no); a step without a postcondition
template has P(done) 0 (it cannot be confirmed by a check). The abstract step question was at chance for both models
on simulator-labelled frames (2026-10-22)."""
from typing import Protocol

import numpy as np
from qev import Choice, Client

from ..models import RemoteMJev, onejev
from .postconditions import fact

YES_NO = {"yes": "yes", "no": "no"}
UNCONFIRMABLE = {"done": 0.0, "possible": 1.0, "blocked": 0.0}
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


def as_states(yes: float) -> dict[str, float]:
    return {"done": yes, "possible": 1.0 - yes, "blocked": 0.0}


class MJevVerifier:
    """All steps in one MJev request; every question sees every camera image."""

    def __init__(self, mjev: RemoteMJev, postconditions: bool = False):
        self.mjev, self.postconditions = mjev, postconditions
        self.name = "mjev-post" if postconditions else "mjev"
        self.last_timings_ms: dict = {}

    def check(self, images: list[np.ndarray], steps: list[str]) -> dict[str, dict[str, float]]:
        options = YES_NO if self.postconditions else STATES
        space = {"kind": "choice", "options": [{"id": k, "text": v} for k, v in options.items()]}
        asked = {s: (fact(s) if self.postconditions else question(s)) for s in steps}
        ids = {s: f"s{i}" for i, s in enumerate(steps) if asked[s] is not None}
        out = {s: dict(UNCONFIRMABLE) for s in steps if asked[s] is None}
        if ids:
            reply = self.mjev.ask(images, [{"id": ids[s], "question": asked[s], "answer_space": space} for s in ids])
            self.last_timings_ms = reply["timings_ms"]
            for s, key in ids.items():
                probabilities = dict(reply["answers"][key]["probabilities"])
                out[s] = as_states(probabilities["yes"]) if self.postconditions else probabilities
        return out


class OneJevVerifier:
    """All steps as questions of one OneJev request (images and context read once, one Choice per step over `STATES`,
    neutral option ids), every camera as its own image; the same question text as `MJevVerifier`."""

    def __init__(self, client: Client, session_id: str | None = None, postconditions: bool = False):
        self.client, self.session_id, self.postconditions = client, session_id, postconditions
        self.name = "onejev-post" if postconditions else "onejev"
        self.last_timings_ms: dict = {}

    def check(self, images: list[np.ndarray], steps: list[str]) -> dict[str, dict[str, float]]:
        options = YES_NO if self.postconditions else STATES
        ids = {f"o{i}": label for i, label in enumerate(options, start=1)}
        asked = {step: (fact(step) if self.postconditions else question(step)) for step in steps}
        keys = {step: f"q{i}" for i, step in enumerate(steps) if asked[step] is not None}
        out = {step: dict(UNCONFIRMABLE) for step in steps if asked[step] is None}
        if not keys:
            return out
        r = onejev.system_one(self.client, list(images), {"images": onejev.camera_views(None, len(images))},
                              {keys[step]: Choice(asked[step], {key: options[label] for key, label in ids.items()})
                               for step in keys},
                              span_name="onejev.verify", session_id=self.session_id)
        for step, key in keys.items():
            probabilities = {ids[option]: p for option, p in r.answers[key].probabilities.items()}
            out[step] = as_states(probabilities["yes"]) if self.postconditions else probabilities
        return out
