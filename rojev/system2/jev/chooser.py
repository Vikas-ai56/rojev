"""Choosers: who scores the options of one typed question for one snapshot (the interaction acts, or the task the
user asked for). `OneJevChooser` asks OneJev one typed Choice."""
from typing import Protocol

import numpy as np
from qev import Choice, Client

from rojev.protocol import ROBOT, ObservationSnapshot

from ..models import onejev
from .acts import TEMPLATES, Option

ACT_QUESTION = ("The user just spoke, or the robot's motion went wrong. Judging by the images now, the instruction and "
                "the conversation, what should the robot do in response?")


class Chooser(Protocol):
    name: str

    def choose(self, snapshot: ObservationSnapshot, options: list[Option], question: str = ACT_QUESTION
               ) -> dict[str, float]:
        """Probability per option label (summing to 1) for `question`."""
        ...


def situation(snapshot: ObservationSnapshot, cameras: list[str]) -> dict:
    """What the model is told besides the images: the task in force, what the user just said, the step, progress, the
    dialogue and the trigger."""
    op = snapshot.active_operation
    state = {"images": onejev.camera_views(cameras, len(cameras)),
             "task": snapshot.instruction, "user_just_said": snapshot.latest_utterance, "trigger": snapshot.trigger,
             "current_step": None if op is None else (op.subtask or op.kind.value),
             "steps_done": [k.subtask for k in snapshot.keyframes],
             "conversation": [{"robot": TEMPLATES.get(t.template, t.template)} if t.speaker == ROBOT else
                              {"user": t.text} for t in snapshot.dialogue]}
    if snapshot.hold is not None:
        state["holding"] = {"because": snapshot.hold.operation.kind.value,
                            "interrupted_step": None if snapshot.hold.interrupted is None
                            else snapshot.hold.interrupted.subtask}
    if snapshot.trigger == "deviation" and snapshot.deviation is not None:
        state["motion_check"] = "the arm did not move as commanded"
    return state


class OneJevChooser:
    """One OneJev Choice over the options. The model sees neutral ids (`o1`, `o2`, …) and each option's text, never the
    act names: a name such as "continue" primes the user's words ("go on") whatever the option means in the situation.
    `debias` averages over cyclic shifts of the option order."""

    name = "onejev"

    def __init__(self, client: Client, cameras: list[str], debias: int = 0, session_id: str | None = None):
        self.client, self.cameras, self.debias, self.session_id = client, cameras, debias, session_id

    def choose(self, snapshot: ObservationSnapshot, options: list[Option], question: str = ACT_QUESTION
               ) -> dict[str, float]:
        images = snapshot.capture.observation.images
        return self.score([images[c] for c in self.cameras], situation(snapshot, self.cameras), options, question)

    def score(self, images: list[np.ndarray], state: dict, options: list[Option], question: str = ACT_QUESTION
              ) -> dict[str, float]:
        """One OneJev Choice on every camera as its own image (one request) and the situation; also how recorded cases
        are re-scored for calibration."""
        params = {"debias": self.debias} if self.debias else {}
        ids = {f"o{i}": o for i, o in enumerate(options, start=1)}
        r = onejev.system_one(self.client, list(images), state,
                              {"act": Choice(question, {key: o.text for key, o in ids.items()})},
                              span_name="onejev.interaction", session_id=self.session_id, **params)
        return {ids[key].label: p for key, p in r.answers["act"].probabilities.items()}
