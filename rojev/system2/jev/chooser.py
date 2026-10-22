"""Choosers: who scores the legal interaction acts for one snapshot. `OneJevChooser` asks OneJev one typed Choice."""
from typing import Protocol

import numpy as np
from qev import Choice, Client

from rojev.protocol import ROBOT, ObservationSnapshot

from ..models import onejev
from .acts import TEMPLATES, Option

QUESTION = ("The user just spoke, or the robot's motion went wrong. Judging by the images now, the instruction and "
            "the conversation, what should the robot do in response?")


class Chooser(Protocol):
    name: str

    def choose(self, snapshot: ObservationSnapshot, options: list[Option]) -> dict[str, float]:
        """Probability per option label (summing to 1)."""
        ...


def situation(snapshot: ObservationSnapshot, cameras: list[str]) -> dict:
    """What the model is told besides the images: the instruction, the step, progress, the dialogue and the trigger."""
    op = snapshot.active_operation
    state = {"image": "<image:1> = robot cameras side by side: " + ", ".join(cameras),
             "instruction": snapshot.instruction, "trigger": snapshot.trigger,
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
    """One OneJev Choice over the options (label → text); `debias` averages over cyclic shifts of the option order."""

    name = "onejev"

    def __init__(self, client: Client, cameras: list[str], debias: int = 0, session_id: str | None = None):
        self.client, self.cameras, self.debias, self.session_id = client, cameras, debias, session_id

    def choose(self, snapshot: ObservationSnapshot, options: list[Option]) -> dict[str, float]:
        images = snapshot.capture.observation.images
        image = np.concatenate([images[c] for c in self.cameras], axis=1)
        params = {"debias": self.debias} if self.debias else {}
        r = onejev.system_one(self.client, image, situation(snapshot, self.cameras),
                              {"act": Choice(QUESTION, {o.label: o.text for o in options})},
                              span_name="onejev.interaction", session_id=self.session_id, **params)
        return dict(r.answers["act"].probabilities)
