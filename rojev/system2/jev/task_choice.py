"""The second interaction question: which task the user asked for, as one of the sentences S1 was trained on.

Asked only after the act question chose REVISE. The options are `sentences` (given by the robot side: the task
sentences its S1 was trained on that can run in this scene, verbatim) plus `none`. The user's words reach the model
only as evidence in the situation; the chosen sentence, never the user's words, becomes the task text S1 is
conditioned on, so a remark, a typo or a speed request cannot become S1's prompt (live 2026-10-23: "make it fast pls"
was revised into the task; "close the cabinet door" could not be chosen because it was never offered).

`none` wins, or the calibrated margin between the two best options is below `calibration.margin` → no revision
(`Revision.operation` None): the interaction decider asks the user instead. A task the list does not hold cannot be
started: the list is the whole repertoire until S1 can be told tasks it was not trained on.
"""
from collections.abc import Sequence

from rojev.protocol import ObservationSnapshot, Operation, OpKind

from .acts import Option, Revision
from .calibration import Calibration
from .chooser import Chooser

NONE = "none"
TASK_QUESTION = ("The user asked the robot for a new task. Judging by the images now, what the user said and the "
                 "conversation, which of these tasks did the user ask for?")
NONE_TEXT = "none of these: the user asked for something else, or did not say which task"


class CanonicalTask:
    """`Reviser`: REVISE = the trained task sentence the user asked for, chosen by `chooser` (see module docstring)."""

    name = "canonical"

    def __init__(self, chooser: Chooser, sentences: Sequence[str], calibration: Calibration = Calibration()):
        if not sentences:
            raise ValueError("CanonicalTask needs at least one task sentence")
        if NONE in sentences or len(set(sentences)) != len(sentences):
            raise ValueError(f"task sentences must be distinct and not {NONE!r}")
        self.chooser, self.sentences, self.calibration = chooser, list(sentences), calibration

    def options(self) -> list[Option]:
        return [Option(s, s) for s in self.sentences] + [Option(NONE, NONE_TEXT)]

    def revise(self, snapshot: ObservationSnapshot) -> Revision:
        options = self.options()
        raw = self.chooser.choose(snapshot, options, TASK_QUESTION)
        probs = self.calibration.apply({o.label: raw.get(o.label, 0.0) for o in options})
        ranked = sorted(probs, key=probs.get, reverse=True)
        best = ranked[0]
        margin = probs[best] - (probs[ranked[1]] if len(ranked) > 1 else 0.0)
        evidence = {"options": {o.label: o.text for o in options}, "raw": raw, "scores": probs,
                    "margin": round(margin, 4), "question": TASK_QUESTION,
                    "calibration": {"temperature": self.calibration.temperature, "fitted": self.calibration.fitted}}
        if best == NONE:
            return Revision(None, None, f"no trained task fits (p={probs[best]:.2f})", evidence)
        if margin < self.calibration.margin:
            return Revision(None, None, f"task abstain: margin {margin:.2f} < {self.calibration.margin:g} between "
                                        f"{best!r} and {ranked[1]!r}", evidence)
        return Revision(Operation(OpKind.SWITCH_TO, subtask=None), best,
                        f"task {best!r} (p={probs[best]:.2f}, margin {margin:.2f})", evidence)
