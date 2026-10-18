"""Arm D option building, evidence and decisions with stand-in OneJev / MJev (no network)."""
import numpy as np
from qev.client import Response, _Answer

from rojev.executor import Observation
from rojev.supervisor import CandidateBuilder, DecisionContext, OneJevSupervisor

EPISODES = [("Put the milk in the fridge.", ["open the fridge door", "pick up the milk", "place the milk in the fridge",
                                             "close the fridge door"])] * 5
CAMS = ("left", "wrist")


class FakeOneJev:
    """Picks the option whose text contains `want`; records the request."""

    def __init__(self, want: str):
        self.want, self.last = want, None

    def system_one(self, state, questions, media, **params):
        self.last = {"state": state, "questions": questions}
        crit = questions["next"].criteria
        pick = next(k for k, v in crit.items() if self.want in v)
        probs = {k: (0.9 if k == pick else 0.1 / (len(crit) - 1)) for k in crit}
        return Response("fake", {"next": _Answer({"type": "choice", "choice": pick, "probabilities": probs})}, {})


class FakeMJev:
    def __init__(self):
        self.asked = []

    def ask(self, images, questions):
        self.asked = [q["question"] for q in questions]
        return {"answers": {q["id"]: {"probabilities": {"done": 0.1, "possible": 0.8, "blocked": 0.1}}
                            for q in questions}, "timings_ms": {"mjev": 1.0}}


def ctx(current, history=()):
    obs = Observation({c: np.zeros((4, 4, 3), np.uint8) for c in CAMS}, np.zeros(16, np.float32),
                      "Put the box in the fridge.")
    return DecisionContext(chunk=1, step=50, current=current, chunks_on_current=1, history=tuple(history),
                           observation=obs)


def arm(want, mjev=None):
    return OneJevSupervisor(FakeOneJev(want), CandidateBuilder.fit(EPISODES), mjev, CAMS, k=5, m=2)


def test_same_template_for_every_judged_subtask_and_evidence_reaches_onejev():
    mjev = FakeMJev()
    a = arm("continue", mjev)
    d = a.decide(ctx("pick up the milk", ["open the fridge door"]))
    assert d.subtask == "pick up the milk" and d.reason.startswith("continue")
    assert all(q.startswith("Robot step: '") and q.endswith("Which is true right now?") for q in mjev.asked)
    assert "Robot step: 'pick up the milk'. Which is true right now?" in mjev.asked  # current is always judged
    assert "Robot step: 'open the fridge door'. Which is true right now?" in mjev.asked  # recent: catches undone work
    assert "evidence_per_subtask" in a.client.last["state"] and d.detail["saycan_top3"]


def test_switch_redo_and_complete_are_ordinary_options():
    assert arm("place the milk").decide(ctx("pick up the milk")).subtask == "place the milk in the fridge"
    redo = arm("open the fridge door").decide(ctx("pick up the milk", ["open the fridge door"]))
    assert redo.subtask == "open the fridge door"  # a previous subtask can be chosen again
    assert arm("whole task is complete").decide(ctx("close the fridge door")).subtask is None


def test_no_continue_option_before_the_first_subtask():
    a = arm("open the fridge door")
    a.decide(ctx(None))
    assert not any(v.startswith("continue") for v in a.client.last["questions"]["next"].criteria.values())
