"""rojev.supervisor.candidates on a toy corpus."""
from rojev.supervisor import CandidateBuilder, normalize

EPISODES = [
    ("Put the milk in the fridge.", ["open the fridge door", "pick up the milk", "place the milk in the fridge",
                                      "close the fridge door"]),
    ("Put the cheese in the refrigerator.", ["open the refrigerator door", "pick up the cheese",
                                              "place the cheese in the refrigerator", "close the refrigerator door"]),
    ("Wash the bowl.", ["pick up the bowl", "place the bowl in the sink", "turn on the faucet"]),
]


def test_normalize_merges_articles_and_synonyms():
    assert normalize("Open the Fridge door!") == normalize("open the refrigerator door") == "open refrigerator door"


def test_canonical_surface_is_most_frequent_phrasing():
    cb = CandidateBuilder.fit(EPISODES + [("x", ["open the refrigerator door"])])
    assert cb.surface["open refrigerator door"] == "open the refrigerator door"


def test_unmentioned_prerequisite_is_offered_first():
    """'open the refrigerator door' is never named in the task, but similar training tasks start with it."""
    cb = CandidateBuilder.fit(EPISODES)
    top = cb.shortlist("Put the box in the fridge.", history=[], k=3)
    assert normalize(top[0]) == "open refrigerator door"


def test_transition_after_last_completed_subtask():
    cb = CandidateBuilder.fit(EPISODES)
    top = cb.shortlist("Wash the bowl.", history=["pick up the bowl"], k=1)
    assert top == ["place the bowl in the sink"]


def test_shortlist_is_deterministic_and_bounded():
    cb = CandidateBuilder.fit(EPISODES)
    a = cb.shortlist("Put the box in the fridge.", history=["open the fridge door"], k=5)
    assert a == cb.shortlist("Put the box in the fridge.", history=["open the fridge door"], k=5) and len(a) <= 5

