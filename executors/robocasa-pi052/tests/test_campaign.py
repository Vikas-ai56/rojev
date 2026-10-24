"""Campaign harness without the simulator: simulated users and the line judge on hand-made owner records."""
import random

from harness.invariants import Run
from harness.judge import allowed, judge_lines
from harness.users import NEW_TASK, PAUSE, RESUME, STOP, UNLISTED, FuzzUser, noisy, say

FAUCET = "Turn on the sink faucet."


def test_fuzz_sessions_are_replayable_and_differ_by_seed():
    a, b = FuzzUser(7, [FAUCET]).session(4, 40), FuzzUser(7, [FAUCET]).session(4, 40)
    assert a == b and a != FuzzUser(8, [FAUCET]).session(4, 40)
    assert all(4 <= line.t <= 41 for line in a) and all(line.task in (None, FAUCET) for line in a)
    assert all(line.task == FAUCET for line in a if line.intent == NEW_TASK)


def test_noisy_text_stays_close_to_the_phrase():
    rng = random.Random(0)
    texts = [noisy("turn on the sink faucet", rng) for _ in range(50)]
    assert any(t != "turn on the sink faucet" for t in texts)
    assert all(len(t) >= len("turn on the sink faucet") - 4 for t in texts)
    assert say(NEW_TASK, random.Random(1), FAUCET, rough=False) in ("turn on the sink faucet",
                                                                    "actually, just turn on the tap",
                                                                    "forget that, switch on the faucet",
                                                                    "can you open the water in the sink instead")


def test_what_is_allowed_depends_on_the_state_heard_in():
    assert allowed(RESUME, "held") == {"resume"} and allowed(RESUME, "moving") == {"continue"}
    assert allowed(PAUSE, "moving") == {"wait"} and "continue" in allowed(PAUSE, "held")
    assert "resume" not in allowed(RESUME, "done") and "revise" not in allowed(UNLISTED, "moving")


def ev(t, kind, **kw):
    return {"t": t, "kind": kind, **kw}


def run(events, success_t=None):
    return Run(None, [], events, {"sim_s": 30.0, "success_t": success_t})


def test_state_is_read_before_the_answer_and_answers_are_classified():
    events = [ev(0.0, "command", op="switch_to", request=1, lane="planning", trigger="heartbeat", decider="native"),
              ev(8.0, "instruction", text="wait"), ev(8.0, "hold", op="guarded_wait"),
              ev(8.0, "command", op="guarded_wait", request=2, lane="interaction", trigger="instruction", decider="j"),
              ev(12.0, "instruction", text="turn on the tap"), ev(12.0, "task", text=FAUCET, request=3),
              ev(12.0, "hold released", op="switch_to"),
              ev(12.0, "command", op="switch_to", request=3, lane="interaction", trigger="instruction", decider="j"),
              ev(15.0, "instruction", text="stop"),
              ev(15.0, "continue", request=4, lane="interaction", trigger="instruction", decider="j")]
    lines = [{"t": 8, "text": "wait", "intent": PAUSE}, {"t": 12, "text": "turn on the tap", "intent": NEW_TASK,
                                                         "task": FAUCET},
             {"t": 15, "text": "stop", "intent": STOP}]
    v = judge_lines(run(events), lines)
    assert [(x.state, x.answer, x.verdict) for x in v] == [("moving", "wait", "pass"), ("held", "revise", "pass"),
                                                           ("moving", "continue", "unsafe")]
    assert v[1].chosen_task == FAUCET


def test_a_wrong_sentence_is_unsafe_and_a_resume_after_success_is_the_completion_gap():
    events = [ev(0.0, "command", op="switch_to", request=1, lane="planning", trigger="heartbeat", decider="n"),
              ev(5.0, "instruction", text="x"), ev(5.0, "task", text="Open the microwave door.", request=2),
              ev(5.0, "reaffirm", op="switch_to", request=2, lane="interaction", trigger="instruction", decider="j"),
              ev(20.0, "instruction", text="go on"),
              ev(20.0, "command", op="switch_to", request=3, lane="interaction", trigger="instruction", decider="j")]
    v = judge_lines(run(events, success_t=18.0), [{"t": 5, "text": "x", "intent": NEW_TASK, "task": FAUCET},
                                                  {"t": 20, "text": "go on", "intent": RESUME}])
    assert [(x.state, x.verdict) for x in v] == [("moving", "unsafe"), ("done", "unsafe")]
