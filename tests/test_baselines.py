"""rojev.eval baselines: arms B (fixed budget) and E (ground-truth oracle)."""
from rojev.eval import FixedSequence, OracleSequence
from rojev.supervisor import DecisionContext

SEQ = ["open the door", "pick up the cup", "close the door"]


def ctx(chunk, current, on_current):
    return DecisionContext(chunk=chunk, step=chunk * 50, current=current, chunks_on_current=on_current, history=())


def test_fixed_sequence_spends_budget_then_advances_and_finishes():
    arm = FixedSequence(SEQ, chunks_per_subtask=2)
    got = [arm.decide(ctx(0, None, 0)).subtask]
    current, on = got[0], 1
    for chunk in range(1, 6):
        d = arm.decide(ctx(chunk, current, on))
        on = on + 1 if d.subtask == current else 1
        current = d.subtask
        got.append(current)
    assert got == ["open the door", "open the door", "pick up the cup", "pick up the cup", "close the door",
                   "close the door"]
    assert arm.decide(ctx(6, current, 2)).subtask is None


def test_oracle_advances_skips_and_goes_back_on_ground_truth():
    done = {0: False, 1: False, 2: False}
    arm = OracleSequence(SEQ, lambda i: done[i])
    assert arm.decide(ctx(0, None, 0)).subtask == "open the door"
    assert arm.decide(ctx(1, "open the door", 1)).subtask == "open the door"
    done[0] = done[1] = True  # door opened and cup picked within one chunk
    d = arm.decide(ctx(2, "open the door", 2))
    assert d.subtask == "close the door" and d.reason.startswith("done")
    done[1] = False  # cup dropped: progress undone
    d = arm.decide(ctx(3, "close the door", 1))
    assert d.subtask == "pick up the cup" and d.reason.startswith("progress undone")
    done[1] = done[2] = True
    assert arm.decide(ctx(4, "pick up the cup", 1)).subtask is None
