"""The scenario matrix's scorecard wiring without the simulator: declared changes against the scenario arguments, the
arm label, and `--score-only` on a synthetic matrix directory."""
import json
from pathlib import Path

import pytest

from rojev.eval import Injection
from scenarios import SCENARIOS, Change, Scenario, main


def flag_values(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


def test_every_scheduled_change_is_declared_at_its_argument_time():
    for s in SCENARIOS:
        says = [float(v.split(":", 1)[0]) for v in flag_values(s.args, "--say")]
        blocks = [float(v) for v in flag_values(s.args, "--block-at")]
        injections = s.injections()
        assert [i.t for i in injections if i.trigger == "instruction"] == says, s.name
        assert [i.t for i in injections if i.trigger == "deviation"] == blocks, s.name
        assert bool(injections) == bool(says or blocks), s.name


def test_goal_texts_appear_in_the_utterance_that_names_them():
    for s in SCENARIOS:
        texts = [v.split(":", 1)[1] for v in flag_values(s.args, "--say")]
        swaps = [(i, text) for i, text in zip(s.injections(), texts) if i.goal is not None]
        assert all(i.goal in text for i, text in swaps), s.name


def test_arm_is_the_interaction_value_else_the_arm():
    arms = {s.name: s.arm() for s in SCENARIOS}
    assert arms["stop"] == "rules" and arms["jev-ask-answer"] == "jev"
    assert arms["one-lane"] == "revisable" and arms["step-events"] == "revisable-truth"


def test_a_declaration_that_disagrees_with_the_arguments_is_an_error():
    with pytest.raises(ValueError, match="2 scheduled changes, 1 declared"):
        Scenario("x", "", ["--say", "1:wait", "--say", "2:go on"], changes=(Change("stop", "wait"),)).injections()
    with pytest.raises(ValueError, match="is a deviation"):
        Scenario("x", "", ["--block-at", "3"], changes=(Change("stop", "abort"),)).injections()
    assert Scenario("x", "", ["--block-at", "3"], changes=(Change("deviation", "revise"),)).injections() == (
        Injection(3.0, "deviation", "revise"),)


def write_run(run_dir: Path, say: list) -> None:
    run_dir.mkdir(parents=True)
    summary = {"sim_s": 12.0, "task_success": False, "control_hz": 20, "say": say, "blocked": None}
    events = [{"t": 0.05, "kind": "command", "cmd": 1, "op": "switch_to", "subtask": "pick up the steak",
               "lane": "planning", "trigger": "heartbeat"},
              {"t": 6.0, "kind": "instruction", "rev": 1, "text": "stop"},
              {"t": 6.0, "kind": "command", "cmd": 2, "op": "abort", "subtask": None, "lane": "interaction",
               "trigger": "instruction", "decider": "rules"}]
    ticks = [{"t": round(0.05 * k, 2), "source": "chunk" if k < 121 else "hold", "cmd": 1 if k < 121 else 2}
             for k in range(1, 240)]
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "owner.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (run_dir / "ticks.jsonl").write_text("".join(json.dumps(t) + "\n" for t in ticks))


def test_score_only_scores_a_recorded_matrix_without_running(tmp_path: Path, capsys):
    write_run(tmp_path / "stop", [[6.0, "stop"]])
    write_run(tmp_path / "stop-stays", [[6.0, "stop"]])
    report = [{"name": "nominal", "ok": True}, {"name": "stop", "ok": True}, {"name": "stop-stays", "ok": True},
              {"name": "blocked", "ok": None, "skipped": "needs --monitor MODEL.json"}]
    (tmp_path / "report.json").write_text(json.dumps(report))

    assert main(["--score-only", str(tmp_path)]) == 0

    scored = {e["name"]: e for e in json.loads((tmp_path / "report.json").read_text())}
    assert "scorecard" not in scored["nominal"] and "scorecard" not in scored["blocked"]
    [stop] = scored["stop"]["scorecard"]["injections"]
    assert (stop["act"], stop["correct"], stop["answer_latency_s"]) == ("abort", True, 0.0)
    assert stop["execution_latency_s"] == pytest.approx(0.05)
    assert scored["stop-stays"]["scorecard"]["skipped"] == "run said [[6.0, 'stop']], scenario schedules [[5.0, 'stop']]"
    [arm] = json.loads((tmp_path / "scorecard.json").read_text())["arms"]
    assert (arm["arm"], arm["runs"], arm["act_accuracy"]) == ("rules", 1, {"value": 1.0, "n": 1})
    assert (tmp_path / "scorecard.md").read_text().startswith("| arm |")
    out = capsys.readouterr().out
    assert "| rules | 1 | 1 | 0 |" in out and "[score skip] stop-stays" in out
