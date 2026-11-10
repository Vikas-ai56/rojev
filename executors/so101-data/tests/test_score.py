"""The scorer on a hand-made case set whose right numbers are known: false and missed done counted from the truth,
a malformed status answer counted as "not done" (never a false done), the paired-step check, next-step accuracy per
ledger variant, and the verdict rule (pass / fail / uncertain against a bar)."""
import json

from score import check, score_set, verdict


def write(tmp_path, cases, answers):
    cases_dir, answers_dir = tmp_path / "cases", tmp_path / "answers"
    cases_dir.mkdir()
    answers_dir.mkdir()
    (cases_dir / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
    (answers_dir / "answers.jsonl").write_text("\n".join(json.dumps(a) for a in answers))
    (answers_dir / "timing.json").write_text(json.dumps({"single_question_ms": {"status": [900.0, 1100.0]}}))
    return cases_dir, answers_dir


def status(case_id, episode, frame, step, truth, goal="g"):
    return {"id": case_id, "kind": "status", "source": "s", "episode": episode, "frame": frame, "step": step,
            "goal": goal, "truth": truth, "images": {"top": ["x.jpg"]}}


def said(case_id, value, ok=True):
    return {"id": case_id, "camera": "top", "answer": value, "format_ok": ok, "raw": "", "new_tokens": 10}


def test_done_figures_pairs_and_malformed_answers(tmp_path):
    cases = [status("a", 0, 5, "pick", "done"), status("b", 0, 5, "stack", "in_progress"),
             status("c", 1, 5, "pick", "done"), status("d", 1, 5, "stack", "in_progress"),
             status("e", 2, 9, "goal", "failed"), status("f", 2, 3, "goal", "done")]
    answers = [said("a", True), said("b", False), said("c", True), said("d", True), said("e", "maybe", ok=False),
               said("f", "yes?", ok=False)]
    report = score_set(*write(tmp_path, cases, answers))
    top = report["done_check"]["top"]
    assert top["false_done_rate"]["value"] == 1 / 3
    assert top["missed_done_rate"]["value"] == 1 / 3
    assert abs(top["balanced_accuracy"]["value"] - 2 / 3) < 1e-9
    assert top["paired_steps"]["value"] == 0.5 and top["paired_steps"]["n"] == 2
    assert report["latency"]["status"]["p50_ms"] == 1000.0


def test_per_goal_balance_ignores_a_goal_with_one_truth(tmp_path):
    cases = [status("a", 0, 1, "s", "done", "g1"), status("b", 1, 1, "s", "failed", "g1"),
             status("c", 2, 1, "s", "failed", "g2"), status("d", 3, 1, "s", "failed", "g2")]
    answers = [said("a", True), said("b", False), said("c", False), said("d", False)]
    top = score_set(*write(tmp_path, cases, answers))["done_check"]["top"]
    assert top["balanced_accuracy_per_goal"]["value"] == 1.0 and top["balanced_accuracy"]["value"] == 1.0


def test_next_step_accuracy_per_variant_and_verdicts(tmp_path):
    cases = [{"id": f"n{i}", "kind": "next_step", "source": "s", "episode": i, "frame": 0, "variant": v,
              "truth": "pick", "record": None, "images": {"top": ["x.jpg"]}}
             for i, v in enumerate(["record", "record", "no_record", "contradiction"])]
    answers = [said("n0", "pick"), said("n1", "pick"), said("n2", "open"), said("n3", "stack it", ok=False)]
    report = score_set(*write(tmp_path, cases, answers))
    assert report["next_step"]["record"]["value"] == 1.0
    assert report["next_step"]["no_record"]["value"] == 0.0
    assert report["next_step"]["contradiction"]["format_failures"] == 1
    assert verdict({"value": 0.95, "low": 0.92, "high": 0.99}, 0.9, True) == "pass"
    assert verdict({"value": 0.95, "low": 0.80, "high": 0.99}, 0.9, True) == "uncertain"
    assert verdict({"value": 0.20, "low": 0.15, "high": 0.25}, 0.1, False) == "fail"
    bars = {"next_step": {"accuracy_with_record_min": 0.9}, "done_check": {"false_done_rate_max": 0.1,
            "balanced_accuracy_min": 0.75}, "latency": {"p50_ms_max": 3000}}
    assert check({"s": report}, bars)["s: next step with record"] in ("pass", "uncertain")
