"""Scores one model's answers on S2 bake-off case sets against the truths, and checks them against the pass bars.

  uv run python score.py --set ../../data/s2-cases/so101-steps <answers dir> --set <cases dir> <answers dir> \
      [--bars ../../config/s2_pass_bars.yaml] [--out report.json]

Inputs: each case set's `cases.jsonl` (contract in `cases.py`) and the model's `answers.jsonl` + `timing.json`
(`services/planner-vlm/bench.py`). An answer outside the asked format is never correct; for status it counts as
"not done" (the runtime takes no step on it), so it can cause a missed done but never a false done.

Measures, per case set:
  next_step  accuracy per ledger variant (`record`, `no_record`, `contradiction`) and the format-failure rate
  status     per camera and for "either camera says done": false-done rate (said done / truly not done: in progress
             or failed), missed-done rate (not done / truly done), balanced accuracy (1 - their mean) pooled and per
             goal (macro over goals that have both truths, so a goal whose attempts all failed cannot reward a model
             for a task prior), and the paired-step pass rate (same episode and frame asked about steps with
             opposite truths: both right)
  latency    p50 / p95 of single-question latency per kind; median generated tokens for right vs wrong answers
Intervals: 95% percentile bootstrap over episodes (cases of one episode are not independent), BOOTSTRAP draws, seed 0.
Verdict per bar: pass / fail on the point estimate, `uncertain` when the interval crosses the bar. Bars are config
(`config/s2_pass_bars.yaml`), fixed before the run; done-check bars apply to the "either camera" figures of every set,
balanced accuracy in its per-goal (macro) form.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

BOOTSTRAP = 1000
VARIANTS = ("record", "no_record", "contradiction")


def load(cases_dir: Path, answers_dir: Path) -> tuple[list[dict], dict, dict]:
    cases = [json.loads(line) for line in (cases_dir / "cases.jsonl").read_text().splitlines()]
    answers = defaultdict(dict)
    for line in (answers_dir / "answers.jsonl").read_text().splitlines():
        a = json.loads(line)
        answers[a["id"]][a["camera"]] = a
    timing = json.loads((answers_dir / "timing.json").read_text())
    return cases, answers, timing


def said_done(answer: dict | None) -> bool:
    return bool(answer and answer["format_ok"] and answer["answer"] is True)


def episode_of(case: dict) -> str:
    return f"{case.get('source', '')}/{case['episode']}"


def bootstrap(rows: list[tuple[str, float]], statistic) -> tuple[float, float, float]:
    point = statistic([v for _, v in rows])
    episodes = sorted({e for e, _ in rows})
    by_episode = defaultdict(list)
    for e, v in rows:
        by_episode[e].append(v)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(BOOTSTRAP):
        picked = rng.choice(len(episodes), len(episodes))
        values = [v for i in picked for v in by_episode[episodes[i]]]
        if values:
            draws.append(statistic(values))
    low, high = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan"))
    return float(point), float(low), float(high)


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def figure(rows: list[tuple[str, float]]) -> dict:
    point, low, high = bootstrap(rows, mean)
    return {"value": point, "low": low, "high": high, "n": len(rows)}


def next_step_scores(cases: list[dict], answers: dict) -> dict:
    out = {}
    for variant in VARIANTS:
        rows, malformed = [], 0
        for case in (c for c in cases if c["kind"] == "next_step" and c.get("variant") == variant):
            answer = answers[case["id"]].get("top")
            malformed += not (answer and answer["format_ok"])
            rows.append((episode_of(case), float(bool(answer and answer["format_ok"]
                                                      and answer["answer"] == case["truth"]))))
        if rows:
            out[variant] = {**figure(rows), "format_failures": malformed}
    return out


def done_scores(cases: list[dict], answers: dict, view) -> dict:
    status = [c for c in cases if c["kind"] == "status"]
    if not status:
        return {}
    not_done = [(episode_of(c), float(view(answers[c["id"]]))) for c in status if c["truth"] != "done"]
    done = [(episode_of(c), float(not view(answers[c["id"]]))) for c in status if c["truth"] == "done"]
    both = [(episode_of(c), (c["goal"], c["truth"] == "done", view(answers[c["id"]]))) for c in status]

    def balanced(rows):
        truly = [said for _, is_done, said in rows if is_done]
        not_truly = [said for _, is_done, said in rows if not is_done]
        if not truly or not not_truly:
            return float("nan")
        return 1 - (mean([float(not s) for s in truly]) + mean([float(s) for s in not_truly])) / 2

    def per_goal(rows):
        goals = defaultdict(list)
        for row in rows:
            goals[row[0]].append(row)
        values = [balanced(g) for g in goals.values()]
        return mean([v for v in values if not np.isnan(v)])

    pooled, macro = bootstrap(both, balanced), bootstrap(both, per_goal)
    return {"false_done_rate": figure(not_done) if not_done else None, "missed_done_rate": figure(done),
            "balanced_accuracy": {"value": pooled[0], "low": pooled[1], "high": pooled[2], "n": len(both)},
            "balanced_accuracy_per_goal": {"value": macro[0], "low": macro[1], "high": macro[2], "n": len(both)}}


def paired_steps(cases: list[dict], answers: dict, view) -> dict | None:
    groups = defaultdict(list)
    for c in (c for c in cases if c["kind"] == "status"):
        groups[(episode_of(c), c["frame"])].append(c)
    rows = []
    for (episode, _), group in groups.items():
        truths = {c["truth"] == "done" for c in group}
        if len(group) > 1 and truths == {True, False}:
            rows.append((episode, float(all(view(answers[c["id"]]) == (c["truth"] == "done") for c in group))))
    return figure(rows) if rows else None


def latency(timing: dict, cases: list[dict], answers: dict) -> dict:
    out = {}
    for kind, values in timing["single_question_ms"].items():
        if values:
            out[kind] = {"p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}
    right, wrong = [], []
    for case in cases:
        for camera, answer in answers[case["id"]].items():
            correct = (answer["answer"] == case["truth"] if case["kind"] == "next_step"
                       else said_done(answer) == (case["truth"] == "done"))
            (right if correct else wrong).append(answer["new_tokens"])
    out["median_tokens"] = {"right": float(np.median(right)) if right else None,
                            "wrong": float(np.median(wrong)) if wrong else None}
    return out


def score_set(cases_dir: Path, answers_dir: Path) -> dict:
    cases, answers, timing = load(cases_dir, answers_dir)
    cameras = sorted({cam for c in cases if c["kind"] == "status" for cam in c["images"]})
    views = {camera: (lambda cam: lambda a: said_done(a.get(cam)))(camera) for camera in cameras}
    views["either"] = lambda a: any(said_done(x) for x in a.values())
    return {"next_step": next_step_scores(cases, answers),
            "done_check": {name: {**done_scores(cases, answers, view),
                                  "paired_steps": paired_steps(cases, answers, view)}
                           for name, view in views.items()} if cameras else {},
            "latency": latency(timing, cases, answers)}


def verdict(figure_: dict | None, bar: float, higher_is_better: bool) -> str | None:
    if not figure_ or np.isnan(figure_["value"]):
        return None
    meets = figure_["value"] >= bar if higher_is_better else figure_["value"] <= bar
    crosses = figure_["low"] < bar < figure_["high"]
    return "uncertain" if crosses else ("pass" if meets else "fail")


def check(report: dict, bars: dict) -> dict:
    out = {}
    for name, scored in report.items():
        record = scored["next_step"].get("record")
        out[f"{name}: next step with record"] = verdict(record, bars["next_step"]["accuracy_with_record_min"], True)
        either = scored["done_check"].get("either", {})
        out[f"{name}: false done"] = verdict(either.get("false_done_rate"),
                                             bars["done_check"]["false_done_rate_max"], False)
        out[f"{name}: balanced accuracy"] = verdict(either.get("balanced_accuracy_per_goal"),
                                                    bars["done_check"]["balanced_accuracy_min"], True)
        for kind, lat in scored["latency"].items():
            if kind != "median_tokens":
                out[f"{name}: {kind} p50 latency"] = ("pass" if lat["p50_ms"] <= bars["latency"]["p50_ms_max"]
                                                      else "fail")
    return {k: v for k, v in out.items() if v is not None}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", nargs=2, action="append", type=Path, required=True, metavar=("CASES", "ANSWERS"))
    p.add_argument("--bars", type=Path, default=root / "config/s2_pass_bars.yaml")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    report = {cases_dir.name: score_set(cases_dir, answers_dir) for cases_dir, answers_dir in a.set}
    result = {"sets": report, "verdicts": check(report, yaml.safe_load(a.bars.read_text()))}
    text = json.dumps(result, indent=1)
    if a.out:
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
