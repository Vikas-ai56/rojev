"""Integration / regression scenarios: the executor run from its entrypoint (`realtime.py`), one case per runtime
feature and fault (spec step 5a), each checked against the invariants (harness/invariants.py) and its own expectations.

  uv run python scenarios.py                       all cases, headless, scripted S1 (harness/skills.py)
  uv run python scenarios.py --viewer --only stop  watch one case in the live MuJoCo window
  uv run python scenarios.py --policy remote --service lambda_service.json   π0.5² on the GPU as S1
  uv run python scenarios.py --trace rojev-scenarios                          every run exported to Phoenix
  uv run python scenarios.py --score-only results/scenarios/<stamp>                  re-score a past matrix, no runs
Report: results/scenarios/<stamp>/report.json (+ one run directory per case, each with view.mp4: three cameras
and a panel with the active command, its lane and the latest decision; `--no-video` skips it).
Scorecard (T8, rojev/eval/scorecard.py): every case that injects changes (`--say`, `--block-at`) declares them as
`changes` (kind, the act its design calls for, goal texts of a swap), paired in order with those arguments for their
times; after the matrix each such case is scored under its arm (its `--interaction`, else its `--arm`), written to
scorecard.md + scorecard.json beside report.json and into its report entry (`scorecard`).

Every case runs MicrowaveThawingFridge (seed 1000; fridge and microwave doors open) with the revisable planner (the
authored steps, rewritten when the user names a new destination). Utterance times are seconds into the run.
"""
import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from embodiment import PANDA_OMRON
from harness.invariants import Run, check
from rojev.eval import ArmScore, Case, Injection, RunRecords, aggregate, render_json, render_markdown, score_run
from rojev.system2.jev import ABORT, ASK, CONTINUE, REVISE, WAIT

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = ["--task", "MicrowaveThawingFridge", "--seed", "1000", "--arm", "revisable"]


def commands(run: Run, lane: str | None = None, op: str | None = None) -> list[dict]:
    return [e for e in run.events if e["kind"] in ("command", "reaffirm") and (lane is None or e.get("lane") == lane)
            and (op is None or e["op"] == op)]


def first(run: Run, kind: str) -> dict | None:
    return next((e for e in run.events if e["kind"] == kind), None)


def moving_after(run: Run, t: float, within: float) -> bool:
    window = [x for x in run.ticks if t <= x["t"] < t + within]
    return bool(window) and all(x["source"] == "chunk" for x in window)


def holding_after(run: Run, t: float, within: float) -> bool:
    window = [x for x in run.ticks if t <= x["t"] < t + within]
    return bool(window) and all(x["source"] == "hold" for x in window)


def scheduled(args: list[str]) -> list[tuple[float, str, str | None]]:
    """The changes `args` make realtime.py inject, in argument order: (t, "instruction", text) per `--say T:TEXT`,
    (t, "deviation", None) per `--block-at T`."""
    out = []
    for flag, value in zip(args, args[1:]):
        if flag == "--say":
            at, _, text = value.partition(":")
            out.append((float(at), "instruction", text))
        elif flag == "--block-at":
            out.append((float(value), "deviation", None))
    return out


def option(args: list[str], flag: str) -> str | None:
    """The last value given to `flag` in `args` (argparse's rule), None if absent."""
    values = [v for f, v in zip(args, args[1:]) if f == flag]
    return values[-1] if values else None


@dataclass(frozen=True)
class Change:
    """One change a scenario injects, as its design states it: the scorecard kind, the act it calls for, and for a
    swap the text naming the goal after / before it as subtask text spells it. Its time is its argument's."""

    kind: str
    expected: str
    goal: str | None = None
    old_goal: str | None = None
    phase: str | None = None


@dataclass
class Scenario:
    name: str
    what: str
    args: list[str]
    expect: list[tuple[str, Callable[[Run], bool]]] = field(default_factory=list)
    seconds: float = 18.0
    needs: str | None = None
    changes: tuple[Change, ...] = ()

    def arm(self) -> str:
        """Scorecard arm: the deciding system and the conditions that change how it answers, from the case's own
        arguments: its `--interaction` value (else its `--arm`), plus `+revoke` for the stop-and-go owner mode and
        `+s2-slow` / `+s2-fail` for injected S2 faults, so a fault or a different owner policy never pools with the
        nominal system."""
        conditions = [name for flag, name in (("--interaction-delay", "s2-slow"), ("--interaction-fail", "s2-fail"))
                      if flag in self.args]
        if option(self.args, "--on-change") == "revoke":
            conditions.insert(0, "revoke")
        base = option(self.args, "--interaction") or option(BASE + self.args, "--arm")
        return "+".join([base, *conditions])

    def injections(self) -> tuple[Injection, ...]:
        """`changes` paired in order with the `--say` / `--block-at` arguments that inject them (their times), sorted
        by time; a count mismatch or a change whose trigger is not its argument's is an error."""
        schedule = scheduled(self.args)
        if len(schedule) != len(self.changes):
            raise ValueError(f"{self.name}: {len(schedule)} scheduled changes, {len(self.changes)} declared")
        out = []
        for (t, trigger, _), c in zip(schedule, self.changes):
            injection = Injection(t, c.kind, c.expected, c.goal, c.old_goal, c.phase)
            if injection.trigger != trigger:
                raise ValueError(f"{self.name}: change {c.kind!r} at {t} s is a {trigger}, declared a "
                                 f"{injection.trigger}")
            out.append(injection)
        return tuple(sorted(out, key=lambda i: i.t))

    def case(self) -> Case:
        return Case(self.name, self.arm(), self.injections())


SCENARIOS = [
    Scenario("nominal", "authored steps on a time budget; keyframes as steps are moved past", [],
             [("steps advance in order", lambda r: [c["subtask"] for c in commands(r)][:2] ==
               r.summary["sequence"][:2]),
              ("a keyframe for the first step", lambda r: bool(r.summary["keyframes"])),
              ("no starvation", lambda r: r.summary["starved_ticks"] == 0)], seconds=16),
    Scenario("speculative", "K = 3 noisy samples, consensus commitment", [
        "--speculative-predict", "--samples", "3", "--sample-noise", "0.01"],
             [("consensus rules fired", lambda r: any(c["reason"] != "fixed" for c in r.commits)),
              ("no starvation", lambda r: r.summary["starved_ticks"] == 0)], seconds=12),
    Scenario("speculative-phase", "consensus capped by phase boundaries", [
        "--speculative-predict", "--samples", "3", "--sample-noise", "0.01", "--commit-policy", "phase"],
             [("commitments shorter than the chunk", lambda r: any(c["steps"] < c["horizon"] for c in r.commits)),
              ("no starvation", lambda r: r.summary["starved_ticks"] == 0)], seconds=12),
    Scenario("harmless", "harmless speech: the interaction lane continues (the words never become the task), no pause",
             ["--interaction", "rules", "--say", "5:nice work"],
             [("continued by the interaction lane", lambda r: any(
                 e["kind"] == "continue" and e.get("lane") == "interaction" and e.get("trigger") == "instruction"
                 for e in r.events)),
              ("no new command from the interaction lane", lambda r: not commands(r, "interaction")),
              ("arm never paused", lambda r: moving_after(r, 5.0, 2.0))], seconds=10,
             changes=(Change("harmless", CONTINUE),)),
    Scenario("revise", "new destination: later step rewritten, then an in-step revision", [
        "--interaction", "rules", "--say", "4:actually put the steak back on the plate",
        "--say", "14:no, put it in the microwave"],
             [("place step goes to the plate", lambda r: any(c["subtask"] == "place the steak on the plate"
                                                              for c in commands(r))),
              ("in-step revision to the microwave by the interaction lane",
               lambda r: any(c["subtask"] == "place the steak in the microwave" and c.get("lane") == "interaction"
                             for c in commands(r)))], seconds=20,
             changes=(Change("goal_swap", CONTINUE, goal="plate", old_goal="microwave"),
                      Change("goal_swap", REVISE, goal="microwave", old_goal="plate"))),
    Scenario("stop", "stop: ABORT from the interaction lane, robot holds", [
        "--interaction", "rules", "--say", "6:stop"],
             [("abort issued by the interaction lane", lambda r: bool(commands(r, "interaction", "abort"))),
              ("holding after the abort", lambda r: holding_after(r, commands(r, op="abort")[0]["t"] + 0.5, 3.0))],
             seconds=12, changes=(Change("stop", ABORT),)),
    Scenario("wait-resume", "wait, then go on: guarded wait, then the same step resumes", [
        "--interaction", "rules", "--say", "5:wait", "--say", "9:go on"],
             [("guarded wait", lambda r: bool(commands(r, "interaction", "guarded_wait"))),
              ("resumed the interrupted step", lambda r: any(c["subtask"] == r.summary["sequence"][0] and c["t"] >= 9
                                                             for c in commands(r, "interaction")))], seconds=13,
             changes=(Change("pause", WAIT), Change("resume", REVISE))),
    Scenario("ask", "vague reference: ASK, then the answer resumes", [
        "--interaction", "rules", "--say", "5:put it over there", "--say", "9:on the plate"],
             [("asked", lambda r: bool(commands(r, "interaction", "ask"))),
              ("resumed after the answer", lambda r: any(c["t"] >= 9 and c["op"] == "switch_to"
                                                         for c in commands(r, "interaction")))], seconds=13,
             changes=(Change("ambiguous", ASK), Change("goal_swap", REVISE, goal="plate", old_goal="microwave"))),
    Scenario("revoke-baseline", "stop-and-go baseline: revoke on any change", [
        "--interaction", "rules", "--on-change", "revoke", "--say", "5:nice work"],
             [("revoked", lambda r: first(r, "revoke") is not None),
              ("paused (hold ticks after the change)", lambda r: any(x["source"] == "hold" for x in r.ticks
                                                                     if 5.0 <= x["t"] < 7.0))], seconds=10,
             changes=(Change("harmless", CONTINUE),)),
    Scenario("one-lane", "no interaction lane: the planner answers the change", [
        "--say", "5:wait"],
             [("planning lane answered", lambda r: any(e["kind"] in ("reaffirm", "continue")
                                                       and e.get("lane") == "planning" and e["t"] >= 5
                                                       for e in r.events))], seconds=9,
             changes=(Change("pause", WAIT),)),
    Scenario("s2-slow", "interaction decider slower than its deadline: the robot holds and says so, planning never "
             "answers the change", [
        "--interaction", "rules", "--interaction-delay", "1.6", "--interaction-timeout", "1.0", "--say", "5:wait"],
             [("interaction timeout", lambda r: first(r, "interaction timeout") is not None),
              ("unanswered hold said", lambda r: any(u["template"] == "unanswered" for u in r.summary["said"])),
              ("planning never answered the change", lambda r: not any(
                  e.get("lane") == "planning" and e.get("trigger") == "instruction" for e in r.events)),
              ("holding after the timeout", lambda r: holding_after(r, 6.5, 2.0))], seconds=10,
             changes=(Change("pause", WAIT),)),
    Scenario("s2-fail", "interaction decider fails: the robot holds at once and says so", [
        "--interaction", "rules", "--interaction-fail", "--say", "5:wait"],
             [("failure recorded", lambda r: first(r, "decision failed") is not None),
              ("unanswered hold within 1 s", lambda r: any(e["kind"] == "unanswered" and 5 <= e["t"] <= 6
                                                           for e in r.events)),
              ("planning never answered the change", lambda r: not any(
                  e.get("lane") == "planning" and e.get("trigger") == "instruction" for e in r.events)),
              ("holding after the failure", lambda r: holding_after(r, 6.0, 2.0))], seconds=9,
             changes=(Change("pause", WAIT),)),
    Scenario("s1-slow", "S1 latency 0.9 s: floor and lead time follow the measured latency", [
        "--latency", "0.9"],
             [("floor ≥ 2·latency/dt + 1", lambda r: (r.summary["s1_floor_last"] or 0) >= 37),
              ("no starvation after the first chunk", lambda r: r.summary["starved_ticks"] == 0)], seconds=14),
    Scenario("stop-stays", "stop, then heartbeats: the planner never restarts the task (D4)", [
        "--interaction", "rules", "--say", "5:stop"],
             [("abort issued", lambda r: bool(commands(r, "interaction", "abort"))),
              ("no moving command after the abort", lambda r: not any(
                  c["op"] in ("switch_to", "retry") and c["t"] > commands(r, op="abort")[0]["t"] for c in commands(r))),
              ("the stop is a standing hold", lambda r: first(r, "hold") is not None),
              ("holding to the end", lambda r: holding_after(r, commands(r, op="abort")[0]["t"] + 0.5, 8.0))],
             seconds=15, changes=(Change("stop", ABORT),)),
    Scenario("jev-wait-resume", "Jev (keyword scores): wait, then go on resumes the interrupted step", [
        "--interaction", "jev", "--chooser", "keywords", "--say", "5:wait", "--say", "9:go on"],
             [("guarded wait", lambda r: bool(commands(r, "interaction", "guarded_wait"))),
              ("resumed the interrupted step", lambda r: any(c["subtask"] == r.summary["sequence"][0] and c["t"] >= 9
                                                             for c in commands(r, "interaction"))),
              ("acknowledged both", lambda r: sum(u["template"] == "ack" for u in r.summary["said"]) == 2)],
             seconds=13, changes=(Change("pause", WAIT), Change("resume", REVISE))),
    Scenario("jev-ask-answer", "Jev: vague reference → question in the dialogue; the answer revises the step", [
        "--interaction", "jev", "--chooser", "keywords", "--say", "5:put it over there", "--say", "9:on the plate"],
             [("asked", lambda r: any(u["template"] == "clarify" for u in r.summary["said"])),
              ("answer resumed with the step", lambda r: any(c["t"] >= 9 and c["op"] == "switch_to"
                                                             for c in commands(r, "interaction")))], seconds=13,
             changes=(Change("ambiguous", ASK), Change("goal_swap", REVISE, goal="plate", old_goal="microwave"))),
    Scenario("jev-wait-budget", "Jev: a question nobody answers → one report, then holding (D7)", [
        "--interaction", "jev", "--chooser", "keywords", "--hold-recheck", "2", "--say", "5:put it over there"],
             [("asked once", lambda r: sum(u["template"] == "clarify" for u in r.summary["said"]) == 1),
              ("reported waiting once", lambda r: sum(u["template"] == "waiting_for_user"
                                                      for u in r.summary["said"]) == 1),
              ("holding to the end", lambda r: holding_after(r, 6.0, 8.0))], seconds=15,
             changes=(Change("ambiguous", ASK),)),
    Scenario("step-events", "step-completion events from end-state checks (truth spoof) drive the planner; needs an S1 "
             "that completes steps (the scripted S1 has completed none in any recorded run)", [
        "--arm", "revisable-truth", "--verify", "truth", "--verify-every", "0.5", "--decide-every", "2.5"],
             [("step event posted", lambda r: bool(r.summary["step_events"])),
              ("a keyframe with the checker's evidence", lambda r: any(k["evidence"].startswith("truth")
                                                                     for k in r.summary["keyframes"]))],
             seconds=20, needs="remote"),
    Scenario("blocked", "blocked arm: monitor deviation → RETRY from the keyword stand-in (`rules`; Jev has no retry "
             "act, see jev-blocked)", [
        "--interaction", "rules", "--block-at", "6", "--block-s", "2"],
             [("deviation raised", lambda r: first(r, "deviation") is not None),
              ("retry from the interaction lane", lambda r: bool(commands(r, "interaction", "retry")))],
             seconds=12, needs="monitor", changes=(Change("deviation", REVISE),)),
    Scenario("jev-blocked", "Jev (keyword scores): blocked arm → monitor deviation → guarded wait (a deviation offers "
             "only continue / wait)", [
        "--interaction", "jev", "--chooser", "keywords", "--block-at", "6", "--block-s", "2"],
             [("deviation raised", lambda r: first(r, "deviation") is not None),
              ("guarded wait from the interaction lane", lambda r: bool(commands(r, "interaction", "guarded_wait"))),
              ("never retried", lambda r: not commands(r, op="retry"))],
             seconds=12, needs="monitor", changes=(Change("deviation", WAIT),)),
]


def run_case(s: Scenario, out: Path, extra: list[str]) -> dict:
    run_dir = out / s.name
    cmd = [sys.executable, str(HERE / "realtime.py"), *BASE, *s.args, "--seconds", str(s.seconds), "--out",
           str(run_dir), *extra]
    w0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"name": s.name, "ok": False, "error": proc.stderr[-2000:], "cmd": cmd}
    run = Run.load(run_dir)
    invariants = check(run, PANDA_OMRON)
    expectations = {}
    for label, fn in s.expect:
        try:
            expectations[label] = bool(fn(run))
        except Exception as e:
            expectations[label] = f"error: {type(e).__name__}: {e}"
    ok = all(not v for v in invariants.values()) and all(v is True for v in expectations.values())
    return {"name": s.name, "what": s.what, "ok": ok, "wall_s": round(time.perf_counter() - w0, 1),
            "invariants": invariants, "expectations": expectations, "run": str(run_dir),
            "realtime_factor": run.summary["realtime_factor"], "starved_ticks": run.summary["starved_ticks"]}


def schedule_mismatch(s: Scenario, summary: dict) -> str | None:
    """Why a recorded run did not inject what `s` schedules (summary `say` and `blocked` against its arguments), or
    None when it did: a past run of a since-edited scenario is not scored under the new declaration."""
    schedule = scheduled(s.args)
    said = [[t, text] for t, trigger, text in schedule if trigger == "instruction"]
    blocked = [t for t, trigger, _ in schedule if trigger == "deviation"]
    recorded_block = [summary["blocked"]["at"]] if summary.get("blocked") else []
    if summary.get("say", []) != said:
        return f"run said {summary.get('say', [])}, scenario schedules {said}"
    if recorded_block != blocked:
        return f"run blocked at {recorded_block}, scenario schedules {blocked}"
    return None


def score_matrix(out: Path, report: list[dict]) -> list[ArmScore]:
    """Scores every reported case that declares changes and has a recorded run in `out/<name>`, adds its per-injection
    result to its report entry (`scorecard`), writes scorecard.md and scorecard.json into `out`; returns the arms."""
    by_name = {s.name: s for s in SCENARIOS}
    runs = []
    for entry in report:
        s = by_name.get(entry["name"])
        run_dir = out / entry["name"]
        if s is None or not s.changes or not (run_dir / "summary.json").exists():
            continue
        run = Run.load(run_dir)
        mismatch = schedule_mismatch(s, run.summary)
        frames = run.summary.get("frame_check")
        if mismatch is None and frames is not None and not frames["valid"]:
            mismatch = f"camera frames not real: {frames['bad']}, first {frames['first_bad']}"
        if mismatch is not None:
            entry["scorecard"] = {"skipped": mismatch}
            continue
        result = score_run(s.case(), RunRecords(run.summary, run.events, run.ticks, run.decisions))
        entry["scorecard"] = asdict(result)
        runs.append(result)
    arms = aggregate(runs)
    (out / "scorecard.md").write_text(render_markdown(arms, runs))
    (out / "scorecard.json").write_text(json.dumps(render_json(arms, runs), indent=2))
    return arms


def write_scored(out: Path, report: list[dict]) -> None:
    """Scores the matrix in `out`, rewrites report.json with each case's scores, prints the per-arm table."""
    arms = score_matrix(out, report)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    skipped = [f"{e['name']}: {e['scorecard']['skipped']}" for e in report if "skipped" in e.get("scorecard", {})]
    for line in skipped:
        print(f"[score skip] {line}")
    print(render_markdown(arms), end="")
    print(f"[scorecard] {sum(a.runs for a in arms)} cases scored -> {out / 'scorecard.md'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", help="scenario names")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--trace", metavar="PROJECT")
    p.add_argument("--policy", choices=["skills", "remote"], default="skills")
    p.add_argument("--service", type=Path)
    p.add_argument("--monitor", type=Path, help="execution-monitor model for the `blocked` case")
    p.add_argument("--no-video", action="store_true", help="skip view.mp4 per case")
    p.add_argument("--out", type=Path)
    p.add_argument("--score-only", type=Path, metavar="RUN_DIR",
                   help="score an existing matrix directory (its report.json and run directories); runs nothing")
    a = p.parse_args(argv)
    if a.score_only is not None:
        write_scored(a.score_only, json.loads((a.score_only / "report.json").read_text()))
        return 0
    out = a.out or REPO / "results/scenarios" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    extra = ["--policy", a.policy] + (["--service", str(a.service)] if a.service else []) + \
        (["--viewer"] if a.viewer else []) + (["--trace", a.trace] if a.trace else []) + \
        ([] if a.no_video else ["--video"])
    report = []
    for s in SCENARIOS:
        if a.only and s.name not in a.only:
            continue
        if s.needs == "monitor":
            if a.monitor is None:
                report.append({"name": s.name, "ok": None, "skipped": "needs --monitor MODEL.json"})
                print(f"[skip] {s.name}: needs --monitor")
                continue
            case_extra = extra + ["--execution-monitor", str(a.monitor)]
        elif s.needs == "remote" and a.policy != "remote":
            report.append({"name": s.name, "ok": None, "skipped": "needs --policy remote (an S1 that completes steps)"})
            print(f"[skip] {s.name}: needs --policy remote")
            continue
        else:
            case_extra = extra
        print(f"[run ] {s.name}: {s.what}", flush=True)
        result = run_case(s, out, case_extra)
        report.append(result)
        failed = [k for k, v in result.get("invariants", {}).items() if v] + \
            [k for k, v in result.get("expectations", {}).items() if v is not True]
        print(f"[{'pass' if result['ok'] else 'FAIL'}] {s.name}" + (f"  ✗ {failed}" if failed else "") +
              (f"  error: {result['error'][-300:]}" if "error" in result else ""), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    passed = sum(r["ok"] is True for r in report)
    print(f"[scenarios] {passed}/{sum(r['ok'] is not None for r in report)} passed -> {out / 'report.json'}")
    write_scored(out, report)
    return 0 if all(r["ok"] is not False for r in report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
