"""Checks over one run's records (ticks.jsonl, owner.jsonl, summary.json): the properties every run must keep.

  no_stale_action      no chunk action of a command after its `revoked` / `expired` / `dropped` ACK
  switches_held        no irreversible switch (the robot's `ActionSpec`) changes on a tick under review or holding
  active_monotonic     activated commands only ever newer (cmd ids strictly increase)
  changes_answered     every user instruction is followed by an applied decision (command, reaffirm or continue)
                       before the next instruction or the run end (when at least `answer_window_s` remains)
  no_violations        no action clamped by the robot's bounds

Each check returns a list of findings (empty = holds); `check(run, spec)` runs them all for the robot whose
`ActionSpec` produced the run. Scenario-specific expectations live in scenarios.py.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rojev.robot import ActionSpec


@dataclass
class Run:
    path: Path
    ticks: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    decisions: list[dict] = field(default_factory=list)
    commits: list[dict] = field(default_factory=list)

    @staticmethod
    def load(path: Path) -> "Run":
        def jsonl(name: str) -> list[dict]:
            f = path / name
            return [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []

        return Run(path, jsonl("ticks.jsonl"), jsonl("owner.jsonl"), json.loads((path / "summary.json").read_text()),
                   jsonl("decisions.jsonl"), jsonl("commits.jsonl"))


def no_stale_action(run: Run) -> list[str]:
    dead = {}
    for e in run.events:
        if e["kind"] == "ack" and e["status"] in ("revoked", "expired", "dropped"):
            dead.setdefault(e["cmd"], e["t"])
    return [f"tick {t['t']}: chunk action of cmd {t['cmd']} after its end at {dead[t['cmd']]}"
            for t in run.ticks if t["source"] == "chunk" and t["cmd"] in dead and t["t"] > dead[t["cmd"]] + 1e-6]


def switches_held(run: Run, spec: ActionSpec) -> list[str]:
    out = []
    for prev, t in zip(run.ticks, run.ticks[1:]):
        if t["review"] or t["source"] == "hold":
            flipped = spec.irreversible_change(np.asarray(prev["action"]), np.asarray(t["action"]))
            if flipped:
                out.append(f"tick {t['t']}: {', '.join(flipped)} switched while "
                           f"{'under review' if t['review'] else 'holding'}")
    return out


def active_monotonic(run: Run) -> list[str]:
    activated = [e["cmd"] for e in run.events if e["kind"] == "ack" and e["status"] == "activated"]
    return [f"cmd {b} activated after cmd {a}" for a, b in zip(activated, activated[1:]) if b <= a]


def changes_answered(run: Run, answer_window_s: float = 2.0) -> list[str]:
    said = [e for e in run.events if e["kind"] == "instruction"]
    end = run.summary["sim_s"]
    out = []
    for i, e in enumerate(said):
        until = said[i + 1]["t"] if i + 1 < len(said) else end
        if until - e["t"] < answer_window_s:
            continue
        if not any(x["kind"] in ("command", "reaffirm", "continue") and e["t"] <= x["t"] <= until for x in run.events):
            out.append(f"instruction {e['text']!r} at {e['t']} never answered")
    return out


def no_violations(run: Run) -> list[str]:
    return [f"tick {t['t']}: {t['violations']}" for t in run.ticks if t["violations"]]


RUN_INVARIANTS = (no_stale_action, active_monotonic, changes_answered, no_violations)


def check(run: Run, spec: ActionSpec) -> dict[str, list[str]]:
    return {"switches_held": switches_held(run, spec), **{f.__name__: f(run) for f in RUN_INVARIANTS}}
