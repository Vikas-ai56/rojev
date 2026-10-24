"""Fit one of Jev's `Calibration`s (temperature + abstention margin) on labelled cases, and ship it only if it holds on
cases it was not fitted on.

Cases come from live runs with `realtime.py --interaction jev --record-interaction DIR`: each is the exact model input
(cameras, situation, question, options) of one question Jev asked. `question` is `act` (what the user wants) or `task`
(which trained task a revision means); each has its own calibration. A person sets `label` (the right option) in
`DIR/labels.jsonl`; unlabelled cases are skipped. Every labelled case is scored again by OneJev (so a new model or
prompt is calibrated on the same cases).

Two sets, never mixed: `--fit` folders (T by NLL on the options, then the smallest margin whose decided cases reach
`--target-accuracy`; for the act question the margin and accuracy are over effects, move / hold / switch, as Jev
decides, when the cases record them) and `--held-out` folders (other runs and scenes, labelled the same way). The calibration is
written into `--out` under its question only if the held-out decided accuracy reaches the target; otherwise the
report is printed, nothing is written and the exit status is 1. Cases recorded before an option set changed are not
valid for it (the 2026-10-23 cases predate the 2026-10-24 act set).

  uv run python calibrate_jev.py --question act --fit runs/a runs/b --held-out runs/c \\
      --service ~/.config/rojev/services/lambda.json --out config/jev_calibration.json
`--out` holds {"act": …, "task": …}; `realtime.py --jev-calibration` loads it (a missing question keeps the declared
defaults).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from rojev.system2.jev import ACT_QUESTION, TASK_QUESTION, Calibration, OneJevChooser, Option
from rojev.system2.jev.calibration import fit_margin, fit_temperature, margins, nll
from rojev.system2.models import onejev
from workflow import service_addresses

QUESTIONS = {"act": ACT_QUESTION, "task": TASK_QUESTION}


def labelled(folder: Path, question: str = "act") -> list[tuple[dict, str]]:
    """(case, label) for every case of `question` a person labelled; a label must be one of the case's options."""
    out = []
    for line in (folder / "labels.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("label") is None or row.get("question", "act") != question:
            continue
        case = json.loads((folder / row["case"] / "case.json").read_text())
        if row["label"] not in case["options"]:
            raise SystemExit(f"case {row['case']}: label {row['label']!r} is not an option {list(case['options'])}")
        out.append((case, row["label"]))
    return out


def images(folder: Path, case: dict) -> list[np.ndarray]:
    return [np.asarray(Image.open(folder / case["case"] / f"{cam}.png").convert("RGB")) for cam in case["cameras"]]


def scored(chooser: OneJevChooser, folders: list[Path], question: str
           ) -> tuple[list[dict[str, float]], list[str], list[dict[str, str] | None]]:
    rows, labels, effects = [], [], []
    for folder in folders:
        for case, label in labelled(folder, question):
            options = [Option(k, v) for k, v in case["options"].items()]
            scores = chooser.score(images(folder, case), case["situation"], options, QUESTIONS[question])
            rows.append({o.label: scores.get(o.label, 0.0) for o in options})
            labels.append(label)
            effects.append(case.get("effects"))
    return rows, labels, effects


def grouped(rows: list[dict[str, float]], labels: list[str], effects: list[dict[str, str] | None], temperature: float
            ) -> tuple[list[dict[str, float]], list[str]]:
    """Rows as Jev decides on them: tempered over the options, then summed per effect (cases without effects stay
    per option)."""
    out_rows, out_labels = [], []
    for row, label, fx in zip(rows, labels, effects):
        tempered = Calibration(temperature=temperature).apply(row)
        if fx is None:
            out_rows.append(tempered)
            out_labels.append(label)
            continue
        summed: dict[str, float] = {}
        for k, p in tempered.items():
            summed[fx[k]] = summed.get(fx[k], 0.0) + p
        out_rows.append(summed)
        out_labels.append(fx[label])
    return out_rows, out_labels


def fit(rows, labels, effects, target: float) -> Calibration:
    temperature = fit_temperature(rows, labels)
    margin, coverage = fit_margin(*grouped(rows, labels, effects, temperature), 1.0, target)
    return Calibration(temperature, margin, True, len(rows), target, coverage)


def decided_accuracy(rows: list[dict[str, float]], labels: list[str], calibration: Calibration,
                     effects: list[dict[str, str] | None] | None = None) -> tuple[float | None, float]:
    """Accuracy of the cases the calibration decides (margin at or above its own, over effects when known), and the
    share it decides."""
    rows, labels = grouped(rows, labels, effects or [None] * len(rows), calibration.temperature)
    gaps, tops = margins(rows, 1.0)
    decided = gaps >= calibration.margin
    if not decided.any():
        return None, 0.0
    correct = np.array([t == y for t, y in zip(tops, labels)])
    return float(correct[decided].mean()), float(decided.mean())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--question", choices=list(QUESTIONS), required=True)
    p.add_argument("--fit", type=Path, nargs="+", required=True)
    p.add_argument("--held-out", type=Path, nargs="+", required=True)
    p.add_argument("--service", required=True, help="service JSON with an `onejev` entry")
    p.add_argument("--target-accuracy", type=float, default=0.9)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if set(a.fit) & set(a.held_out):
        raise SystemExit("a folder cannot be both fit and held-out")
    first = next((labelled(f, a.question) for f in a.fit if labelled(f, a.question)), None)
    if not first:
        raise SystemExit(f"no labelled {a.question} cases in the fit folders")
    chooser = OneJevChooser(onejev.KeptClient(service_addresses(a.service, "serve")["onejev"]["url"]),
                            first[0][0]["cameras"])
    fit_rows, fit_labels, fit_effects = scored(chooser, a.fit, a.question)
    held_rows, held_labels, held_effects = scored(chooser, a.held_out, a.question)
    if not held_rows:
        raise SystemExit(f"no labelled {a.question} cases in the held-out folders")
    calibration = fit(fit_rows, fit_labels, fit_effects, a.target_accuracy)
    accuracy, coverage = decided_accuracy(held_rows, held_labels, calibration, held_effects)
    shipped = accuracy is not None and accuracy >= a.target_accuracy
    report = {"question": a.question, "fit_cases": len(fit_rows), "held_out_cases": len(held_rows),
              "temperature": round(calibration.temperature, 3), "margin": round(calibration.margin, 3),
              "fit_coverage": calibration.coverage, "nll_fit_t1": round(nll(fit_rows, fit_labels, 1.0), 4),
              "nll_fit": round(nll(fit_rows, fit_labels, calibration.temperature), 4),
              "held_out_decided_accuracy": None if accuracy is None else round(accuracy, 3),
              "held_out_coverage": round(coverage, 3), "target_accuracy": a.target_accuracy, "shipped": shipped,
              "labels": {label: fit_labels.count(label) for label in sorted(set(fit_labels))}}
    print(json.dumps(report, indent=2))
    if not shipped:
        print(f"[calibrate_jev] held-out decided accuracy below {a.target_accuracy}: nothing written")
        return 1
    current = json.loads(a.out.read_text()) if a.out.exists() else {}
    current[a.question] = json.loads(calibration.to_json())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(current, indent=2))
    print(f"[calibrate_jev] {a.question} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
