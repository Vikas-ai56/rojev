"""Fit Jev's interaction `Calibration` (temperature + abstention margin) from labelled cases.

Cases come from live runs with `realtime.py --interaction jev --record-interaction DIR`: each is the exact model input
(cameras, situation, legal options) of one decision. A person sets `label` (the right act) in `DIR/labels.jsonl`;
unlabelled cases are skipped. Every labelled case is scored again by OneJev (so a new model or prompt is calibrated
on the same cases), then `fit_calibration` picks T by NLL and the smallest margin whose decided cases reach
`--target-accuracy`.

  uv run python calibrate_jev.py DIR --service ~/.config/rojev/services/lambda.json --out config/jev_calibration.json
Prints cases, accuracy, NLL at T = 1 and at the fitted T, margin and coverage; writes the `Calibration` JSON that
`realtime.py --jev-calibration` loads.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from qev import Client

from rojev.system2.jev import ABORT, ASK, CONTINUE, RESUME, RETRY, REVISE, WAIT, OneJevChooser, Option, fit_calibration
from rojev.system2.jev.calibration import margins, nll
from workflow import service_addresses

ACTS = (CONTINUE, REVISE, WAIT, ASK, ABORT, RETRY, RESUME)


def labelled(folder: Path) -> list[tuple[dict, str]]:
    """(case, label) for every case a person labelled; a label must be one of the case's legal options."""
    out = []
    for line in (folder / "labels.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("label") is None:
            continue
        case = json.loads((folder / row["case"] / "case.json").read_text())
        if row["label"] not in case["options"]:
            raise SystemExit(f"case {row['case']}: label {row['label']!r} is not a legal option {list(case['options'])}")
        out.append((case, row["label"]))
    return out


def images(folder: Path, case: dict) -> list[np.ndarray]:
    return [np.asarray(Image.open(folder / case["case"] / f"{cam}.png").convert("RGB")) for cam in case["cameras"]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", type=Path)
    p.add_argument("--service", required=True, help="service JSON with an `onejev` entry")
    p.add_argument("--target-accuracy", type=float, default=0.9)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    cases = labelled(a.cases)
    if not cases:
        raise SystemExit(f"no labelled cases in {a.cases / 'labels.jsonl'}")
    cameras = cases[0][0]["cameras"]
    chooser = OneJevChooser(Client(base_url=service_addresses(a.service, "serve")["onejev"]["url"]), cameras)
    rows, labels = [], []
    for case, label in cases:
        options = [Option(k, v) for k, v in case["options"].items()]
        scores = chooser.score(images(a.cases, case), case["situation"], options)
        rows.append({o.label: scores.get(o.label, 0.0) for o in options})
        labels.append(label)
    calibration = fit_calibration(rows, labels, a.target_accuracy)
    _, tops = margins(rows, 1.0)
    report = {"cases": len(rows), "accuracy": round(float(np.mean([t == y for t, y in zip(tops, labels)])), 3),
              "nll_t1": round(nll(rows, labels, 1.0), 4),
              "nll_fitted": round(nll(rows, labels, calibration.temperature), 4),
              "temperature": round(calibration.temperature, 3), "margin": round(calibration.margin, 3),
              "coverage": calibration.coverage, "target_accuracy": a.target_accuracy,
              "labels": {act: labels.count(act) for act in ACTS if act in labels}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(calibration.to_json())
    print(json.dumps(report, indent=2))
    print(f"[calibrate_jev] -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
