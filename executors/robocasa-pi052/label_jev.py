"""Label recorded Jev interaction cases for calibration (`calibrate_jev.py`), one case at a time in the terminal.

For each unlabelled case in the given case directories (written by `realtime.py --record-interaction DIR`): the
cameras side by side are written to `<case>/view.png` and opened, and the terminal shows what the user just said, the
task, the step, the conversation, the question (`act`, or `task` for which trained task a revision means) and its
options with the model's prediction. Type the number of the right option, `s` to skip, `q` to stop; the label is
written back into that directory's `labels.jsonl` at once.

  uv run python label_jev.py results/gpu-suite/*/jev/cases [--relabel]
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def rows(folder: Path) -> list[dict]:
    return [json.loads(line) for line in (folder / "labels.jsonl").read_text().splitlines() if line.strip()]


def save(folder: Path, table: list[dict]) -> None:
    (folder / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in table))


def view(folder: Path, case: dict) -> Path:
    images = [np.asarray(Image.open(folder / case["case"] / f"{cam}.png").convert("RGB")) for cam in case["cameras"]]
    path = folder / case["case"] / "view.png"
    Image.fromarray(np.concatenate(images, axis=1)).save(path)
    return path


def show(case: dict, predicted: str | None) -> list[str]:
    s = case["situation"]
    print("\n" + "=" * 100)
    print(f"case {case['case']}  question {case.get('question', 'act')}  trigger {case['trigger']}")
    print(f"user just said: {s.get('user_just_said')!r}")
    print(f"task: {s.get('task')!r}\ncurrent step: {s.get('current_step')!r}  holding: {s.get('holding')}")
    print(f"conversation: {s.get('conversation')}")
    labels = list(case["options"])
    for i, label in enumerate(labels, start=1):
        mark = "  <- model" if label == predicted else ""
        print(f"  {i}. {label:<9} {case['options'][label]}{mark}")
    return labels


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", type=Path, nargs="+")
    p.add_argument("--relabel", action="store_true", help="also show cases that already have a label")
    a = p.parse_args()
    done = 0
    for folder in a.folders:
        table = rows(folder)
        for row in table:
            if row.get("label") is not None and not a.relabel:
                continue
            case = json.loads((folder / row["case"] / "case.json").read_text())
            subprocess.run(["open", str(view(folder, case))], check=False)
            labels = show(case, row.get("predicted"))
            while True:
                answer = input("right act (number, s = skip, q = quit): ").strip().lower()
                if answer == "q":
                    print(f"[label_jev] {done} labelled")
                    return 0
                if answer == "s":
                    break
                if answer.isdigit() and 1 <= int(answer) <= len(labels):
                    row["label"] = labels[int(answer) - 1]
                    save(folder, table)
                    done += 1
                    break
                print(f"  enter 1–{len(labels)}, s or q", file=sys.stderr)
    print(f"[label_jev] {done} labelled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
