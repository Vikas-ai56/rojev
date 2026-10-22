"""Validate the bf16-trunk MJev (what `serve_mjev.py` runs on a T4) against upstream fp32 language weights.

Inputs: `testdata/<task>/` (built by `make_testdata.py`: randomly sampled RoboCasa scenes, 3 cameras each, and the
subtasks the candidate filter shortlists for each instruction). Same arm-D questions, first fp32 on CPU (reference),
then bf16 on the GPU. Reports top-1 agreement and the largest probability difference; writes the JSON report.

  uv run python check_precision.py --checkpoint <dir> --base-dir <qwen snapshot> --device cuda:0 --out report.json
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from serve_mjev import ask, load

HERE = Path(__file__).resolve().parent
# Same template arm D sends (rojev/supervisor/onejev_arm.py), for subtasks of each test scene.
STATES = {"done": "done: its goal is already achieved",
          "possible": "possible: it can be started now; what it needs is present and reachable",
          "blocked": "blocked: it cannot be done now; something it needs is missing, hidden, or not ready"}
def questions(subtasks: list[str]) -> list[dict]:
    return [{"id": f"s{i}", "question": f"Robot step: '{s}'. Which is true right now?",
             "answer_space": {"kind": "choice", "options": [{"id": k, "text": v} for k, v in STATES.items()]}}
            for i, s in enumerate(subtasks)]


def scenes(root: Path) -> dict[str, tuple[list[bytes], list[str]]]:
    out = {}
    for meta in sorted(root.glob("*/meta.json")):
        m = json.loads(meta.read_text())
        out[m["task"]] = ([p.read_bytes() for p in sorted(meta.parent.glob("*.png"))], m["subtasks"])
    return out


def run(model, builder, device: str, root: Path) -> dict:
    return {task: ask(model, builder, device, images, questions(subtasks))
            for task, (images, subtasks) in scenes(root).items()}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-dir", type=Path, help="base model snapshot at the checkpoint's base_revision")
    p.add_argument("--testdata", type=Path, default=HERE / "testdata", help="output of make_testdata.py")
    a = p.parse_args()
    model, builder = load(a.checkpoint, "cpu", "fp32", a.base_dir)
    ref = run(model, builder, "cpu", a.testdata)
    del model
    gc.collect()
    model, builder = load(a.checkpoint, a.device, "bf16", a.base_dir)
    test = run(model, builder, a.device, a.testdata)
    rows = []
    for scene in ref:
        for qid, r in ref[scene]["answers"].items():
            t = test[scene]["answers"][qid]
            diff = max(abs(r["probabilities"][k] - t["probabilities"][k]) for k in r["probabilities"])
            rows.append({"scene": scene, "question": qid, "fp32": r["probabilities"], "bf16": t["probabilities"],
                         "top1_agree": r["prediction"] == t["prediction"], "max_abs_prob_diff": round(diff, 4)})
    report = {"rows": rows, "top1_agreement": sum(r["top1_agree"] for r in rows) / len(rows),
              "max_abs_prob_diff": max(r["max_abs_prob_diff"] for r in rows),
              "timings_ms": {"fp32_cpu": {s: v["timings_ms"] for s, v in ref.items()},
                             "bf16_gpu": {s: v["timings_ms"] for s, v in test.items()}},
              "gpu_memory_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if a.device.startswith("cuda") else None}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
