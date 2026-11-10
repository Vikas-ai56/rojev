"""Runs Embodied-R1.5 over S2 bake-off case sets (one model load for all) and writes its answers; scoring happens
on the Mac (`executors/so101-data/score.py`).

  python bench.py --cases ~/rojev-data/s2-cases/so101-steps ~/rojev-data/s2-cases/armnetbench --out <dir>

Cases follow the contract in `executors/so101-data/cases.py`. next_step cases are asked once over their top-camera
frames; status cases once per camera (the wrist sees lid, grasp and stack that the arm hides from the top camera).
All questions go to vLLM as one batch; then LATENCY_SAMPLES questions of each kind are repeated one at a time, the
latency an event-driven S2 sees. Writes, per set, `<out>/<set>/answers.jsonl` (id, camera, raw text, parsed answer,
format_ok, new tokens) and `<out>/<set>/timing.json`. Weights: the revision pinned in `config/embodied_r1_weights.txt`.
"""
import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import prompts

ROOT = Path(__file__).resolve().parents[2]
PIN_FILE = ROOT / "config/embodied_r1_weights.txt"
LATENCY_SAMPLES = 8


@dataclass(frozen=True)
class Question:
    case: dict
    camera: str
    text: str
    paths: list[str]


def questions(cases: list[dict]) -> list[Question]:
    out = []
    for case in cases:
        if case["kind"] == "next_step":
            text = prompts.next_step(case["goal"], case["record"], case["options"])
            out.append(Question(case, "top", text, case["images"]["top"]))
        else:
            for camera, paths in sorted(case["images"].items()):
                out.append(Question(case, camera, prompts.status(case["step"]), paths))
    return out


def read(question: Question, raw: str) -> prompts.Parsed:
    if question.case["kind"] == "next_step":
        return prompts.read_next_step(raw, question.case["options"])
    return prompts.read_status(raw)


def run(engine, request_type, cases_dir: Path, out: Path) -> dict:
    cases = [json.loads(line) for line in (cases_dir / "cases.jsonl").read_text().splitlines()]
    asked = questions(cases)
    requests = [request_type([Image.open(cases_dir / p).convert("RGB") for p in q.paths], q.text) for q in asked]
    start = time.perf_counter()
    replies = engine.generate(requests)
    batch_s = time.perf_counter() - start
    out.mkdir(parents=True, exist_ok=True)
    with (out / "answers.jsonl").open("w") as f:
        for question, reply in zip(asked, replies):
            parsed = read(question, reply.text)
            f.write(json.dumps({"id": question.case["id"], "camera": question.camera, "raw": reply.text,
                                "answer": parsed.value, "format_ok": parsed.format_ok,
                                "new_tokens": reply.new_tokens}) + "\n")
    latency = {}
    for kind in ("next_step", "status"):
        sample = [r for q, r in zip(asked, requests) if q.case["kind"] == kind][:LATENCY_SAMPLES]
        latency[kind] = sorted(engine.latency_ms(r) for r in sample)
    timing = {"questions": len(requests), "batch_s": batch_s, "single_question_ms": latency}
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    return timing


def snapshot() -> str:
    return subprocess.run([sys.executable, str(ROOT / "scripts/hf_pins.py"), "snapshot", "--pin-file", str(PIN_FILE),
                           "--offline"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from engine import Engine, Request

    engine = Engine.load(snapshot())
    for cases_dir in a.cases:
        print(cases_dir.name, json.dumps(run(engine, Request, cases_dir, a.out / cases_dir.name)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
