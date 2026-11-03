"""RynnValue-4B as a step-completion signal, scored on E17's simulator-labelled kitchen frames (no training).

RynnValue (`Alibaba-DAMO-Academy/RynnValue-4B`, Apache-2.0) reads an instruction and an ordered frame sequence and
returns, per frame, the predicted seconds left until the instruction is done (value head), then generates an analysis
ending in `Match: Yes/No` and `Success: Yes/No`. P(Yes) for each verdict is read from the generation logits.

  smoke  one video, three prefixes and a mismatched instruction: proves the install before any scoring.
  score  every (frame, step) pair of `<done-check>/scores.jsonl` (the questions MJev and OneJev answered in E17),
         the step as the instruction:
           still  the one labelled frame, each camera in --cameras asked separately (E17's input);
           chain  the seed's states up to the labelled one (initial → placed → door closed → on) as an ordered sequence
                  from each camera, closer to the trajectories the model was trained on;
           repeat control for chain: its first frame repeated to the same length. Later states get longer chains, so
                  a drop in remaining time with length alone would score as "done" without reading the step; the
                  report also gives the within-frame AUROC (done vs not-done steps on identical images).

Loading: the revision pinned in config/rynnvalue_weights.txt; bf16 weights quantised to int8 weight-only (torchao) as
they load, so the 10.3 GB checkpoint fits in about 4 GB; causal sdpa attention, which the model card's reported numbers
use. Default device is the Apple GPU (mps): PyTorch's CPU path on macOS has no fast bf16 matmul. On mps, long visual
inputs (8 frames at 640 × 360) came back non-finite while one 256 × 256 frame matched across attention kernels, so smoke
frames are scaled to 320 px and every answer records `valid`: finite logits and remaining times inside the value head's
support. Invalid answers are counted and left out of the metrics.

Answers are appended to `<out>/answers.jsonl` (a rerun skips answered questions); `<out>/report.json` holds, per mode
and camera and per step type: balanced accuracy, recall on done and false-done rate for `Success: Yes` and for
P(Success) ≥ 0.7, rank AUROC of P(Success) and of the remaining time (pooled, and within frame: done vs not-done steps
on identical images), median remaining time per chain length, and median latency. "any view" counts a pair done when
any camera says so.

  uv run python check_done.py smoke --video <mp4> --instruction "Put the box in the drawer and close it"
  uv run python check_done.py score ../../results/done-check/20261022 --mode still --out ../../results/rynnvalue-check/<stamp>
  uv run python check_done.py report ../../results/rynnvalue-check/<stamp>
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import psutil
import torch
from PIL import Image
from torchao.quantization import Int8WeightOnlyConfig
from transformers import AutoConfig, AutoModel, AutoProcessor, TorchAoConfig

PIN_FILE = Path(__file__).resolve().parents[2] / "config/rynnvalue_weights.txt"
PIN_LINE = re.compile(r"^repo\s+(\S+)\s+@\s+([0-9a-f]{40})\s*$")
ROBOT = "a Franka single-arm robot on a mobile base"
CAMERAS = {"robot0_agentview_left": "the left third-person camera",
           "robot0_agentview_right": "the right third-person camera",
           "robot0_eye_in_hand": "the wrist camera"}
MAX_NEW_TOKENS = 96
SMOKE_MAX_SIDE = 320
THRESHOLD = 0.7
VERDICT_PROMPT = re.compile(r"(Match|Success):\s*$")


def pinned_model() -> tuple[str, str]:
    for line in PIN_FILE.read_text().splitlines():
        if match := PIN_LINE.match(line):
            return match.group(1), match.group(2)
    raise SystemExit(f"{PIN_FILE}: no line `repo <id> @ <40-hex revision>`")


class RynnValue:
    def __init__(self, device: str):
        model_id, revision = pinned_model()
        self.name = f"{model_id}@{revision[:7]}"
        config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_id, revision=revision, config=config, dtype=torch.bfloat16, trust_remote_code=True,
            attn_implementation="sdpa", quantization_config=TorchAoConfig(quant_type=Int8WeightOnlyConfig()),
            device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.device = torch.device(device)
        self.eos = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.answer_ids = {word: sorted({self.tokenizer.convert_tokens_to_ids(t) for t in (word, "Ġ" + word)})
                           for word in ("Yes", "No")}

    def ask(self, instruction: str, images: list[Image.Image], camera: str) -> dict:
        started = time.perf_counter()
        sample = self.processor.process_episode(instruction=instruction, images=images, robot_description=ROBOT,
                                                camera_description=camera)
        inputs = {"input_ids": sample["input_ids"].long(), "attention_mask": sample["attention_mask"].long(),
                  "pixel_values": sample["pixel_values"].flatten(0, 1),
                  "image_grid_thw": sample["image_grid_thw"].flatten(0, 1).long()}
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            values = self.model(**inputs).value.pred_value.float()
            generated = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                                            eos_token_id=self.eos, pad_token_id=self.eos, output_scores=True,
                                            return_dict_in_generate=True)
        remaining = values.reshape(-1, len(images)).mean(dim=0)
        tokens = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
        text, verdicts = self.read_verdicts(tokens, generated.scores)
        support = self.model.config.value_tokenizer_config
        valid = (bool(torch.isfinite(remaining).all()) and all(bool(torch.isfinite(s).all()) for s in generated.scores)
                 and bool(((remaining >= support.min_value) & (remaining <= support.max_value)).all()))
        return {"valid": valid, "remaining_s": [round(v, 2) for v in remaining.tolist()], "analysis": text,
                "match": verdicts.get("Match"), "success": verdicts.get("Success"),
                "ms": round((time.perf_counter() - started) * 1000, 1)}

    def read_verdicts(self, tokens: list[int], scores: tuple[torch.Tensor, ...]) -> tuple[str, dict]:
        text, verdicts = "", {}
        for token, logits in zip(tokens, scores):
            piece = self.tokenizer.decode([token], skip_special_tokens=True)
            prompt = VERDICT_PROMPT.search(text)
            if prompt and piece.strip() in ("Yes", "No") and prompt.group(1) not in verdicts:
                probs = torch.softmax(logits[0].float(), dim=-1)
                yes, no = (float(probs[self.answer_ids[w]].sum()) for w in ("Yes", "No"))
                verdicts[prompt.group(1)] = {"answer": piece.strip(), "p_yes": round(yes / (yes + no), 4)}
            text += piece
        return text.strip(), verdicts


def memory_report() -> str:
    rss = psutil.Process().memory_info().rss / 1e9
    available = psutil.virtual_memory().available / 1e9
    return f"process {rss:.1f} GB resident, system {available:.1f} GB available"


def load(device: str) -> RynnValue:
    started = time.perf_counter()
    model = RynnValue(device)
    print(f"loaded {model.name} on {device} in {time.perf_counter() - started:.0f} s; {memory_report()}", flush=True)
    return model


def smoke(args) -> int:
    frames = [Image.fromarray(frame).convert("RGB") for frame in iio.imiter(args.video)]
    for frame in frames:
        frame.thumbnail((SMOKE_MAX_SIDE, SMOKE_MAX_SIDE), Image.BICUBIC)
    print(f"{len(frames)} frames of {frames[0].size}")
    model = load(args.device)
    camera = "the main right camera"
    for fraction in (0.33, 0.66, 1.0):
        end = max(round(fraction * (len(frames) - 1)), 1)
        picked = [frames[i] for i in np.linspace(0, end, args.frames, dtype=int)]
        answer = model.ask(args.instruction, picked, camera)
        print(json.dumps({"prefix": fraction, **answer}), flush=True)
    picked = [frames[i] for i in np.linspace(0, len(frames) - 1, args.frames, dtype=int)]
    print(json.dumps({"mismatch": args.mismatch, **model.ask(args.mismatch, picked, camera)}), flush=True)
    print(memory_report())
    return 0


def labelled_pairs(done_check: Path) -> list[dict]:
    rows = [json.loads(line) for line in (done_check / "scores.jsonl").open()]
    verifier = rows[0]["verifier"]
    return [{k: row[k] for k in ("frame", "task", "seed", "state", "step", "truth")}
            for row in rows if row["verifier"] == verifier]


def frame_sequence(done_check: Path, pair: dict, camera: str, mode: str) -> list[Image.Image]:
    prefix, index = pair["frame"].rsplit("-", 1)
    chain = [f"{prefix}-{i}" for i in range(int(index) + 1) if (done_check / f"{prefix}-{i}").is_dir()]
    names = {"still": [pair["frame"]], "chain": chain, "repeat": [chain[0]] * len(chain)}[mode]
    return [Image.open(done_check / name / f"{camera}.png").convert("RGB") for name in names]


def score(args) -> int:
    done_check, out = Path(args.done_check), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answers_path = out / "answers.jsonl"
    answered = set()
    if answers_path.exists():
        answered = {(a["mode"], a["camera"], a["frame"], a["step"]) for a in map(json.loads, answers_path.open())}
    pairs = labelled_pairs(done_check)
    if args.limit:
        pairs = pairs[:args.limit]
    todo = [(camera, pair) for pair in pairs for camera in args.cameras
            if (args.mode, camera, pair["frame"], pair["step"]) not in answered]
    print(f"{len(pairs)} pairs × {len(args.cameras)} cameras, {len(todo)} to ask ({args.mode})", flush=True)
    if not todo:
        return report_main(argparse.Namespace(out=args.out))
    model = load(args.device)
    with answers_path.open("a") as sink:
        for n, (camera, pair) in enumerate(todo, 1):
            images = frame_sequence(done_check, pair, camera, args.mode)
            answer = model.ask(pair["step"], images, CAMERAS[camera])
            if args.device == "mps":
                torch.mps.empty_cache()
            record = {"mode": args.mode, "camera": camera, "frames": len(images), **pair, **answer}
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            success = (answer["success"] or {}).get("answer")
            print(f"[{n}/{len(todo)}] {pair['frame']} {camera[7:]} '{pair['step']}' truth={pair['truth']} "
                  f"success={success} left={answer['remaining_s'][-1]}s {answer['ms'] / 1000:.1f}s", flush=True)
    print(memory_report())
    return report_main(argparse.Namespace(out=args.out))


def auroc(scores: list[float], truth: list[bool]) -> float | None:
    positives = [s for s, t in zip(scores, truth) if t]
    negatives = [s for s, t in zip(scores, truth) if not t]
    if not positives or not negatives:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in positives for q in negatives)
    return round(wins / (len(positives) * len(negatives)), 3)


def decision_metrics(said_done: list[bool], truth: list[bool]) -> dict:
    tp = sum(s and t for s, t in zip(said_done, truth))
    fp = sum(s and not t for s, t in zip(said_done, truth))
    pos, neg = sum(truth), len(truth) - sum(truth)
    recall = tp / pos if pos else None
    false_done = fp / neg if neg else None
    balanced = (recall + 1 - false_done) / 2 if recall is not None and false_done is not None else None
    return {"balanced_accuracy": balanced and round(balanced, 3), "recall": recall and round(recall, 3),
            "false_done": false_done and round(false_done, 3), "pos": pos, "neg": neg}


def within_frame_auroc(records: list[dict], score) -> float | None:
    by_frame = defaultdict(list)
    for r in records:
        by_frame[r["frame"]].append(r)
    wins = pairs = 0.0
    for rows in by_frame.values():
        done = [score(r) for r in rows if r["truth"]]
        not_done = [score(r) for r in rows if not r["truth"]]
        for p in done:
            for q in not_done:
                pairs += 1
                wins += (p > q) + 0.5 * (p == q)
    return round(wins / pairs, 3) if pairs else None


def remaining_by_length(records: list[dict]) -> dict:
    lengths = defaultdict(list)
    for r in records:
        lengths[r["frames"]].append(r["remaining_s"][-1])
    return {k: round(float(np.median(v)), 2) for k, v in sorted(lengths.items())}


def summarise(all_records: list[dict]) -> dict:
    records = [r for r in all_records if r["valid"]]
    truth = [r["truth"] for r in records]
    p_success = [(r["success"] or {}).get("p_yes", 0.0) for r in records]
    said_yes = [(r["success"] or {}).get("answer") == "Yes" for r in records]
    left = [r["remaining_s"][-1] for r in records]
    matched = [(r["match"] or {}).get("answer") == "Yes" for r in records]
    return {"n": len(records), "invalid": len(all_records) - len(records),
            "success_yes": decision_metrics(said_yes, truth),
            f"p_success>={THRESHOLD}": decision_metrics([p >= THRESHOLD for p in p_success], truth),
            "auroc_p_success": auroc(p_success, truth),
            "auroc_remaining": auroc([-v for v in left], truth),
            "within_frame_auroc_p_success": within_frame_auroc(records, lambda r: (r["success"] or {}).get("p_yes", 0.0)),
            "within_frame_auroc_remaining": within_frame_auroc(records, lambda r: -r["remaining_s"][-1]),
            "remaining_s_median_by_frames": remaining_by_length(records),
            "remaining_s_median": {"done": float(np.median([v for v, t in zip(left, truth) if t] or [np.nan])),
                                   "not_done": float(np.median([v for v, t in zip(left, truth) if not t] or [np.nan]))},
            "match_yes_rate": round(sum(matched) / len(matched), 3),
            "no_success_verdict": sum(r["success"] is None for r in records),
            "ms_p50": float(np.median([r["ms"] for r in records]))}


def step_type(step: str) -> str:
    return step.split()[0]


def report_main(args) -> int:
    out = Path(args.out)
    records = [json.loads(line) for line in (out / "answers.jsonl").open()]
    groups = defaultdict(list)
    for r in records:
        groups[(r["mode"], r["camera"])].append(r)
    report = {}
    for (mode, camera), rows in sorted(groups.items()):
        per_type = defaultdict(list)
        for r in rows:
            per_type[step_type(r["step"])].append(r)
        report[f"{mode}/{camera}"] = {"all": summarise(rows),
                                      "per_step_type": {k: summarise(v) for k, v in sorted(per_type.items())}}
    by_pair = defaultdict(dict)
    for r in records:
        by_pair[(r["mode"], r["frame"], r["step"])][r["camera"]] = r
    for mode in sorted({r["mode"] for r in records}):
        complete = [views for (m, _, _), views in by_pair.items() if m == mode and len(views) == len(CAMERAS)]
        if complete:
            truth = [next(iter(v.values()))["truth"] for v in complete]
            any_yes = [any((r["success"] or {}).get("answer") == "Yes" for r in v.values()) for v in complete]
            mean_p = [float(np.mean([(r["success"] or {}).get("p_yes", 0.0) for r in v.values()])) for v in complete]
            report[f"{mode}/any view"] = {"n": len(complete), "success_yes": decision_metrics(any_yes, truth),
                                          "auroc_mean_p_success": auroc(mean_p, truth)}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    for key, value in report.items():
        summary = value.get("all", value)
        print(key, json.dumps({k: summary[k] for k in summary if k in
                               ("n", "invalid", "success_yes", "auroc_p_success", "auroc_remaining",
                                "within_frame_auroc_p_success", "within_frame_auroc_remaining",
                                "remaining_s_median_by_frames", "auroc_mean_p_success", "match_yes_rate", "ms_p50")}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--video", required=True)
    smoke_parser.add_argument("--instruction", required=True)
    smoke_parser.add_argument("--mismatch", default="Pour water from the bottle into the cup")
    smoke_parser.add_argument("--frames", type=int, default=8)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("done_check")
    score_parser.add_argument("--mode", choices=("still", "chain", "repeat"), default="still")
    score_parser.add_argument("--cameras", nargs="+", choices=tuple(CAMERAS), default=list(CAMERAS))
    score_parser.add_argument("--limit", type=int, default=0, help="first N pairs only")
    score_parser.add_argument("--out", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("out")
    for sub in (smoke_parser, score_parser):
        sub.add_argument("--device", choices=("mps", "cpu"), default="mps",
                         help="mps: Apple GPU; cpu has no fast bf16 matmul on macOS (generic fallback kernel)")
    args = parser.parse_args()
    return {"smoke": smoke, "score": score, "report": report_main}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
