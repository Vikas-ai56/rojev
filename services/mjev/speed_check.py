"""MJev speed levers on a T4 vs the fp32 reference from `check_precision.py` (E11 → E12).

Variants: image tokens per image (1024 = predict.py, 512 = training), trunk dtype (bf16 / fp16; T4 has fast fp16 only),
cameras per question (3 / 2). Each variant answers the same scenes; reported per variant: median latency per request
(3 questions), top-1 agreement and max |Δp| vs the fp32 reference, NaN check.

  uv run python speed_check.py --checkpoint <dir> --base-dir <dir> --testdata <dir> --reference mjev_precision.json \
      --out mjev_speed.json
"""
import argparse
import gc
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from spatial_jev.kev import KevBuilder, build_model, make_processor

from check_precision import questions, scenes
from serve_mjev import ask, fast_patch_embed

VARIANTS = [  # (name, trunk dtype, image tokens per image, cameras kept)
    ("bf16-1024-3cam", torch.bfloat16, 1024, 3), ("bf16-512-3cam", torch.bfloat16, 512, 3),
    ("fp16-1024-3cam", torch.float16, 1024, 3), ("fp16-512-3cam", torch.float16, 512, 3),
    ("fp16-512-2cam", torch.float16, 512, 2),
]


def load(checkpoint: Path, base: Path, dtype, image_tokens: int):
    config = json.loads((checkpoint / "config.json").read_text())
    config.update(preserve_option_order=True, max_image_tokens=image_tokens, base_model=str(base),
                  max_sequence_length=max(8192, int(config.get("max_sequence_length", 4096))),
                  inference_language_dtype="bf16")
    builder = KevBuilder(make_processor(config), config)
    model = build_model(config, checkpoint, trainable=False)
    lm = model.native.language_model
    lm.to(dtype)
    lm.native_qk.float()  # readout stays fp32, as in serve_mjev.py
    model.native.visual.to(dtype)
    model = model.to("cuda:0").eval()
    fast_patch_embed(model)  # E12e: same output, ≈ 4000× faster patch embedding on T4
    return model, builder


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--checkpoint", "--base-dir", "--testdata", "--reference", "--out"):
        p.add_argument(name, type=Path, required=True)
    p.add_argument("--variants", default=",".join(v[0] for v in VARIANTS), help="comma-separated subset")
    a = p.parse_args()
    # Which Gated-DeltaNet path transformers will take: fast FLA kernels or the PyTorch fallback (E12).
    print(json.dumps({"fla_installed": importlib.util.find_spec("fla") is not None,
                      "causal_conv1d_installed": importlib.util.find_spec("causal_conv1d") is not None}), flush=True)
    ref = {(r["scene"], r["question"]): r["fp32"] for r in json.loads(a.reference.read_text())["rows"]}
    data = scenes(a.testdata)  # images sorted: agentview_left, agentview_right, eye_in_hand
    results = []
    for name, dtype, tokens, cams in [v for v in VARIANTS if v[0] in a.variants.split(",")]:
        model, builder = load(a.checkpoint, a.base_dir, dtype, tokens)
        first = next(iter(data.values()))
        ask(model, builder, "cuda:0", first[0][:cams], questions(first[1]))  # warm-up
        times, diffs, agree, nan = [], [], [], False
        for scene, (images, subtasks) in data.items():
            t0 = time.perf_counter()
            out = ask(model, builder, "cuda:0", images[:cams], questions(subtasks))
            times.append((time.perf_counter() - t0) * 1000)
            for qid, ans in out["answers"].items():
                probs, fp32 = ans["probabilities"], ref[(scene, qid)]
                nan = nan or any(v != v for v in probs.values())
                diffs.append(max(abs(probs[k] - fp32[k]) for k in fp32))
                agree.append(max(probs, key=probs.get) == max(fp32, key=fp32.get))
        row = {"variant": name, "median_ms_per_request": round(statistics.median(times), 1),
               "top1_agreement_vs_fp32": round(sum(agree) / len(agree), 3), "max_abs_prob_diff": round(max(diffs), 4),
               "nan": nan, "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        print(json.dumps(row), flush=True)
        results.append(row)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    a.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
