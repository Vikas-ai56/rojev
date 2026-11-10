"""Input / output variety probe of an official GR00T checkpoint (GPU-session step `droid_probe`), one model load.

Observations come from a LeRobot dataset read with Isaac-GR00T's own `LeRobotEpisodeLoader` / `extract_step_data` and
shaped by its `parse_observation_gr00t`, as `open_loop_eval.py` does; `Gr00tPolicy` is called directly (no rojev
code in the path: this measures the model). For every episode in `--variants` (a JSON file: episode → {variant
name: instruction}, e.g. a paraphrase, another target, an unrelated sentence, `empty`; the same names across
episodes), `--points` evenly spaced time points; for each, the episode's own instruction (`original`) and every
variant, every seed in `--seeds`, every denoising-step count in `--denoising-steps`. Reported per (time point,
variant, denoising): the action error against the recorded chunk and
the spread across seeds; per variant: how far it moves the actions from `original` under the same seed; per denoising
setting: latency p50 / p95. A variant whose call fails (GR00T's DROID notes warn an empty instruction may) is recorded
with its error and counted, never stops the probe. Writes `probe.json` and `actions.npz` (every predicted chunk with
its labels) to `--out`.

`run` holds the loop and takes the model, the samples and the seeding function, so it runs on the Mac with stand-ins.

  uv run python probe.py --model <checkpoint> --embodiment-tag <tag> --dataset <lerobot dataset> \\
      --variants gpu/droid_variants.json --out <dir>
"""
import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORIGINAL = "original"


@dataclass(frozen=True)
class ProbeSample:
    episode: int
    step: int
    groot: dict
    recorded: np.ndarray
    text: str


def with_text(groot: dict, text: str) -> dict:
    return {**groot, "language": {key: [[text]] for key in groot["language"]}}


def flatten(action: dict, keys: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(action[key])[0] for key in keys], axis=-1)


def percentiles(ms: list[float]) -> dict:
    return {"n": len(ms), "p50": round(float(np.percentile(ms, 50)), 1), "p95": round(float(np.percentile(ms, 95)), 1),
            "max": round(float(np.max(ms)), 1)} if ms else {"n": 0}


def run(model, samples: list[ProbeSample], action_keys: list[str], variants: dict[int, dict[str, str]],
        seeds: list[int], denoising: list[int], set_denoising: Callable[[int], None],
        seed: Callable[[int], None]) -> dict:
    records, chunks, labels, latency = [], [], [], {d: [] for d in denoising}
    for steps in denoising:
        set_denoising(steps)
        for sample in samples:
            texts = {ORIGINAL: sample.text, **variants[sample.episode]}
            per_variant = {}
            for name, prompt in texts.items():
                draws = []
                try:
                    for s in seeds:
                        seed(s)
                        t0 = time.perf_counter()
                        action, _ = model.get_action(with_text(sample.groot, prompt))
                        latency[steps].append((time.perf_counter() - t0) * 1000)
                        draws.append(flatten(action, action_keys))
                except Exception as e:
                    records.append({"episode": sample.episode, "step": sample.step, "variant": name, "text": prompt,
                                    "denoising_steps": steps, "error": f"{type(e).__name__}: {e}"})
                    continue
                draws = np.stack(draws)
                per_variant[name] = draws
                chunks.append(draws)
                labels.append(f"{sample.episode}|{sample.step}|{name}|{steps}")
                horizon = min(draws.shape[1], sample.recorded.shape[0])
                error = np.abs(draws[:, :horizon] - sample.recorded[:horizon])
                records.append({"episode": sample.episode, "step": sample.step, "variant": name, "text": prompt,
                                "denoising_steps": steps, "error_vs_recorded": float(np.mean(error)),
                                "spread_across_seeds": float(np.mean(np.std(draws, axis=0)))})
            for record in records[-len(texts):]:
                if "error" not in record and ORIGINAL in per_variant:
                    record["moved_from_original"] = float(np.mean(np.abs(per_variant[record["variant"]]
                                                                         - per_variant[ORIGINAL])))
    summary = {}
    for name in dict.fromkeys(r["variant"] for r in records):
        rows = [r for r in records if r["variant"] == name and "error" not in r]
        summary[name] = {key: float(np.mean([r[key] for r in rows if key in r])) if any(key in r for r in rows)
                         else None for key in ("error_vs_recorded", "spread_across_seeds", "moved_from_original")}
        summary[name]["failed"] = sum(1 for r in records if r["variant"] == name and "error" in r)
    return {"variants": {str(e): v for e, v in variants.items()}, "seeds": seeds,
            "denoising_steps": denoising, "summary": summary,
            "latency_ms": {str(d): percentiles(v) for d, v in latency.items()}, "records": records,
            "chunks": np.stack(chunks), "labels": np.array(labels)}


def episode_samples(model, dataset: Path, episodes: list[int], points: int) -> tuple[list[ProbeSample], list[str]]:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    loader = LeRobotEpisodeLoader(dataset_path=str(dataset), modality_configs=model.get_modality_config())
    configs = loader.modality_configs
    action_keys = list(configs["action"].modality_keys)
    horizon = len(configs["action"].delta_indices)
    earliest = max(0, -min(min(configs[m].delta_indices) for m in ("video", "state")))
    samples = []
    for episode in episodes:
        trajectory = loader[episode]
        last = len(trajectory) - horizon
        for step in np.linspace(earliest, last, points).astype(int):
            data = extract_step_data(trajectory, int(step), configs, model.embodiment_tag)
            flat = {f"state.{k}": v for k, v in data.states.items()}
            flat.update({f"video.{k}": np.array(v) for k, v in data.images.items()})
            for key in configs["language"].modality_keys:
                flat[key] = data.text
            recorded = np.concatenate([np.asarray(data.actions[k]) for k in action_keys], axis=-1)
            samples.append(ProbeSample(episode, int(step), parse_observation_gr00t(flat, configs), recorded, data.text))
    return samples, action_keys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--embodiment-tag", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--variants", type=Path, required=True, help="JSON: episode -> {variant name: instruction}")
    p.add_argument("--points", type=int, default=8, help="evenly spaced time points per episode")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--denoising-steps", type=int, nargs="+", default=[4, 8])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    variants = {int(e): v for e, v in json.loads(a.variants.read_text()).items()}
    import torch
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    model = Gr00tPolicy(model_path=a.model, embodiment_tag=EmbodimentTag.resolve(a.embodiment_tag), device=a.device,
                        strict=True)

    def set_denoising(steps: int) -> None:
        model.model.action_head.num_inference_timesteps = steps

    def seed(s: int) -> None:
        torch.manual_seed(s)
        np.random.seed(s)

    samples, action_keys = episode_samples(model, a.dataset, sorted(variants), a.points)
    report = run(model, samples, action_keys, variants, a.seeds, a.denoising_steps, set_denoising, seed)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "actions.npz", chunks=report.pop("chunks"), labels=report.pop("labels"))
    (a.out / "probe.json").write_text(json.dumps({"model": a.model, "dataset": str(a.dataset), **report}, indent=2))
    print(json.dumps({"summary": report["summary"], "latency_ms": report["latency_ms"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
