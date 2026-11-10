"""Parity and latency of the SO-101 GR00T adapter on the GPU box (GPU-session step `so101_parity`).

Observations come from a LeRobot SO-101 episode read with Isaac-GR00T's own `LeRobotEpisodeLoader` and
`extract_step_data`, exactly as its `open_loop_eval.py` reads them. Each one goes two ways with the same torch seed:
(a) straight to `Gr00tPolicy.get_action`, shaped by GR00T's own `parse_observation_gr00t`; (b) through rojev, as an
`Observation` in joint radians to `GrootPolicy` (our binding) behind `PolicyService` over the loopback HTTP
transport. GR00T's sampling noise is the same for both, so any difference beyond float rounding from the units round
trip is ours. Reported per step: max |a − b| in the dataset's units; overall: p50 / p95 / max of the direct call,
of the adapter in process, and of the round trip through our service. With `--swap-text`, the direct call is repeated
with that instruction under the same seed (a language probe: how far the actions move when only the text changes).
Writes `parity.json` and `actions.npz` (every step's direct, adapter and swapped-text chunks) to `--out`.

`compare` holds the logic and takes the model, the observations and the clock as arguments, so it runs on the Mac
with stand-ins; `main` loads the real checkpoint and dataset with `gr00t` on the box.

  uv run python parity.py --model <checkpoint> --dataset <lerobot dataset> --out <dir>
"""
import argparse
import json
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from groot_io import Binding
from rojev.protocol import Observation
from rojev.robot.so101 import LeRobotUnits, lerobot_units
from rojev.system1 import PolicyService, RemotePolicy
from rojev.transport import Endpoint, HttpServer
from serve import MAX_BODY, GrootPolicy

TOKEN = "parity"


@dataclass(frozen=True)
class Sample:
    step: int
    groot: dict
    observation: Observation


def to_observation(binding: Binding, images: dict[str, list], states: dict[str, np.ndarray], text: str,
                   units: LeRobotUnits) -> Observation:
    values = np.empty(binding.joints)
    for key, (start, end) in binding.state.items():
        values[start:end] = np.asarray(states[key])[-1]
    frames = {camera: np.asarray(images[key][-1], dtype=np.uint8) for key, camera in binding.cameras.items()}
    return Observation(frames, units.from_lerobot(values).astype(np.float32), text)


def percentiles(ms: list[float]) -> dict:
    return {"n": len(ms), "p50": round(float(np.percentile(ms, 50)), 1), "p95": round(float(np.percentile(ms, 95)), 1),
            "max": round(float(np.max(ms)), 1)} if ms else {"n": 0}


def with_text(groot: dict, text: str) -> dict:
    return {**groot, "language": {key: [[text]] for key in groot["language"]}}


def compare(model, units: LeRobotUnits, samples: Iterable[Sample], seed: Callable[[int], None],
            swap_text: str | None = None) -> dict:
    policy = GrootPolicy(model, units)
    binding = policy.binding
    server = HttpServer(Endpoint(PolicyService(policy, {"policy": "parity"}, name="groot").handle, TOKEN, MAX_BODY),
                        "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    remote = RemotePolicy(f"http://127.0.0.1:{server.server_port}", TOKEN, codec="raw")
    steps, direct_ms, adapter_ms, service_ms = [], [], [], []
    chunks: dict[str, list[np.ndarray]] = {"direct": [], "adapter": [], "swapped": []}
    try:
        for i, sample in enumerate(samples):
            seed(i)
            t0 = time.perf_counter()
            direct, _ = model.get_action(sample.groot)
            direct_ms.append((time.perf_counter() - t0) * 1000)
            expected = np.concatenate([np.asarray(direct[key])[0] for key in binding.action], axis=-1)
            seed(i)
            t0 = time.perf_counter()
            ours = policy.act(sample.observation, sample.observation.task).actions
            adapter_ms.append((time.perf_counter() - t0) * 1000)
            seed(i)
            t0 = time.perf_counter()
            served = remote.act(sample.observation, sample.observation.task).actions
            service_ms.append((time.perf_counter() - t0) * 1000)
            record = {"step": sample.step,
                      "max_abs_diff": float(np.max(np.abs(units.to_lerobot(ours) - expected))),
                      "service_max_abs_diff": float(np.max(np.abs(units.to_lerobot(served) - expected)))}
            chunks["direct"].append(expected)
            chunks["adapter"].append(units.to_lerobot(ours))
            if swap_text is not None:
                seed(i)
                swapped, _ = model.get_action(with_text(sample.groot, swap_text))
                moved = np.concatenate([np.asarray(swapped[key])[0] for key in binding.action], axis=-1)
                chunks["swapped"].append(moved)
                record["swap_mean_abs_diff"] = float(np.mean(np.abs(moved - expected)))
                record["swap_max_abs_diff"] = float(np.max(np.abs(moved - expected)))
            steps.append(record)
    finally:
        server.shutdown()
    worst = max((s["max_abs_diff"] for s in steps), default=None)
    probe = None if swap_text is None else {
        "swap_text": swap_text, "mean_abs_diff": float(np.mean([s["swap_mean_abs_diff"] for s in steps])),
        "max_abs_diff": float(np.max([s["swap_max_abs_diff"] for s in steps]))}
    return {"steps": steps, "worst_max_abs_diff": worst, "language_key": binding.language, "language_probe": probe,
            "direct_ms": percentiles(direct_ms), "adapter_ms": percentiles(adapter_ms),
            "service_ms": percentiles(service_ms),
            "chunks": {name: np.stack(values) for name, values in chunks.items() if values}}


def episode_samples(model, binding: Binding, dataset: Path, episode: int, count: int, stride: int,
                    units: LeRobotUnits) -> list[Sample]:
    from copy import deepcopy

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    modality = model.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(dataset), modality_configs=modality)
    trajectory = loader[episode]
    inputs = deepcopy(loader.modality_configs)
    inputs.pop("action")
    samples = []
    for step in range(0, min(len(trajectory), count * stride), stride):
        data = extract_step_data(trajectory, step, inputs, model.embodiment_tag)
        flat = {f"state.{k}": v for k, v in data.states.items()}
        flat.update({f"video.{k}": np.array(v) for k, v in data.images.items()})
        for key in loader.modality_configs["language"].modality_keys:
            flat[key] = data.text
        samples.append(Sample(step, parse_observation_gr00t(flat, loader.modality_configs),
                              to_observation(binding, data.images, data.states, data.text, units)))
    return samples


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="fine-tuned GR00T N1.7 SO-101 checkpoint directory")
    p.add_argument("--dataset", type=Path, required=True, help="LeRobot SO-101 dataset (front + wrist, LeRobot units)")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--count", type=int, default=20, help="observations compared")
    p.add_argument("--stride", type=int, default=16, help="frames between compared observations")
    p.add_argument("--swap-text", help="language probe: repeat each direct call with this instruction, same seed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import torch

    from serve import load

    def seed(i: int) -> None:
        torch.manual_seed(i)
        np.random.seed(i)

    units = lerobot_units()
    model = load(a.model, a.device, strict=True)
    binding = GrootPolicy(model, units).binding
    samples = episode_samples(model, binding, a.dataset, a.episode, a.count, a.stride, units)
    report = compare(model, units, samples, seed, a.swap_text)
    chunks = report.pop("chunks")
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "actions.npz", **chunks)
    (a.out / "parity.json").write_text(json.dumps({"model": a.model, "dataset": str(a.dataset), **report}, indent=2))
    print(json.dumps({k: report[k] for k in ("worst_max_abs_diff", "language_probe", "direct_ms", "adapter_ms",
                                             "service_ms")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
