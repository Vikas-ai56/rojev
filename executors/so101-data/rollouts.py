"""Whole-attempt done-check cases from ArmnetBench's single-arm pi0.5 rollouts (`config/armnetbench.txt`), each judged
by the benchmark's on-site operator.

  uv run python rollouts.py sample --out ../../data/s2-labels/armnetbench-sample.json [--target 300] [--seed 0]
      a seeded, label-balanced sample of attempts, spread evenly over the tasks, into a committed list
  uv run python rollouts.py build --sample ../../data/s2-labels/armnetbench-sample.json
                                  --out ../../data/s2-cases/armnetbench
      `cases.jsonl` + frames (JPEG) under --out, fetching only the video files that hold the sampled episodes

Labels: each episode's `success_class` in the dataset's meta/episodes; `successful` is truth `done`, `failure` is
`failed`. The suboptimal tier, the teleoperated demonstrations and the other policies are left out, as in FailBench's
ArmnetBench slice. Successful episodes are end-trimmed by the source at the operator's success time; failures run to
the end of the attempt.

Sampling: n = min(target / 2, successes, failures) attempts per label. First matched success/failure pairs per task,
handed out one at a time to the task with the fewest so far, so every task's labels balance where it has both; then
each label is topped up to n the same way from its remaining attempts. Each (task, label) cell is a seeded random draw.

Case contract (one JSON object per line), `cases.py`'s `status` kind asked about a whole attempt:
  id, kind (`status`), source (`armnetbench`), episode, frame (the attempt's last frame), goal and step (both the
  instruction), truth (`done` | `failed`), images {top, overhead, wrist: [the last frame's path]}
  images: relative to the case dir; the attempt's last frame alone, as the live done checker sees the newest frame
Views: `top` is the source's fixed front camera, the external view our SO-101 has (docs/target-system.md); in the
ring / block / tool cell it shows stacks and insertions the overhead camera loses under the arm. In the eye-drops /
cable cell it faces the wall with the basket and cable holder at its edge, so the overhead camera is kept as
`overhead` for a judge that is shown both; the wrist camera is `wrist`.
"""
import argparse
import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from recording import Recording, fetch_metadata, read_pin

PIN_FILE = Path(__file__).resolve().parents[2] / "config/armnetbench.txt"
SOURCE = "armnetbench"
POLICY = "pi0.5"
TRUTH = {"successful": "done", "failure": "failed"}
VIEWS = {"top": "front", "wrist": "wrist", "overhead": "top"}
JPEG_QUALITY = 95


@dataclass(frozen=True, order=True)
class Attempt:
    episode: int
    label: str
    instruction: str


def attempts(episode_rows: Iterable[dict]) -> list[Attempt]:
    return sorted(Attempt(int(row["episode_index"]), row["success_class"], row["tasks"][0])
                  for row in episode_rows if row["policy_type"] == POLICY and row["success_class"] in TRUTH)


def spread(counts: dict[str, int], capacity: dict[str, int], units: int) -> dict[str, int]:
    counts = dict(counts)
    for _ in range(units):
        open_keys = [key for key in sorted(counts) if counts[key] < capacity[key]]
        if not open_keys:
            break
        counts[min(open_keys, key=lambda key: counts[key])] += 1
    return counts


def quotas(pool: list[Attempt], per_label: int) -> dict[tuple[str, str], int]:
    available = Counter((a.instruction, a.label) for a in pool)
    tasks = sorted({a.instruction for a in pool})
    pairs = spread({task: 0 for task in tasks},
                   {task: min(available[task, label] for label in TRUTH) for task in tasks}, per_label)
    quota = {}
    for label in TRUTH:
        filled = spread(pairs, {task: available[task, label] for task in tasks}, per_label - sum(pairs.values()))
        quota.update({(task, label): n for task, n in filled.items()})
    return quota


def choose(pool: list[Attempt], target: int, seed: int) -> list[Attempt]:
    per_label = min(target // 2, *(sum(a.label == label for a in pool) for label in TRUTH))
    rng = random.Random(seed)
    chosen = []
    for (task, label), n in sorted(quotas(pool, per_label).items()):
        cell = sorted(a for a in pool if a.instruction == task and a.label == label)
        chosen += rng.sample(cell, n)
    return sorted(chosen)


def frame_path(episode: int, view: str, frame: int) -> str:
    return f"frames/ep{episode:04d}/{view}/f{frame:04d}.jpg"


def plan(chosen: list[dict], lengths: dict[int, int]) -> list[dict]:
    cases = []
    for attempt in chosen:
        episode, last = attempt["episode"], lengths[attempt["episode"]] - 1
        cases.append({"id": f"{SOURCE}-ep{episode:04d}", "kind": "status", "source": SOURCE, "episode": episode,
                      "frame": last, "goal": attempt["instruction"], "step": attempt["instruction"],
                      "truth": TRUTH[attempt["label"]],
                      "images": {view: [frame_path(episode, view, last)] for view in VIEWS}})
    return cases


def sample(target: int, seed: int, out: Path) -> list[Attempt]:
    meta = fetch_metadata(PIN_FILE)
    pool = attempts(meta.episodes)
    chosen = choose(pool, target, seed)
    available, taken = Counter(a.label for a in pool), Counter(a.label for a in chosen)
    note = (f"{taken['successful']} successful + {taken['failure']} failure from a pool of {available['successful']} "
            f"successful + {available['failure']} failure {POLICY} rollouts; target {target}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": " @ ".join(read_pin(PIN_FILE)), "policy": POLICY, "seed": seed,
                               "target": target, "note": note, "attempts": [asdict(a) for a in chosen]},
                              indent=1) + "\n")
    print(note)
    for (instruction, label), n in sorted(Counter((a.instruction, a.label) for a in chosen).items()):
        print(f"  {n:3d}  {label:10s}  {instruction}")
    return chosen


def build(sample_file: Path, out: Path) -> int:
    chosen = json.loads(sample_file.read_text())
    if chosen["dataset"] != " @ ".join(read_pin(PIN_FILE)):
        raise SystemExit(f"{sample_file} samples {chosen['dataset']}, but {PIN_FILE} pins another revision")
    episodes = [attempt["episode"] for attempt in chosen["attempts"]]
    rec = Recording.fetch(PIN_FILE, tuple(VIEWS.values()), episodes=episodes)
    cases = plan(chosen["attempts"], {episode: len(rec.states(episode)) for episode in episodes})
    for case in cases:
        for view, camera in VIEWS.items():
            path = out / case["images"][view][0]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rec.frames(case["episode"], [case["frame"]], camera)[0]).save(path, quality=JPEG_QUALITY)
    with (out / "cases.jsonl").open("w") as f:
        f.writelines(json.dumps(case) + "\n" for case in cases)
    (out / "source.json").write_text(json.dumps({"dataset": chosen["dataset"], "sample": sample_file.name,
                                                 "views": VIEWS, "cases": len(cases)}, indent=1))
    return len(cases)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--target", type=int, default=300)
    a.add_argument("--seed", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--sample", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.what == "sample":
        sample(args.target, args.seed, args.out)
    else:
        print(f"{build(args.sample, args.out)} cases in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
