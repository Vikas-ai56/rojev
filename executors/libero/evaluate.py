"""Score an S1 policy on LIBERO through the whole rojev runtime: one supervised `rojev.app.Session` per task episode.

  uv run python evaluate.py --suite libero_goal --tasks 0 1 --episodes 2 --seed 1000 \
      --service ~/.config/rojev/services/<host>.json --entry lerobot-policy --out results/libero/goal \
      [--video] [--config run.yaml] [--codec raw]
  uv run python evaluate.py --policy zero --suite libero_goal --tasks 0 --episodes 1 --out <dir>

Per task and episode: a `LiberoBackend` (episode e = lerobot-eval's e-th rollout of the task: init state e, seed
`--seed` + e, the same seeds for every task), the policy reset first (lerobot-eval resets its policy before every
rollout), `rojev.eval.NativePlanner` (one command without a subtask: the whole instruction goes to S1, no S2 steps), and
one session run for the task's step budget in robot time (LeRobot's max steps × dt), stopped as soon as the task
succeeds. If S1 delivered no chunk and its first BROKEN_S1_ERRORS requests failed, the episode stops and the whole
evaluation exits (code 2): a run against a broken service would only score that breakage as task failures. Settings: the `runtime` section of `--config` (`rojev.app.load_config`), else `SessionConfig()` (lockstep:
model computation takes no robot time, how benchmarks are scored).
Policies: `remote` = a `RemotePolicy` for the `--entry` of the `--service` JSON, frames sent with `--codec` (default
raw: lossless, as lerobot-eval hands the rendered frames to its policy); `zero` = holds the arm still with the gripper
open, for plumbing runs without a GPU.
Output under `--out`: one session directory per episode (`task<NN>/episode<NN>/`: the session's records, `view.mp4` with
`--video`) and `results.json` (settings, one row per episode, success per task and for the suite), rewritten after
every episode so an interrupted run keeps what it scored; the per-task table is printed at the end.
"""
import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rojev.app import Session, SessionConfig, load_config
from rojev.eval import NativePlanner
from rojev.system1 import ActionChunk, Policy, RemotePolicy, wire

from backend import LiberoBackend
from embodiment import EMBODIMENT
from suites import load_suite

HOLD_CHUNK_S = 1.0


class HoldingPolicy(Policy):
    def __init__(self, action: np.ndarray, horizon: int):
        self.chunk = np.tile(np.asarray(action, dtype=np.float32), (horizon, 1))

    def act(self, obs, subtask: str, timeout_s: float | None = None, continuation=None) -> ActionChunk:
        return ActionChunk(self.chunk.copy())

    def reset(self) -> None:
        pass


@dataclass(frozen=True)
class EpisodeResult:
    task: int
    episode: int
    seed: int
    instruction: str
    success: bool
    success_step: int | None
    steps: int
    max_steps: int
    sim_s: float
    wall_s: float
    ended: str
    out: str
    s1_accepted: int = 0
    s1_errors: int = 0

    @property
    def s1_broken(self) -> bool:
        return self.s1_accepted == 0 and self.s1_errors >= BROKEN_S1_ERRORS


BROKEN_S1_ERRORS = 5


def episode_dir(out: Path, task: int, episode: int) -> Path:
    return out / f"task{task:02d}" / f"episode{episode:02d}"


def run_episode(suite: str, task: int, episode: int, seed: int, policy: Policy, config: SessionConfig, out: Path,
                video: bool, open_backend: Callable[..., LiberoBackend] = LiberoBackend,
                session_type: type[Session] = Session) -> EpisodeResult:
    policy.reset()
    backend = open_backend(suite, task, episode, seed)
    try:
        if backend.control_hz != EMBODIMENT.spec.control_hz:
            raise ValueError(f"{suite} steps at {backend.control_hz} Hz, the embodiment at "
                             f"{EMBODIMENT.spec.control_hz} Hz")
        session = session_type(EMBODIMENT, backend, policy, NativePlanner(), backend.instruction, config, out,
                               video=video)
        stats = session.runner.stats

        def done() -> bool:
            return backend.success or (stats.accepted == 0 and stats.errors >= BROKEN_S1_ERRORS)

        run = session.run(backend.max_steps * EMBODIMENT.spec.dt, stop_when=done)
    finally:
        backend.close()
    return EpisodeResult(task, episode, seed, backend.instruction, backend.success, backend.success_step,
                         backend.steps, backend.max_steps, run.sim_s, run.wall_s, run.ended, str(out),
                         stats.accepted, stats.errors)


def summarize(results: list[EpisodeResult]) -> dict:
    by_task: dict[int, list[EpisodeResult]] = {}
    for result in results:
        by_task.setdefault(result.task, []).append(result)
    tasks = [{"task": task, "instruction": rows[0].instruction, "episodes": len(rows),
              "successes": sum(r.success for r in rows), "success_rate": round(sum(r.success for r in rows) / len(rows), 4)}
             for task, rows in sorted(by_task.items())]
    successes = sum(r.success for r in results)
    return {"tasks": tasks, "episodes": len(results), "successes": successes,
            "success_rate": round(successes / len(results), 4) if results else None}


def write_results(path: Path, settings: dict, results: list[EpisodeResult]) -> None:
    document = {**settings, **summarize(results), "runs": [asdict(r) for r in results]}
    path.write_text(json.dumps(document, indent=1))


def render_table(suite: str, summary: dict) -> str:
    lines = [f"{'task':>4}  {'success':>7}  {'rate':>5}  instruction"]
    lines += [f"{row['task']:>4}  {row['successes']:>3}/{row['episodes']:<3}  {row['success_rate']:>5.2f}  "
              f"{row['instruction']}" for row in summary["tasks"]]
    if summary["episodes"]:
        lines.append(f"{suite}: {summary['successes']}/{summary['episodes']} = {summary['success_rate']:.4f}")
    return "\n".join(lines)


def make_policy(args) -> Policy:
    if args.policy == "zero":
        return HoldingPolicy(EMBODIMENT.start, round(HOLD_CHUNK_S * EMBODIMENT.spec.control_hz))
    services = json.loads(args.service.expanduser().read_text())
    if args.entry not in services:
        raise SystemExit(f"{args.service} has no entry {args.entry!r}; entries: {sorted(services)}")
    return RemotePolicy(**{**services[args.entry], "codec": args.codec})


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, help="LIBERO suite, e.g. libero_goal (see suites.py)")
    parser.add_argument("--tasks", type=int, nargs="+", help="task indices (default: every task of the suite)")
    parser.add_argument("--episodes", type=int, default=1, help="episodes per task")
    parser.add_argument("--seed", type=int, default=1000, help="seed of episode 0; episode e uses seed + e")
    parser.add_argument("--policy", choices=("remote", "zero"), default="remote")
    parser.add_argument("--service", type=Path, help="service JSON ({entry: {url, token}}) for --policy remote")
    parser.add_argument("--entry", default="lerobot-policy", help="the S1 service's entry in --service")
    parser.add_argument("--codec", choices=wire.CODECS, default="raw", help="frame encoding sent to the service")
    parser.add_argument("--config", type=Path, help="YAML whose runtime section is the SessionConfig")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video", action="store_true", help="write view.mp4 per episode")
    args = parser.parse_args(argv)
    if args.policy == "remote" and args.service is None:
        parser.error("--policy remote needs --service")
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)[0] if args.config else SessionConfig()
    tasks = args.tasks if args.tasks is not None else list(range(load_suite(args.suite).n_tasks))
    policy = make_policy(args)
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {"suite": args.suite, "policy": args.policy, "service_entry": args.entry if args.policy == "remote"
                else None, "codec": args.codec if args.policy == "remote" else None, "seed": args.seed,
                "tasks_run": tasks, "episodes_per_task": args.episodes, "embodiment": EMBODIMENT.name,
                "runtime": asdict(config)}
    results: list[EpisodeResult] = []
    for task in tasks:
        for episode in range(args.episodes):
            result = run_episode(args.suite, task, episode, args.seed + episode, policy, config,
                                 episode_dir(args.out, task, episode), args.video)
            if result.s1_broken:
                print(f"S1 never answered task {task} episode {episode} ({result.s1_errors} failed requests, "
                      f"see {result.out}/s1_requests.jsonl): stopping, this run would only measure a broken "
                      f"service", flush=True)
                return 2
            results.append(result)
            write_results(args.out / "results.json", settings, results)
            print(f"task {task:2d} episode {episode:2d}: {'success' if result.success else 'failure'} "
                  f"after {result.steps}/{result.max_steps} steps ({result.ended}, {result.wall_s:.1f} s wall)",
                  flush=True)
    print(render_table(args.suite, summarize(results)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
