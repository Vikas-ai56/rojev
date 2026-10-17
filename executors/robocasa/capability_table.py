"""G0-C capability table: frozen executor alone, N episodes per atomic task, one `lerobot-eval` job per task.

Each task's row is written as soon as it finishes, so the run can be stopped at any time (Ctrl-C) and keep everything
completed so far. No new task starts once the time budget would be exceeded. System free memory and swap are sampled
during every task.

  uv run python capability_table.py --episodes 5 --batch 5 --budget-min 30
"""
import argparse
import json
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CI_TASKS = ["CloseFridge", "OpenCabinet", "OpenDrawer", "TurnOnMicrowave", "TurnOffStove", "CloseToasterOvenDoor",
            "SlideDishwasherRack", "TurnOnSinkFaucet", "NavigateKitchen", "TurnOnElectricKettle"]  # LeRobot CI smoke set


def checkpoint_rename_map(policy: str) -> dict[str, str]:
    """Camera-key mapping the checkpoint was trained with, read from its own preprocessor config.

    `lerobot-eval` replaces the stored mapping with `--rename_map` (default: empty), so it must be passed explicitly.
    LeRobot's RoboCasa docs/CI map eye_in_hand->camera2 and right->camera3; smolvla_robocasa was trained with
    right->camera2 and eye_in_hand->camera3, so a hardcoded doc mapping silently swaps two cameras.
    """
    from huggingface_hub import hf_hub_download
    cfg = json.loads(Path(hf_hub_download(policy, "policy_preprocessor.json")).read_text())
    maps = [s["config"]["rename_map"] for s in cfg["steps"] if "rename_map" in (s.get("config") or {})]
    if len(maps) != 1 or not maps[0]:
        raise ValueError(f"{policy}: expected exactly one non-empty rename_map in policy_preprocessor.json, got {maps}")
    return maps[0]


def system_memory_gb() -> tuple[float, float]:
    """(free + inactive GB, swap used GB) from vm_stat / sysctl."""
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    pages = lambda name: int(re.search(rf"{name}:\s+(\d+)", vm).group(1))
    avail = (pages("Pages free") + pages("Pages inactive")) * page / 1e9
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = re.search(r"used = ([\d.]+)M", swap)
    return avail, (float(used.group(1)) / 1e3 if used else 0.0)


class MemorySampler(threading.Thread):
    def __init__(self, every_s: float = 2.0):
        super().__init__(daemon=True)
        self.every_s, self.min_avail, self.max_swap, self._stop = every_s, float("inf"), 0.0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            a, s = system_memory_gb()
            self.min_avail, self.max_swap = min(self.min_avail, a), max(self.max_swap, s)
            self._stop.wait(self.every_s)

    def stop(self):
        self._stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", default=",".join(CI_TASKS))
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--batch", type=int, default=5, help="parallel environments per task")
    p.add_argument("--sync-envs", action="store_true", help="step environments in one process instead of subprocesses")
    p.add_argument("--policy", default="lerobot/smolvla_robocasa")
    p.add_argument("--device", default="mps")
    p.add_argument("--split", default="target",
                   help="RoboCasa scene split; smolvla_robocasa trained on the target split (single-task eval defaults to 'all')")
    p.add_argument("--budget-min", type=float, default=30.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out = Path(args.out or REPO / f"results/runs/g0c_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    rename = checkpoint_rename_map(args.policy)
    (out / "rename_map.json").write_text(json.dumps(rename, indent=2))
    (out / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"camera mapping from checkpoint: {rename}", flush=True)
    rows, t0, current = [], time.perf_counter(), {"proc": None}
    stop = {"flag": False}

    def on_sigint(*_):
        stop["flag"] = True
        if current["proc"] is not None:
            current["proc"].terminate()
    signal.signal(signal.SIGINT, on_sigint)

    def write():
        (out / "capability.json").write_text(json.dumps(rows, indent=2))
        lines = ["| Task | Success | Rate | Min free RAM GB | Max swap GB | Wall s |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['task']} | {r['n_success']}/{r['n_episodes']} | {r['rate']} | {r['min_avail_gb']} | "
                  f"{r['max_swap_gb']} | {r['wall_s']} |" for r in rows]
        (out / "capability.md").write_text("\n".join(lines) + "\n")

    for task in args.tasks.split(","):
        elapsed = time.perf_counter() - t0
        per_task = max((r["wall_s"] for r in rows), default=0.0)
        if stop["flag"] or (rows and elapsed + per_task > args.budget_min * 60):
            print(f"stopping before {task}: elapsed {elapsed / 60:.1f} min, budget {args.budget_min} min", flush=True)
            break
        task_dir = out / task
        cmd = [str(Path(__file__).parent / ".venv/bin/lerobot-eval"), f"--policy.path={args.policy}",
               "--policy.load_vlm_weights=false", f"--policy.device={args.device}", "--env.type=robocasa",
               f"--env.task={task}", f"--env.split={args.split}", f"--eval.batch_size={args.batch}", f"--eval.n_episodes={args.episodes}",
               f"--eval.use_async_envs={'false' if args.sync_envs else 'true'}", f"--output_dir={task_dir}",
               f"--rename_map={json.dumps(rename)}"]
        sampler = MemorySampler()
        sampler.start()
        ts = time.perf_counter()
        with open(out / f"{task}.log", "w") as log:
            current["proc"] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            code = current["proc"].wait()
        current["proc"] = None
        sampler.stop()
        info_path = task_dir / "eval_info.json"
        row = {"task": task, "exit": code, "wall_s": round(time.perf_counter() - ts, 1),
               "min_avail_gb": round(sampler.min_avail, 2), "max_swap_gb": round(sampler.max_swap, 2)}
        if code == 0 and info_path.exists():
            o = json.loads(info_path.read_text())["overall"]
            row |= {"n_success": int(o.get("n_success", 0)), "n_episodes": int(o.get("n_episodes", args.episodes)),
                    "rate": f"{o.get('pc_success', 0.0):.0f}%", "ci95": o.get("pc_success_ci95")}
        else:
            row |= {"n_success": 0, "n_episodes": 0, "rate": "ERROR (see log)", "ci95": None}
        rows.append(row)
        write()
        print(f"{task:22s} {row['rate']:>15s}  ({row.get('n_success')}/{row.get('n_episodes')})  wall {row['wall_s']}s  "
              f"min free RAM {row['min_avail_gb']} GB  swap {row['max_swap_gb']} GB", flush=True)
    write()
    print(f"done: {len(rows)} task(s) in {(time.perf_counter() - t0) / 60:.1f} min -> {out}/capability.md", flush=True)


if __name__ == "__main__":
    main()
