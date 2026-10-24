"""MJev vs OneJev on "is this step done?": frames whose answer the simulator knows (decides MJev's scope, 2026-10-22).

  build   (Mac, no GPU) for each task and seed, put the kitchen into a chain of states (initial → object placed → door
          closed → appliance on), let physics settle, render the three cameras, and label every checked step of the
          task with the task's own completion test (tasks.py). A state whose intended predicate does not hold after
          settling is dropped and counted, never relabelled. "pick up" steps are not checked: holding needs a real
          grasp, which a state cannot be set to. Each frame also stores the simulator's pixel masks per camera
          (`masks.npz`, key `<camera>:<role>`, roles object / container / gripper) as truth for object-mask checks
          (sam_check.py).
  score   (GPU services) ask both verifiers the identical question per (frame, step) (`--postconditions`: the step's
          yes / no postcondition instead of "is this step done?") and report, per verifier and per
          step: balanced accuracy and false-"done" rate of the top answer, recall on done frames, the same at the
          step watcher's threshold (P(done) ≥ 0.7), and latency.

  uv run python done_check_ab.py build --out results/done-check/<stamp> [--seeds 1000-1009]
  uv run python done_check_ab.py score results/done-check/<stamp> --service ~/.config/rojev/services/lambda.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from harness.tracking import truth_masks
from realtime import START, RoboCasaBackend
from rojev.system2.models import RemoteMJev, onejev
from rojev.system2.verify import MJevVerifier, OneJevVerifier
from tasks import TASKS
from workflow import CAMS, service_addresses

SETTLE_STEPS = 40
THRESHOLD = 0.7


def close_door(k, joints: list[str]) -> None:
    """Every door joint to its closed end (the end of its range nearer 0)."""
    for name in joints:
        lo, hi = k.sim.model.jnt_range[k.sim.model.joint_name2id(name)]
        k.sim.data.set_joint_qpos(name, lo if abs(lo) <= abs(hi) else hi)
    k.sim.forward()


def place_inside(k, obj: str, regions: dict) -> None:
    """The object's free joint to the centre of the first interior region, just above its floor."""
    p0, px, py, pz = next(iter(regions.values()))
    target = p0 + (px - p0) / 2 + (py - p0) / 2 + np.array([0.0, 0.0, 0.03])
    joint = k.objects[obj].joints[0]
    qpos = k.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = target
    k.sim.data.set_joint_qpos(joint, qpos)
    k.sim.data.set_joint_qvel(joint, np.zeros(6))
    k.sim.forward()


def microwave_states(k) -> list[tuple[str, callable]]:
    return [("placed", lambda k: place_inside(k, "meat", k.microwave.get_int_sites(relative=False))),
            ("door closed", lambda k: close_door(k, k.microwave.door_joint_names)),
            ("turned on", lambda k: setattr(k.microwave, "_turned_on", True))]


def freezer_states(k) -> list[tuple[str, callable]]:
    shelf = {n: v for n, v in k.fridge.get_int_sites(relative=False).items() if n == "freezer_shelf1"}
    return [("placed", lambda k: place_inside(k, "refrigerated_item", shelf)),
            ("freezer door closed", lambda k: close_door(k, k.fridge._freezer_door_joint_names)),
            ("fridge door closed", lambda k: close_door(k, k.fridge._fridge_door_joint_names))]


STATES = {"MicrowaveThawingFridge": microwave_states, "MoveFridgeToFreezer": freezer_states}


def settle(backend: RoboCasaBackend) -> None:
    for _ in range(SETTLE_STEPS):
        backend.step(START)


def build(out: Path, seeds: list[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows, dropped = [], []
    for task, make_states in STATES.items():
        for seed in seeds:
            backend = RoboCasaBackend(task, seed, horizon=4000)
            k = backend.kitchen
            steps = [s for s in TASKS[task].steps(k) if not s.subtask.startswith("pick up")]
            chain = [("initial", None)] + make_states(k)
            for index, (state, apply) in enumerate(chain):
                if apply is not None:
                    backend.read(apply)
                settle(backend)
                truth = {s.subtask: backend.truth(s.done) for s in steps}
                intended = steps[index - 1].subtask if index else None
                if intended is not None and not truth[intended]:
                    dropped.append({"task": task, "seed": seed, "state": state, "step": intended})
                    break
                frame = f"{task}-s{seed}-{index}"
                (out / frame).mkdir(exist_ok=True)
                for cam in CAMS:
                    Image.fromarray(backend.raw["pixels"][cam]).save(out / frame / f"{cam}.png")
                size = backend.raw["pixels"][CAMS[0]].shape[:2]
                np.savez_compressed(out / frame / "masks.npz", **backend.read(lambda k: truth_masks(k, task, list(CAMS), size)))
                rows += [{"frame": frame, "task": task, "seed": seed, "state": state, "step": step, "truth": t}
                         for step, t in truth.items()]
            backend.env.close()
            print(f"[build] {task} seed {seed}: {sum(r['seed'] == seed and r['task'] == task for r in rows)} labels",
                  flush=True)
    (out / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "dropped.json").write_text(json.dumps(dropped, indent=1))
    print(json.dumps({"labels": len(rows), "done": sum(r["truth"] for r in rows), "dropped_states": len(dropped)}))


def metrics(rows: list[dict]) -> dict:
    """Top answer and threshold decisions vs truth."""
    def rates(pred):
        pos = [r for r in rows if r["truth"]]
        neg = [r for r in rows if not r["truth"]]
        recall = sum(pred(r) for r in pos) / len(pos) if pos else None
        false_done = sum(pred(r) for r in neg) / len(neg) if neg else None
        balanced = None if recall is None or false_done is None else (recall + 1 - false_done) / 2
        return {"balanced_accuracy": balanced, "recall": recall, "false_done": false_done, "pos": len(pos),
                "neg": len(neg)}

    return {"top_answer": rates(lambda r: max(r["states"], key=r["states"].get) == "done"),
            f"p_done>={THRESHOLD}": rates(lambda r: r["states"]["done"] >= THRESHOLD),
            "check_ms_p50": float(np.percentile([r["ms"] for r in rows], 50))}


def score(folder: Path, service: Path, postconditions: bool = False) -> None:
    labels = [json.loads(line) for line in (folder / "labels.jsonl").read_text().splitlines()]
    addresses = service_addresses(str(service), "serve")
    verifiers = [MJevVerifier(RemoteMJev(**addresses["mjev"]), postconditions=postconditions),
                 OneJevVerifier(onejev.KeptClient(addresses["onejev"]["url"]), postconditions=postconditions)]
    suffix = "_post" if postconditions else ""
    frames = defaultdict(list)
    for r in labels:
        frames[r["frame"]].append(r)
    results = []
    for frame, rows in frames.items():
        images = [np.asarray(Image.open(folder / frame / f"{cam}.png").convert("RGB")) for cam in CAMS]
        for v in verifiers:
            w0 = time.perf_counter()
            answers = v.check(images, [r["step"] for r in rows])
            ms = (time.perf_counter() - w0) * 1000 / len(rows)
            results += [{**r, "verifier": v.name, "states": answers[r["step"]], "ms": round(ms, 1)} for r in rows]
    (folder / f"scores{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    report = {}
    for v in verifiers:
        mine = [r for r in results if r["verifier"] == v.name]
        report[v.name] = {"all": metrics(mine),
                          "per_step_type": {kind: metrics([r for r in mine if kind in r["step"]])
                                            for kind in ("place", "close", "press")
                                            if any(kind in r["step"] for r in mine)}}
    (folder / f"report{suffix}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({v: report[v]["all"] for v in report}, indent=1))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seeds", default="1000-1009")
    s = sub.add_parser("score")
    s.add_argument("folder", type=Path)
    s.add_argument("--service", type=Path, required=True)
    s.add_argument("--postconditions", action="store_true",
                   help="ask each step's postcondition (yes / no fact) instead of the abstract step question")
    a = p.parse_args()
    if a.command == "build":
        lo, _, hi = a.seeds.partition("-")
        build(a.out, list(range(int(lo), int(hi or lo) + 1)))
    else:
        score(a.folder, a.service.expanduser(), a.postconditions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
