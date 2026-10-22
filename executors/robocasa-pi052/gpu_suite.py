"""The v1 GPU test suite, run end to end from the Mac against the services on the GPU box (one command, sequential).

Steps run one after another on purpose: π0.5², OneJev and MJev share one GPU and π0.5²'s service serves one request at
a time, so parallel runs would measure each other's load.

  1 baseline     π0.5² alone (native planner, fixed commitment) on each authored task: real-time factor, starvation,
                 S1 round trip, task success, π0.5²'s own subtask switches (step events)
  2 k_samples    K = 1 / 3 / 5 action samples in one request on a real RoboCasa frame: latency, draw diversity (mean
                 pairwise action distance vs the K = 1 call-to-call distance), GPU memory
  3 jev          OneJev latency per interaction decision (instruction and deviation snapshots), then live Jev runs with
                 π0.5² (wait / go on, stop, ambiguous then answer); the interaction deadline is set from the measured
                 latency; every model decision is saved as a calibration case
  4 mjev         MJev end-state checks against simulator truth: π0.5² driven through the authored steps (oracle arm)
                 with the step watcher on MJev; per check P(done) vs truth
  5 t8a_gate     commitment ablation with π0.5² (fixed, phase, consensus K = 3 / 5, consensus + phase), then the v1 gate
                 run: π0.5² + Jev + step events + a change of mind, with video

  uv run python gpu_suite.py --service ~/.config/rojev/services/lambda.json [--only baseline jev] [--quick]
Writes results/gpu-suite/<stamp>/{<step>/…, suite.json, suite.md}; a step that fails is recorded and the suite goes on.
"""
import argparse
import itertools
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from qev import Client

from realtime import RoboCasaBackend
from rojev.protocol import Capture, Deviation, ObservationSnapshot, Operation, OpKind
from rojev.system1 import RemotePolicy
from rojev.system2.jev import JevInteraction, OneJevChooser
from tasks import TASKS
from workflow import CAMS, service_addresses

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SEED = 1000
MAIN_TASK = "MicrowaveThawingFridge"


def realtime(out: Path, service: Path, args: list[str], seconds: float) -> dict:
    """One `realtime.py` run in its own process; returns its summary (or the failure)."""
    cmd = [sys.executable, str(HERE / "realtime.py"), "--policy", "remote", "--service", str(service), "--seed",
           str(SEED), "--seconds", str(seconds), "--out", str(out), *args]
    w0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    (out.parent / f"{out.name}.log").write_text(proc.stdout[-20000:] + "\n--- stderr ---\n" + proc.stderr[-20000:])
    summary = out / "summary.json"
    if proc.returncode != 0 or not summary.exists():
        return {"ok": False, "returncode": proc.returncode, "error": proc.stderr.strip().splitlines()[-3:]}
    s = json.loads(summary.read_text())
    requests = [json.loads(line) for line in (out / "s1_requests.jsonl").read_text().splitlines()]
    walls = sorted(r["wall_ms"] for r in requests[1:] if "wall_ms" in r)
    return {"ok": True, "wall_s": round(time.perf_counter() - w0, 1), "realtime_factor": s["realtime_factor"],
            "starved_ticks": s["starved_ticks"], "violations": s["violations"], "task_success": s["task_success"],
            "success_t": s["success_t"], "s1_requests": s["s1_requests"], "s1_errors": s["s1_errors"],
            "s1_wall_ms_p50": pct(walls, 50), "s1_wall_ms_p90": pct(walls, 90),
            "first_request_ms": requests[0].get("wall_ms") if requests else None,
            "s1_floor_last": s["s1_floor_last"], "planner_steps": s.get("s1_planner_steps"),
            "keyframes": [k["subtask"] for k in s["keyframes"]], "commands": len(s["commands"]),
            "commit_reasons": s["s1_commit_reasons"], "said": [u["template"] for u in s.get("said", [])],
            "holds": s.get("holds"), "interaction_lane": s.get("interaction_lane"), "verify": s.get("verify")}


def pct(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 1) if values else None


def gpu_memory(service: Path) -> str:
    """GPU memory on the box the service JSON describes (its file name is the ssh host, as gpu_setup.sh writes it)."""
    out = subprocess.run(["ssh", "-o", "BatchMode=yes", service.stem, "nvidia-smi --query-gpu=memory.used,"
                          "memory.total --format=csv,noheader"], capture_output=True, text=True, timeout=30)
    return out.stdout.strip()


def frame(task: str) -> Capture:
    backend = RoboCasaBackend(task, SEED, horizon=100)
    obs = backend.observe()
    backend.env.close()
    return Capture(1, 0.0, obs)


def baseline(out: Path, service: Path, quick: bool) -> dict:
    seconds = 30 if quick else 60
    tasks = [MAIN_TASK] if quick else list(TASKS)
    return {task: realtime(out / task, service, ["--task", task, "--arm", "native"], seconds) for task in tasks}


def k_samples(out: Path, service: Path, quick: bool) -> dict:
    capture = frame(MAIN_TASK)
    policy = RemotePolicy(**service_addresses(str(service), "serve")["pi052"])
    obs, repeats = capture.observation, (3 if quick else 8)
    policy.act(obs, "")
    results = {"gpu_memory_idle": gpu_memory(service)}
    single = [policy.act(obs, "pick up the steak").actions for _ in range(repeats)]
    baseline_spread = mean_pairwise(single)
    for k in (1, 3, 5):
        times, spreads = [], []
        try:
            for _ in range(repeats):
                w0 = time.perf_counter()
                chunks = policy.act_samples(obs, "pick up the steak", k)
                times.append((time.perf_counter() - w0) * 1000)
                if k > 1:
                    spreads.append(mean_pairwise([c.actions for c in chunks]))
            results[f"k{k}"] = {"ok": True, "round_trip_ms_p50": pct(times, 50), "round_trip_ms_p90": pct(times, 90),
                                "draw_spread": round(float(np.mean(spreads)), 4) if spreads else None,
                                "gpu_memory": gpu_memory(service)}
        except Exception as e:
            results[f"k{k}"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    results["call_to_call_spread_k1"] = round(baseline_spread, 4)
    (out / "k_samples.json").write_text(json.dumps(results, indent=2))
    return results


def mean_pairwise(chunks: list[np.ndarray]) -> float:
    """Mean absolute difference between every pair of chunks over their common horizon (all action dims)."""
    n = min(len(c) for c in chunks)
    return float(np.mean([np.abs(a[:n] - b[:n]).mean() for a, b in itertools.combinations(chunks, 2)]))


def jev(out: Path, service: Path, quick: bool) -> dict:
    capture = frame(MAIN_TASK)
    url = service_addresses(str(service), "serve")["onejev"]["url"]
    decider = JevInteraction(OneJevChooser(Client(base_url=url), list(CAMS)))
    step = Operation(OpKind.SWITCH_TO, subtask="pick up the steak")
    snaps = {"instruction": ObservationSnapshot(1, 1, 5.0, capture, 1, "actually, put the steak on the plate", 1,
                                                step, 0.0, (), "instruction"),
             "deviation": ObservationSnapshot(1, 2, 5.0, capture, 0, capture.observation.task, 1, step, 0.0, (),
                                              "deviation", deviation=Deviation(1, 1, 5.0, 3.0, 2.0))}
    latency = {}
    for name, snap in snaps.items():
        times, acts = [], []
        for _ in range(3 if quick else 8):
            w0 = time.perf_counter()
            v = decider.decide(snap)
            times.append((time.perf_counter() - w0) * 1000)
            acts.append(v.operation.kind.value)
        latency[name] = {"ms_p50": pct(times, 50), "ms_p90": pct(times, 90), "ms_max": round(max(times), 1),
                         "acts": acts}
    worst_s = max(v["ms_max"] for v in latency.values()) / 1000
    timeout = round(max(1.0, 1.5 * worst_s), 2)
    common = ["--task", MAIN_TASK, "--arm", "native", "--interaction", "jev", "--chooser", "onejev",
              "--interaction-timeout", str(timeout), "--hold-recheck", "3", "--record-interaction", str(out / "cases")]
    seconds = 25 if quick else 35
    runs = {"wait-resume": realtime(out / "wait-resume", service, common + ["--say", "10:wait", "--say", "18:go on"],
                                    seconds),
            "stop": realtime(out / "stop", service, common + ["--say", "10:stop"], seconds),
            "ask-answer": realtime(out / "ask-answer", service,
                                   common + ["--say", "10:put it over there", "--say", "18:on the plate"], seconds)}
    return {"latency": latency, "interaction_timeout_s": timeout, "runs": runs}


def mjev(out: Path, service: Path, quick: bool) -> dict:
    tasks = [MAIN_TASK] if quick else [MAIN_TASK, "MoveFridgeToFreezer"]
    runs, checks = {}, []
    for task in tasks:
        runs[task] = realtime(out / task, service, ["--task", task, "--arm", "oracle", "--verify", "mjev",
                                                    "--verify-every", "1.0"], 30 if quick else 60)
        log = out / task / "verify_checks.jsonl"
        if log.exists():
            checks += [json.loads(line) for line in log.read_text().splitlines()]
    scored = [c for c in checks if c["truth"] is not None]
    agree = [(c["states"]["done"] >= 0.5) == c["truth"] for c in scored]
    ms = [c["timings_ms"].get("round_trip") for c in checks if c["timings_ms"].get("round_trip") is not None]
    return {"runs": runs, "checks": len(checks), "with_truth": len(scored),
            "truth_done": sum(c["truth"] for c in scored), "agreement_at_0.5": round(float(np.mean(agree)), 3)
            if agree else None, "check_round_trip_ms_p50": pct(ms, 50), "check_round_trip_ms_p90": pct(ms, 90)}


def t8a_gate(out: Path, service: Path, quick: bool) -> dict:
    seconds = 30 if quick else 45
    variants = {"fixed": [], "phase": ["--commit-policy", "phase"],
                "consensus-k3": ["--speculative-predict", "--samples", "3"],
                "consensus-k5": ["--speculative-predict", "--samples", "5"],
                "consensus-k3-phase": ["--speculative-predict", "--samples", "3", "--commit-policy", "phase"]}
    if quick:
        variants = {k: variants[k] for k in ("fixed", "consensus-k3")}
    ablation = {name: realtime(out / name, service, ["--task", MAIN_TASK, "--arm", "native", *args], seconds)
                for name, args in variants.items()}
    gate = realtime(out / "gate", service, ["--task", MAIN_TASK, "--arm", "native", "--interaction", "jev",
                                            "--chooser", "onejev", "--interaction-timeout", "3", "--hold-recheck", "3",
                                            "--say", "20:actually, put the steak on the plate", "--video"],
                    60 if quick else 90)
    return {"ablation": ablation, "gate": gate}


STEPS = {"baseline": baseline, "k_samples": k_samples, "jev": jev, "mjev": mjev, "t8a_gate": t8a_gate}


def markdown(results: dict) -> str:
    lines = ["# GPU suite", ""]
    for name, r in results.items():
        lines += [f"## {name} ({r.get('minutes')} min)", "", "```json", json.dumps(r.get("result", r.get("error")),
                                                                                   indent=1)[:6000], "```", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--service", type=Path, required=True)
    p.add_argument("--only", nargs="*", choices=list(STEPS))
    p.add_argument("--quick", action="store_true", help="shorter runs, one task (smoke test of the suite)")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = a.out or REPO / "results/gpu-suite" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in a.only or list(STEPS):
        step_out = out / name
        step_out.mkdir(exist_ok=True)
        print(f"[suite] {name} …", flush=True)
        w0 = time.perf_counter()
        try:
            results[name] = {"ok": True, "result": STEPS[name](step_out, a.service.expanduser(), a.quick)}
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        results[name]["minutes"] = round((time.perf_counter() - w0) / 60, 1)
        (out / "suite.json").write_text(json.dumps(results, indent=2, default=str))
        (out / "suite.md").write_text(markdown(results))
        print(f"[suite] {name} {'done' if results[name]['ok'] else 'FAILED'} in {results[name]['minutes']} min",
              flush=True)
    print(f"[suite] -> {out}")
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
