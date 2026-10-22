"""Live workflow episode on the Mac: RoboCasa sim (local, live window) + π0.5² (remote GPU) + a subtask arm.

Loop, once per 50-step chunk (synchronous: the sim waits while the models answer):
  arm.decide (B fixed budget / E oracle / D OneJev+MJev)  →  executor.act(images + state + subtask)  →  50 env steps
Arm D uses both cloud GPUs at once: while MJev (GPU 1) and OneJev (Mac) decide, π0.5² (GPU 0) already computes the
chunk for "continue the current subtask"; it is used if the decision is "continue" (speculation hit), else discarded.
Every stage is visible three ways:
  live window   3 policy cameras + task, arm decision and reason, subtask i/n, step, timings, ground-truth checklist
                (step first done; "now" = true at this moment; yellow = was done, no longer, e.g. object dropped)
  Phoenix       one trace per episode: decide → executor.act (with the camera image sent) → sim.chunk spans
  files         <out>/{run_config.json, chunks.jsonl, summary.json, episode.mp4}
Any RoboCasa pretrain task runs with arm D, scored by RoboCasa's own task-success check. Tasks with an authored
sequence in tasks.py (evaluation only) also get per-subtask ground truth and arms B / E; that attributes failures:
wrong subtask chosen (arm) vs right subtask not executed (executor).

  uv run python workflow.py --task MicrowaveThawingFridge --arm E --service kaggle
  uv run python workflow.py --task MicrowaveThawingFridge --arm D --service services.json   # needs `make qev`
  uv run python workflow.py --task MicrowaveThawingFridge --arm B --executor hold   # loop/view/logging check, no GPU
Keys in the live window: q = abort episode, p = pause/resume.
"""
import argparse
import base64
import json
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import cv2
import imageio.v3 as iio
import numpy as np
from lerobot.envs.robocasa import RoboCasaEnv
from robocasa.utils.dataset_registry import PRETRAINING_TASKS
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from qev import Client

from rojev.legacy_v2 import CandidateBuilder, DecisionContext, FixedSequence, OneJevSupervisor, OracleSequence, normalize
from rojev.observability import tracing
from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy, RemotePolicy
from rojev.system2.models import RemoteMJev
from tasks import TASKS

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAMS = ("robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right")
SEQUENCES = REPO / "data/vocab/pi052_robocasa_sequences.jsonl"  # π0.5² training timelines (E8) for the candidate filter
PANEL_H = 230
DEFAULT_MAX_STEPS = 2000  # tasks without a tasks.py spec (lever; specs use ≈ 1.5 × p90 demo length)


class HoldPolicy(Policy):
    """Zero actions (robot holds still): checks the loop, live window and logging without a GPU."""

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        return ActionChunk(np.zeros((50, 12), np.float32), {"infer": 0.0})

    def reset(self) -> None:
        pass


def service_addresses(spec: str, mode: str) -> dict:
    """{"pi052": {url, token}, "mjev": {url, token}?}. `kaggle` = follow the kernel log until it prints them; else a
    JSON file. A bare {url, token} (π0.5²-only serve mode) is read as {"pi052": …}."""
    if spec == "kaggle":
        out = subprocess.run([sys.executable, str(HERE / "kaggle/push.py"), "service", mode], capture_output=True,
                             text=True, check=True).stdout
        d = json.loads(out.strip().splitlines()[-1])
    else:
        d = json.loads(Path(spec).read_text())
    return {"pi052": d} if "url" in d else d


class LiveView:
    """One composite frame per env step: cameras on top, a text panel below. Shown live and written to the mp4."""

    def __init__(self, show: bool, video: Path):
        self.show, self.paused, self.abort = show, False, False
        self.writer = iio.imopen(video, "w", plugin="pyav")
        self.writer.init_video_stream("libx264", fps=20)

    def frame(self, images: dict, lines: list[tuple[str, tuple]]) -> np.ndarray:
        top = np.concatenate([images[c] for c in CAMS], axis=1)
        panel = np.full((PANEL_H, top.shape[1], 3), 24, np.uint8)
        y = 20
        for text, color in lines:
            for part in textwrap.wrap(text, 120) or [""]:
                cv2.putText(panel, part, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                y += 17
        return np.concatenate([top, panel], axis=0)

    def update(self, frame: np.ndarray, record: bool = True) -> None:
        if record:
            self.writer.write_frame(frame)
        if self.show:
            cv2.imshow("rojev live", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.abort = True
            elif key == ord("p"):
                self.paused = not self.paused
            while self.paused and not self.abort:
                key = cv2.waitKey(100) & 0xFF
                self.paused = key != ord("p")
                self.abort = key == ord("q")

    def close(self) -> None:
        self.writer.close()
        if self.show:
            cv2.destroyAllWindows()


def jpeg_uri(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(PRETRAINING_TASKS["pretrain300"]), required=True,
                   help="any RoboCasa pretrain task; arms B/E need an authored sequence in tasks.py")
    p.add_argument("--arm", choices=["B", "D", "E"], required=True)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--executor", choices=["remote", "hold"], default="remote")
    p.add_argument("--service", default="kaggle",
                   help="'kaggle' (read from the running kernel's log: rojev-armd for arm D, else rojev-pi052) or a JSON file")
    p.add_argument("--max-steps", type=int, help="override the task's step limit")
    p.add_argument("--onejev", default="http://127.0.0.1:8000", help="arm D: qev server (`make qev`)")
    p.add_argument("--k", type=int, default=20, help="arm D: shortlist size from the candidate filter")
    p.add_argument("--m", type=int, default=5, help="arm D: top candidates MJev judges (plus current + recent)")
    p.add_argument("--no-mjev", action="store_true", help="arm D without MJev evidence (ablation)")
    p.add_argument("--no-view", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    spec = TASKS.get(a.task)
    if spec is None and a.arm in ("B", "E"):
        p.error(f"arms B/E need an authored sequence in tasks.py; available: {sorted(TASKS)}")
    max_steps = a.max_steps or (spec.max_steps if spec else DEFAULT_MAX_STEPS)
    run_id = f"{a.task}_{a.arm}_s{a.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = a.out or REPO / "results/runs/workflow" / run_id
    out.mkdir(parents=True, exist_ok=True)

    provider = tracing.setup("rojev-workflow")
    tracer = trace.get_tracer("rojev.workflow")
    need_services = a.executor == "remote" or (a.arm == "D" and not a.no_mjev)
    services = service_addresses(a.service, "armd" if a.arm == "D" else "serve") if need_services else {}
    executor: Policy = HoldPolicy() if a.executor == "hold" else RemotePolicy(**services["pi052"])
    if isinstance(executor, RemotePolicy):
        print(f"[workflow] executor service: {executor.health()}", flush=True)

    t0 = time.perf_counter()
    env = RoboCasaEnv(task=a.task, split="pretrain", obj_registries=("objaverse", "lightwheel"),
                      terminate_on_success=False, horizon=max_steps + 1000)  # robosuite horizon must not end it first
    obs, _ = env.reset(seed=a.seed)
    kitchen = env._env.env
    steps = spec.steps(kitchen) if spec else []  # per-subtask ground truth only where a sequence is authored
    seq = [s.subtask for s in steps]
    if a.arm == "B":
        arm = FixedSequence(seq, spec.chunks_per_subtask)
    elif a.arm == "E":
        arm = OracleSequence(seq, lambda i: steps[i].done(kitchen))
    else:
        timelines = [json.loads(line) for line in SEQUENCES.read_text().splitlines()]
        builder = CandidateBuilder.fit([(e["task"], e["subtasks"]) for e in timelines if e["subtasks"]])
        mjev = None if a.no_mjev or "mjev" not in services else RemoteMJev(**services["mjev"])
        if mjev is not None:
            print(f"[workflow] mjev service: {mjev.health()}", flush=True)
        arm = OneJevSupervisor(Client(a.onejev), builder, mjev, CAMS, k=a.k, m=a.m, seed=a.seed, session_id=run_id)
    speculate = a.arm == "D"  # B/E decide instantly: nothing to overlap
    executor.reset()
    config = {"task": a.task, "instruction": env.task_description, "arm": arm.name, "seed": a.seed,
              "executor": a.executor, "max_steps": max_steps,
              "chunks_per_subtask": spec.chunks_per_subtask if spec else None,
              "sequence": seq, "scene_setup_s": round(time.perf_counter() - t0, 1),
              **({"k": a.k, "m": a.m, "mjev": not a.no_mjev and "mjev" in services, "speculate": speculate}
                 if a.arm == "D" else {})}
    (out / "run_config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps(config, indent=2), flush=True)

    view = LiveView(not a.no_view, out / "episode.mp4")
    first_done: list[int | None] = [None] * len(steps)
    step, chunk_i, current, on_current, history = 0, 0, None, 0, []
    success_step, end_reason, last_timings, decision = None, None, {}, None
    log = (out / "chunks.jsonl").open("w")
    pool = ThreadPoolExecutor(max_workers=1)
    spec_stats = {"hit": 0, "miss": 0}
    mjev_scores: list[dict] = []  # arm D: MJev "done" judgement vs simulator ground truth, per judged subtask

    def score_evidence(evidence: dict, truth_now: dict) -> list[dict]:
        """MJev's done-probability vs ground truth for judged subtasks that are steps of this task (same meaning)."""
        by_key = {normalize(k): v for k, v in truth_now.items()}
        return [{"subtask": sub, "p_done": round(e["done"], 3), "truth_done": by_key[normalize(sub)],
                 "correct": (max(e, key=e.get) == "done") == by_key[normalize(sub)]}
                for sub, e in evidence.items() if normalize(sub) in by_key]

    def panel(status: str) -> list[tuple[str, tuple]]:
        white, grey, green, yellow = (235, 235, 235), (150, 150, 150), (90, 220, 90), (240, 200, 60)
        lines = [(f"TASK  {env.task_description}", white),
                 (f"ARM {arm.name}  |  step {step}/{max_steps}  chunk {chunk_i}  |  {status}", yellow),
                 (f"SUBTASK [{(arm.index + 1) if current and hasattr(arm, 'index') else len(history) + bool(current)}"
                  f"/{len(seq) if hasattr(arm, 'index') else '?'}]  {current or '-'}"
                  f"   <- {decision.reason if decision else ''}", white),
                 (f"timings ms  {last_timings}", grey)]
        for i, s in enumerate(steps):
            held = s.done(kitchen)
            mark = (f"[x] step {first_done[i]}" if first_done[i] is not None else "[ ]") + (" now" if held else "")
            lines.append((f"  truth {mark:18s} {s.subtask}", green if held else (yellow if first_done[i] else grey)))
        if success_step is not None:
            lines.append((f"TASK SUCCESS at step {success_step}", green))
        return lines

    with tracer.start_as_current_span(f"episode:{a.task}:{a.arm}") as ep:
        ep.set_attribute("openinference.span.kind", "CHAIN")
        ep.set_attribute("session.id", run_id)
        ep.set_attribute("input.value", json.dumps(config))
        ep.set_attribute("input.mime_type", "application/json")
        try:
            while step < max_steps and not view.abort:
                images = {c: obs["pixels"][c] for c in CAMS}
                observation = Observation(images, obs["agent_pos"], env.task_description)
                ctx = DecisionContext(chunk_i, step, current, on_current, tuple(history), observation)
                truth_at_decision = {s.subtask: bool(s.done(kitchen)) for s in steps}
                # Speculation (arm D): π0.5² on GPU 0 computes "continue current" while MJev + OneJev decide.
                speculative = pool.submit(executor.act, observation, current) if speculate and current else None
                view.update(view.frame(images, panel("deciding …")), record=False)
                with tracer.start_as_current_span(f"decide:{arm.name}") as sp:
                    sp.set_attribute("openinference.span.kind", "CHAIN")
                    sp.set_attribute("input.value", json.dumps({k: v for k, v in ctx.__dict__.items()
                                                                if k != "observation"}))
                    t_dec = time.perf_counter()
                    decision = arm.decide(ctx)
                    decide_ms = round((time.perf_counter() - t_dec) * 1000, 1)
                    sp.set_attribute("output.value", json.dumps({**decision.__dict__, "decide_ms": decide_ms}))
                if decision.subtask is None:
                    end_reason = f"arm: {decision.reason}"
                    break
                hit = speculative is not None and decision.subtask == current
                if speculative is not None:
                    spec_stats["hit" if hit else "miss"] += 1
                if decision.subtask != current:
                    if current is not None:
                        history.append(current)
                    current, on_current = decision.subtask, 0
                view.update(view.frame(images, panel("waiting for π0.5² …")), record=False)
                with tracer.start_as_current_span("executor.act") as sp:
                    sp.set_attribute("openinference.span.kind", "TOOL")
                    sp.set_attribute("input.value", json.dumps({"subtask": current, "task": env.task_description,
                                                                "state": obs["agent_pos"].round(4).tolist(),
                                                                "speculation": None if speculative is None
                                                                else ("hit" if hit else "miss")}))
                    sp.set_attribute("input.image", jpeg_uri(np.concatenate([images[c] for c in CAMS], axis=1)))
                    t_act = time.perf_counter()
                    if hit:
                        chunk = speculative.result()
                    else:
                        if speculative is not None:
                            speculative.result()  # the server is serial: let the discarded call finish first
                        chunk = executor.act(observation, current)
                    chunk = ActionChunk(chunk.actions, {**chunk.timings_ms, "decide": decide_ms,
                                                        "wait_after_decide": round((time.perf_counter() - t_act) * 1000, 1)})
                    last_timings = chunk.timings_ms
                    sp.set_attribute("output.value", json.dumps({"actions": chunk.actions.shape,
                                                                 "timings_ms": chunk.timings_ms}))
                with tracer.start_as_current_span("sim.chunk") as sp:
                    sp.set_attribute("openinference.span.kind", "CHAIN")
                    start = step
                    for action in chunk.actions:
                        obs, _r, _term, _trunc, info = env.step(action)
                        step += 1
                        for i, s in enumerate(steps):
                            if first_done[i] is None and s.done(kitchen):
                                first_done[i] = step
                        if info["is_success"] and success_step is None:
                            success_step = step
                        view.update(view.frame({c: obs["pixels"][c] for c in CAMS}, panel("running")))
                        if success_step is not None or step >= max_steps or view.abort:
                            break
                    truth = {s.subtask: first_done[i] for i, s in enumerate(steps)}
                    now = {s.subtask: bool(s.done(kitchen)) for s in steps}  # e.g. still grasped vs dropped again
                    sp.set_attribute("output.value", json.dumps({"steps": [start, step], "truth_first_done": truth,
                                                                 "truth_now": now, "success_step": success_step}))
                scored = score_evidence(decision.detail.get("evidence", {}), truth_at_decision)
                mjev_scores += scored
                record = {"chunk": chunk_i, "steps": [start, step], "decision": decision.__dict__,
                          "evidence_vs_truth": scored,
                          "subtask": current, "timings_ms": chunk.timings_ms, "truth_first_done": truth,
                          "truth_now": now, "success_step": success_step}
                log.write(json.dumps(record) + "\n")
                log.flush()
                chunk_i, on_current = chunk_i + 1, on_current + 1
                if success_step is not None:
                    end_reason = "task success (env check)"
                    break
            end_reason = end_reason or ("aborted by user" if view.abort else "step limit")
            summary = {**config, "success": success_step is not None, "success_step": success_step, "steps": step,
                       "chunks": chunk_i, "end_reason": end_reason, "subtasks_switched": history + [current],
                       "speculation": spec_stats,
                       "mjev_done_accuracy": (round(sum(x["correct"] for x in mjev_scores) / len(mjev_scores), 3)
                                              if mjev_scores else None), "mjev_scored_judgements": len(mjev_scores),
                       "truth_first_done": dict(zip(seq, first_done)),
                       "wall_s": round(time.perf_counter() - t0, 1)}
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            ep.set_attribute("output.value", json.dumps(summary))
            ep.set_status(Status(StatusCode.OK if summary["success"] else StatusCode.ERROR, end_reason))
            print(json.dumps(summary, indent=2), flush=True)
        finally:
            log.close()
            view.update(view.frame({c: obs["pixels"][c] for c in CAMS}, panel(end_reason or "stopped")), record=True)
            view.close()
            pool.shutdown()
            env.close()
            provider.shutdown()
    print(f"[workflow] -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
