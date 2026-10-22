"""RoboCasa in real time under the v3 runtime: every part on its own thread, nothing waits for a model.

  control loop   env.step at 20 Hz (wall clock) ← guard ← execution runtime
  S1 runner      keeps the command that needs actions supplied with chunks from the S1 policy
  S2 workers     planning lane (the arm's decider) and, optionally, the interaction lane
  state owner    turns proposals and user instructions into commands; ACKs say what is active

S1 (`--policy`): `remote` π0.5² on a GPU (`--service` JSON {url, token}) · `skills` scripted reach / grasp / place /
press from simulator poses (harness/skills.py; a declared spoof) · `wiggle` hand back and forth (plumbing only).
Planning arms (`--arm`): `fixed` one command · `scripted` authored sequence on a time budget · `oracle` authored
sequence switched by simulator truth · `revisable` / `revisable-truth` the same, rewritten to the destination the
current instruction names · `native` π0.5² plans its own subtasks. Authored sequences are in tasks.py.
Interaction lane (`--interaction`): `none` (one lane) · `continue` (always-continue baseline) · `rules` (keyword
stand-in for Jev's acts) · `jev` (Jev v1, `rojev.system2.jev`: `--chooser onejev` scores acts with OneJev from the
service JSON, `--chooser keywords` with the keyword stand-in; `--reviser s1|rewrite`, `--max-asks`, `--jev-calibration`);
`--hold-recheck S` asks the lane again while a hold stands; `--interaction-delay` / `--interaction-fail` inject S2
faults on that lane. Step-completion events (`--verify mjev|truth`): end-state checks of the running step
(`rojev.system2.verify`; `truth` = simulator truth of authored steps, a declared spoof), posted to the owner. What the robot says is printed and logged (utterances.jsonl).
`--say T:TEXT` (repeatable) sends a user instruction T seconds in; `--on-change review|revoke`.
Commitment: `--commit-policy fixed|phase`, `--speculative-predict --samples K`, `--sample-noise` (skills: target noise
per sample). Execution monitor: `--execution-monitor MODEL.json` / `--monitor-record OUT.npz`; `--block-at T --block-s
D` blocks the arm. `--viewer` opens a live MuJoCo window (harness/viewer.py); `--trace PROJECT` exports the run to
Phoenix (`make phoenix`).

  uv run python realtime.py --task LoadCondimentsInFridge --arm revisable --policy skills --interaction rules \\
      --say "8:wait" --say "11:go on" --viewer --trace rojev-live
Outputs: <out>/{config.json, ticks.jsonl, owner.jsonl, s1_requests.jsonl, commits.jsonl, decisions.jsonl,
utterances.jsonl, summary.json, view.mp4 (with --video: three cameras + command / lane / decision panel, 10 fps)}.
After setup the process freezes the objects it built (`gc.freeze`), so a full garbage collection never walks them
during the run (measured 45–96 ms per full pass otherwise: docs/research/2026-10-22-native-runtime.md).
"""
import argparse
import gc
import json
import threading
import time
from pathlib import Path

import numpy as np
from lerobot.envs.robocasa import RoboCasaEnv
from qev import Client

from embodiment import ARM_DIMS, PANDA_OMRON, PandaOmronEffect
from harness import (FailingDecider, KeywordChooser, MirrorPublisher, RecordingCommit, RecordingDecider,
                     RecordingPolicy, RevisableSequence, RewriteReviser, RuleBasedInteraction, SkillScriptPolicy,
                     SlowDecider, TruthVerifier, VideoRecorder)
from rojev.control import ControlLoop, ExecutionRuntime, Guard, TickRecord
from rojev.eval import AlwaysContinue, NativePlanner, ScriptedSequence, TruthSequence
from rojev.owner import INSTRUCTION_CHANGE_MODES, INTERACTION_TRIGGERS, StateOwner
from rojev.protocol import (ACK, Deviation, Observation, ObservationSnapshot, Operation, OpKind, UserInstruction,
                            Utterance)
from rojev.system1 import (COMMIT_POLICIES, ActionChunk, ExecutionMonitor, MonitorModel, Policy, RemotePolicy, S1Runner,
                           make_commit_policy)
from rojev.system2 import DecisionWorker, Verdict
from rojev.system2.jev import TEMPLATES, Calibration, JevInteraction, OneJevChooser, S1Replans
from rojev.system2.models import RemoteMJev
from rojev.system2.verify import MJevVerifier, StepWatcher
from tasks import TASKS
from workflow import CAMS, service_addresses

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
START = np.array([0.0] * 6 + [-1.0] + [0.0] * 4 + [-1.0])
CHUNK_S = 50 / PANDA_OMRON.control_hz
VIDEO_EVERY = 2
PANEL_FPS = int(PANDA_OMRON.control_hz // VIDEO_EVERY)


class RoboCasaBackend:
    """The LeRobot RoboCasa environment as a runtime `Backend`. `success` latches RoboCasa's own task check; `read`
    evaluates a function of the kitchen between two steps (never during one); `publisher` mirrors each step to the
    live viewer."""

    def __init__(self, task: str, seed: int, horizon: int, blocked: tuple[float, float] | None = None):
        self.env = RoboCasaEnv(task=task, split="pretrain", obj_registries=("objaverse", "lightwheel"),
                               terminate_on_success=False, horizon=horizon)
        self.raw, _ = self.env.reset(seed=seed)
        self.success = False
        self.success_t: float | None = None
        self.steps = 0
        self.blocked = blocked
        self.publisher: MirrorPublisher | None = None
        self._lock = threading.Lock()

    @property
    def kitchen(self):
        return self.env._env.env

    def read(self, fn):
        with self._lock:
            return fn(self.kitchen)

    def truth(self, check) -> bool:
        return bool(self.read(check))

    def observe(self) -> Observation:
        return Observation({c: self.raw["pixels"][c] for c in CAMS}, self.raw["agent_pos"], self.env.task_description)

    def step(self, action) -> Observation:
        action = np.asarray(action, dtype=np.float32).copy()
        t = self.steps / PANDA_OMRON.control_hz
        if self.blocked is not None and self.blocked[0] <= t < self.blocked[0] + self.blocked[1]:
            action[list(ARM_DIMS)] = 0.0
        with self._lock:
            self.raw, _r, _term, _trunc, info = self.env.step(action)
            self.steps += 1
            if self.publisher is not None:
                self.publisher.publish(self.kitchen.sim.data._data)
        if info.get("is_success", False) and not self.success:
            self.success, self.success_t = True, self.steps / PANDA_OMRON.control_hz
        return self.observe()


class FixedCommand:
    """One command for the whole episode (plumbing and S1-alone checks)."""

    name = "fixed"

    def __init__(self, subtask: str):
        self.subtask = subtask

    def decide(self, s: ObservationSnapshot) -> Verdict:
        if s.active_operation is None:
            return Verdict(Operation(OpKind.SWITCH_TO, subtask=self.subtask), "start")
        return Verdict(Operation(OpKind.CONTINUE), "fixed command")


class WigglePolicy(Policy):
    """Stand-in S1: after `latency_s`, a chunk that moves the hand left and right with the gripper open. Samples differ
    by a random phase drift of size `noise` (a stochastic policy); K samples cost one batched inference."""

    def __init__(self, latency_s: float, horizon: int = 50, noise: float = 0.0, seed: int = 0):
        self.latency_s, self.horizon, self.noise = latency_s, horizon, noise
        self.rng = np.random.default_rng(seed)

    def sample(self) -> ActionChunk:
        drift = self.rng.normal(0.0, self.noise) * np.linspace(0, 1, self.horizon) ** 2
        a = np.zeros((self.horizon, PANDA_OMRON.size), np.float32)
        a[:, 1] = 0.3 * np.sin(np.linspace(0, 2 * np.pi, self.horizon) + drift)
        a[:, 6] = -1.0
        a[:, 11] = -1.0
        return ActionChunk(a, {"infer": self.latency_s * 1000})

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        return self.act_samples(obs, subtask, 1)[0]

    def act_samples(self, obs: Observation, subtask: str, k: int) -> list[ActionChunk]:
        time.sleep(self.latency_s)
        return [self.sample() for _ in range(k)]

    def reset(self) -> None:
        pass


def panel(t, owner: StateOwner) -> list[str]:
    """Video text: time and tick state; the desired command; the owner's latest decision or event."""
    goal = owner.desired
    command = "none" if goal is None else f"cmd {goal.cmd_id} {goal.operation.kind.value} {goal.operation.subtask or ''}"
    last = next((e for e in reversed(owner.record.events) if e["kind"] not in ("ack", "planning deferred")), None)
    latest = "" if last is None else (f"{last['t']:.2f}s {last['kind']} " + " ".join(
        str(last[k]) for k in ("lane", "op", "subtask", "text", "why", "reason") if last.get(k)))
    state = f"t={t.t:5.2f}s  {t.source:<5} {t.mode.value:<10}" + ("  REVIEW" if t.review else "") + \
        ("  STARVED" if t.starved else "") + (f"  held {','.join(t.held_switches)}" if t.held_switches else "")
    return [state, command, latest]


def utterance(spec: str) -> tuple[float, str]:
    at, _, text = spec.partition(":")
    if not text:
        raise argparse.ArgumentTypeError(f"--say expects T:TEXT, got {spec!r}")
    return float(at), text


def parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="LoadCondimentsInFridge")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--arm", choices=["fixed", "scripted", "oracle", "revisable", "revisable-truth", "native"],
                   default="fixed")
    p.add_argument("--subtask", help="fixed arm: text for S1 (default: the task instruction)")
    p.add_argument("--policy", choices=["remote", "skills", "wiggle"], default="skills")
    p.add_argument("--service", help="remote: JSON file with {url, token} (or `kaggle`)")
    p.add_argument("--latency", type=float, default=0.3, help="skills / wiggle: simulated inference time (s)")
    p.add_argument("--sample-noise", type=float, default=0.0, help="skills / wiggle: per-sample noise (m / rad)")
    p.add_argument("--interaction", choices=["none", "continue", "rules", "jev"], default="none")
    p.add_argument("--chooser", choices=["onejev", "keywords"], default="onejev", help="jev: who scores the acts")
    p.add_argument("--reviser", choices=["s1", "rewrite"],
                   help="jev: REVISE = S1 re-plans (default with --policy remote) or the step rewritten (default else)")
    p.add_argument("--max-asks", type=int, default=2)
    p.add_argument("--jev-calibration", type=Path, help="jev: Calibration JSON (default: T=1, declared margin)")
    p.add_argument("--hold-recheck", type=float, help="ask the interaction lane again after a hold stood this long (s)")
    p.add_argument("--verify", choices=["none", "mjev", "truth"], default="none",
                   help="step-completion events from MJev, or from simulator truth (spoof)")
    p.add_argument("--verify-every", type=float, default=1.0)
    p.add_argument("--done-threshold", type=float, default=0.7)
    p.add_argument("--confirmations", type=int, default=2)
    p.add_argument("--interaction-timeout", type=float, default=1.0)
    p.add_argument("--interaction-delay", type=float, default=0.0, help="fault: interaction-lane decider latency (s)")
    p.add_argument("--interaction-fail", action="store_true", help="fault: interaction-lane decider raises")
    p.add_argument("--say", type=utterance, action="append", default=[], help="T:TEXT user instruction at T seconds")
    p.add_argument("--on-change", choices=list(INSTRUCTION_CHANGE_MODES), default="review")
    p.add_argument("--review-s", type=float, default=1.0)
    p.add_argument("--stop-s", type=float, default=0.3)
    p.add_argument("--refill", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=0.2)
    p.add_argument("--lease", type=float, default=3.0)
    p.add_argument("--decide-every", type=float, default=1.0)
    p.add_argument("--commit-policy", choices=list(COMMIT_POLICIES), default="fixed")
    p.add_argument("--speculative-predict", action="store_true")
    p.add_argument("--samples", type=int)
    p.add_argument("--tolerance", type=float)
    p.add_argument("--execution-monitor", type=Path)
    p.add_argument("--monitor-record", type=Path)
    p.add_argument("--block-at", type=float)
    p.add_argument("--block-s", type=float, default=2.0)
    p.add_argument("--viewer", action="store_true", help="live MuJoCo window (separate mjpython process)")
    p.add_argument("--trace", metavar="PROJECT", help="export the run to Phoenix under this project")
    p.add_argument("--video", action="store_true")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if a.policy == "remote" and not a.service:
        p.error("--policy remote needs --service")
    if (a.interaction_delay or a.interaction_fail) and a.interaction == "none":
        p.error("interaction faults need an interaction lane (--interaction continue|rules|jev)")
    needs_service = (a.interaction == "jev" and a.chooser == "onejev") or a.verify == "mjev"
    if needs_service and not a.service:
        p.error("--chooser onejev and --verify mjev need --service with onejev / mjev entries")
    a.reviser = a.reviser or ("s1" if a.policy == "remote" else "rewrite")
    try:
        a.commit = make_commit_policy(a.commit_policy, a.speculative_predict, **{
            k: v for k, v in (("samples", a.samples), ("tolerance", a.tolerance)) if v is not None})
    except ValueError as e:
        p.error(str(e))
    return a


def make_policy(a, backend: RoboCasaBackend) -> Policy:
    if a.policy == "remote":
        return RemotePolicy(**service_addresses(a.service, "serve")["pi052"])
    if a.policy == "skills":
        return SkillScriptPolicy(backend.read, a.latency, a.sample_noise, seed=a.seed)
    return WigglePolicy(a.latency, noise=a.sample_noise, seed=a.seed)


def make_planner(a, backend: RoboCasaBackend):
    """The planning-lane decider and the authored steps (for truth logging)."""
    if a.arm == "fixed":
        return FixedCommand(a.subtask or backend.env.task_description), []
    if a.arm == "native":
        return NativePlanner(), []
    spec = TASKS.get(a.task)
    if spec is None:
        raise SystemExit(f"arm {a.arm} needs an authored sequence in tasks.py; available: {sorted(TASKS)}")
    steps = spec.steps(backend.kitchen)
    seq = [s.subtask for s in steps]
    budget = spec.chunks_per_subtask * CHUNK_S

    def is_done(i: int) -> bool:
        return backend.truth(steps[i].done)

    return {"scripted": lambda: ScriptedSequence(seq, budget), "oracle": lambda: TruthSequence(seq, is_done),
            "revisable": lambda: RevisableSequence(seq, budget_s=budget),
            "revisable-truth": lambda: RevisableSequence(seq, is_done=is_done)}[a.arm](), steps


def make_jev(a) -> JevInteraction:
    chooser = (OneJevChooser(Client(base_url=service_addresses(a.service, "serve")["onejev"]["url"]), list(CAMS))
               if a.chooser == "onejev" else KeywordChooser())
    calibration = Calibration.from_json(a.jev_calibration.read_text()) if a.jev_calibration else Calibration()
    reviser = S1Replans() if a.reviser == "s1" else RewriteReviser()
    return JevInteraction(chooser, calibration, reviser, max_asks=a.max_asks)


def make_interaction(a):
    if a.interaction == "none":
        return None
    decider = {"continue": AlwaysContinue, "rules": RuleBasedInteraction, "jev": lambda: make_jev(a)}[a.interaction]()
    if a.interaction_delay:
        decider = SlowDecider(decider, a.interaction_delay)
    if a.interaction_fail:
        decider = FailingDecider(decider, INTERACTION_TRIGGERS)
    return decider


def make_verifier(a, backend: RoboCasaBackend, steps):
    if a.verify == "mjev":
        return MJevVerifier(RemoteMJev(**service_addresses(a.service, "serve")["mjev"]))
    checks = {s.subtask: s.done for s in steps}
    return TruthVerifier(lambda step: backend.truth(checks[step]) if step in checks else None)


class Speaker:
    """Renders what the robot says (`Utterance` → text): printed and kept for utterances.jsonl."""

    def __init__(self):
        self.log: list[dict] = []

    def say(self, u: Utterance) -> None:
        text = "(heard you)" if u.template == ACK else TEMPLATES.get(u.template, u.template)
        if u.objects:
            text += f" [{', '.join(u.objects)}]"
        self.log.append({"t": round(u.t, 3), "template": u.template, "objects": list(u.objects), "cmd": u.cmd_id,
                         "text": text})
        print(f"[robot] {u.t:6.2f}s {text}", flush=True)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))


def main(argv=None) -> int:
    a = parse(argv)
    run_id = f"realtime_{a.task}_{a.arm}_{a.policy}_s{a.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    out = a.out or REPO / "results/runs/realtime" / run_id
    out.mkdir(parents=True, exist_ok=True)
    backend = RoboCasaBackend(a.task, a.seed, horizon=int(a.seconds * PANDA_OMRON.control_hz) + 1000,
                              blocked=(a.block_at, a.block_s) if a.block_at is not None else None)
    if a.viewer:
        backend.publisher = MirrorPublisher(backend.kitchen.sim.model._model, backend.kitchen.sim.data._data)
        backend.publisher.launch()
    instruction = backend.env.task_description
    runtime = ExecutionRuntime(PANDA_OMRON, epoch=1, last=START.copy(), blend_s=a.blend, stop_s=a.stop_s)
    guard = Guard(runtime)
    log = (out / "ticks.jsonl").open("w")
    video = VideoRecorder(out / "view.mp4", fps=PANEL_FPS) if a.video else None
    owner: StateOwner | None = None
    runner: S1Runner | None = None

    def on_deviation(d: Deviation) -> None:
        owner.post(d)
        runner.refill_now(d.cmd_id)

    model = MonitorModel.from_json(a.execution_monitor.read_text()) if a.execution_monitor else None
    monitor = (ExecutionMonitor(PandaOmronEffect(), 1, model, on_deviation, record=a.monitor_record is not None)
               if model is not None or a.monitor_record is not None else None)

    def on_tick(r: TickRecord) -> None:
        t = r.tick
        if monitor is not None:
            monitor.observe(r)
        log.write(json.dumps({"t": round(t.t, 3), "source": t.source, "mode": t.mode.value, "cmd": t.cmd_id,
                              "skipped": t.skipped, "held": list(t.held_switches), "violations": list(t.violations),
                              "hold_reason": t.hold_reason, "review": t.review, "starved": t.starved,
                              "step_ms": round(r.step_ms, 1), "lag_ms": round(r.lag_s * 1000, 1),
                              "action": np.round(t.action, 4).tolist(),
                              "score": None if monitor is None or monitor.score is None else round(monitor.score, 3)})
                  + "\n")
        if video is not None and t.index % VIDEO_EVERY == 0:
            video.add([backend.raw["pixels"][c] for c in CAMS], panel(t, owner))

    loop = ControlLoop(backend, guard, PANDA_OMRON, realtime=True, on_tick=on_tick)
    policy = RecordingPolicy(make_policy(a, backend), loop.now)
    commit = RecordingCommit(a.commit, loop.now)
    runner = S1Runner(runtime, policy, loop.latest, loop.now, refill_s=a.refill, commit=commit)
    planner, steps = make_planner(a, backend)
    planning = RecordingDecider(planner, loop.now, "planning")
    worker = DecisionWorker(planning, lambda prop: owner.post(prop), loop.now, name="planning")
    responder_decider = make_interaction(a)
    interaction = RecordingDecider(responder_decider, loop.now, "interaction") if responder_decider else None
    responder = (DecisionWorker(interaction, lambda prop: owner.post(prop), loop.now, name="interaction")
                 if interaction is not None else None)
    speaker = Speaker()
    owner = StateOwner(runtime, guard, worker, loop.latest, loop.now, instruction, lease_s=a.lease,
                       decide_every_s=a.decide_every, on_instruction_change=a.on_change, review_s=a.review_s,
                       interaction=responder, interaction_timeout_s=a.interaction_timeout, say=speaker.say,
                       hold_recheck_s=a.hold_recheck)
    watcher = (StepWatcher(make_verifier(a, backend, steps), lambda: runtime.active, loop.latest, owner.post, loop.now,
                           list(CAMS), every_s=a.verify_every, done_threshold=a.done_threshold,
                           confirmations=a.confirmations)
               if a.verify != "none" else None)
    config = {"task": a.task, "instruction": instruction, "arm": a.arm, "planner": planner.name,
              "interaction": interaction.name if interaction else None, "policy": a.policy, "seed": a.seed,
              "sequence": [s.subtask for s in steps], "say": a.say, "latency_s": a.latency, "sample_noise": a.sample_noise,
              "seconds": a.seconds, "refill_s": a.refill, "blend_s": a.blend, "lease_s": a.lease,
              "decide_every_s": a.decide_every, "on_change": a.on_change, "review_s": a.review_s, "stop_s": a.stop_s,
              "interaction_timeout_s": a.interaction_timeout, "interaction_delay_s": a.interaction_delay,
              "interaction_fail": a.interaction_fail, "commit": repr(a.commit),
              "chooser": a.chooser if a.interaction == "jev" else None,
              "reviser": a.reviser if a.interaction == "jev" else None, "max_asks": a.max_asks,
              "jev_calibration": str(a.jev_calibration) if a.jev_calibration else None,
              "hold_recheck_s": a.hold_recheck,
              "verify": None if watcher is None else {"every_s": a.verify_every, "done_threshold": a.done_threshold,
                                                      "confirmations": a.confirmations},
              "execution_monitor": str(a.execution_monitor) if a.execution_monitor else None,
              "blocked": {"at": a.block_at, "s": a.block_s} if a.block_at is not None else None,
              "control_hz": PANDA_OMRON.control_hz}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    threads = (loop, runner, worker, owner) + tuple(t for t in (responder, watcher) if t is not None)
    pending = sorted(a.say)
    gc.collect()
    gc.freeze()
    t0_ns, w0 = time.time_ns(), time.perf_counter()
    for th in threads:
        th.start()
    try:
        while time.perf_counter() - w0 < a.seconds and loop.error is None:
            while pending and loop.now() >= pending[0][0]:
                owner.post(UserInstruction(pending.pop(0)[1]))
            time.sleep(0.01)
    finally:
        for th in reversed(threads):
            th.stop()
        log.close()
        if video is not None:
            video.close()
        if backend.publisher is not None:
            backend.publisher.close()
    wall = time.perf_counter() - w0
    ev = owner.record.events
    decisions = sorted(planning.log + (interaction.log if interaction else []), key=lambda d: d["t_start"])
    write_jsonl(out / "owner.jsonl", ev)
    write_jsonl(out / "s1_requests.jsonl", policy.log)
    write_jsonl(out / "commits.jsonl", commit.log)
    write_jsonl(out / "decisions.jsonl", decisions)
    write_jsonl(out / "utterances.jsonl", speaker.log)
    summary = summarize(a, config, out, loop, runtime, runner, owner, worker, responder, monitor, model, backend,
                        steps, wall, watcher)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    if monitor is not None and monitor.recording is not None:
        monitor.recording.conditions = {"realtime_factor": summary["realtime_factor"], "policy": a.policy,
                                        "latency_s": a.latency, "task": a.task, "seed": a.seed,
                                        "commit": repr(a.commit), "starved_ticks": runtime.starved_ticks}
        monitor.recording.save(a.monitor_record)
    if a.trace:
        from opentelemetry import trace

        from rojev.observability import run_trace, tracing

        provider = tracing.setup(a.trace)
        run_trace.export_run(trace.get_tracer("rojev.realtime"), out.name, t0_ns, loop.now(), config, summary, ev,
                             policy.log, commit.log, decisions)
        provider.force_flush()
        provider.shutdown()
    print(json.dumps({k: summary[k] for k in ("realtime_factor", "starved_ticks", "violations", "commands",
                                              "keyframes", "task_success")}, indent=2, default=str))
    print(f"[realtime] -> {out}")
    backend.env.close()
    return 0


def summarize(a, config, out, loop, runtime, runner, owner, worker, responder, monitor, model, backend, steps,
              wall, watcher) -> dict:
    rows = [json.loads(line) for line in (out / "ticks.jsonl").read_text().splitlines()]
    chunk_ticks = sum(r["source"] == "chunk" for r in rows)
    step_ms = sorted(r["step_ms"] for r in rows)
    ev = owner.record.events
    st = runner.stats
    return {**config, "wall_s": round(wall, 2), "ticks": loop.ticks, "sim_s": round(loop.now(), 2),
            "realtime_factor": round(loop.now() / wall, 3), "max_lag_ms": round(loop.max_lag_s * 1000, 1),
            "step_ms_p50": step_ms[len(step_ms) // 2], "step_ms_p95": step_ms[int(len(step_ms) * 0.95)],
            "chunk_ticks": chunk_ticks, "hold_ticks": len(rows) - chunk_ticks, "starved_ticks": runtime.starved_ticks,
            "task_success": backend.success, "success_t": backend.success_t, "complete": owner.complete,
            "commands": [{"t": e["t"], "cmd": e["cmd"], "op": e["op"], "subtask": e["subtask"], "lane": e.get("lane"),
                          "reason": e["reason"]} for e in ev if e["kind"] in ("command", "reaffirm")],
            "refused": [e for e in ev if e["kind"] == "refused"],
            "keyframes": [{"cmd": k.cmd_id, "subtask": k.subtask, "rev": k.instruction_rev, "t": k.capture.t,
                           "evidence": k.evidence} for k in owner.keyframes],
            "interaction_lane": None if responder is None else {
                "decisions": responder.stats.decisions, "errors": responder.stats.errors,
                "timeouts": sum(e["kind"] == "interaction timeout" for e in ev),
                "failures": sum(e["kind"] == "decision failed" for e in ev),
                "deferred_planning": sum(e["kind"] == "planning deferred" for e in ev)},
            "ack_counts": {s: sum(e["kind"] == "ack" and e["status"] == s for e in ev)
                           for s in ("activated", "revoked", "rejected", "chunk_rejected", "expired", "dropped")},
            "holds": [{"t": e["t"], "op": e["op"], "interrupted": e["interrupted"]} for e in ev if e["kind"] == "hold"],
            "said": [{"t": e["t"], "template": e["template"]} for e in ev if e["kind"] == "say"],
            "step_events": [{"t": e["t"], "cmd": e["cmd"], "source": e["source"], "detail": e["detail"]}
                            for e in ev if e["kind"] == "step event"],
            "verify": None if watcher is None else {
                "checks": watcher.stats.checks, "errors": watcher.stats.errors, "events": watcher.stats.events,
                "last_error": watcher.stats.last_error},
            "truth_now": {s.subtask: backend.truth(s.done) for s in steps},
            "s1_requests": st.requests, "s1_accepted": st.accepted, "s1_errors": st.errors, "s1_last_error": st.last_error,
            "s1_latency_est_s": None if runner.latency_s() is None else round(runner.latency_s(), 3),
            "s1_floor_last": st.floors[-1] if st.floors else None, "s1_commit_reasons": st.commit_reasons,
            "s1_forced": st.forced,
            "monitor": None if monitor is None or model is None else {
                "threshold": round(model.threshold, 3), "max_score": round(monitor.max_score, 3),
                "uncovered_ticks": monitor.uncovered,
                "deviations": [{"t": d.t, "cmd": d.cmd_id, "score": d.score} for d in monitor.deviations]},
            "s2_planning": {"decisions": worker.stats.decisions, "errors": worker.stats.errors},
            "violations": sum(bool(r["violations"]) for r in rows)}


if __name__ == "__main__":
    raise SystemExit(main())
