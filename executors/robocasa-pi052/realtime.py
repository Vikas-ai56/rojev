"""RoboCasa in real time under the v3 runtime: every part on its own thread, nothing waits for a model.
`--interactive` puts a person in the loop: each line typed while the run goes is a user instruction (as `--say`), `quit`
ends the run; `--instruction TEXT` gives the overall task (default: the environment's own sentence).
`--lockstep` instead stops simulated time while any model thinks: one thread alternates S1 supply and one control tick,
and does not tick while an S2 decision is in flight (planning or interaction lane), so model computation takes no
simulated time and deadlines in simulated seconds do not depend on how fast the simulator runs. The setting benchmarks
are scored in; it separates the models' own skill from the cost of real-time execution.

  control loop   env.step at 20 Hz (wall clock) ← guard ← execution runtime
  S1 runner      keeps the command that needs actions supplied with chunks from the S1 policy
  S2 workers     planning lane (the arm's decider) and, optionally, the interaction lane
  state owner    turns proposals and user instructions into commands; ACKs say what is active

S1 (`--policy`): `remote` a model on a GPU, `--s1 pi052|xr1` (π0.5² or Xiaomi-Robotics-1; `--service` JSON with its
entry) · `skills` scripted reach / grasp / place /
press from simulator poses (harness/skills.py; a declared spoof) · `wiggle` hand back and forth (plumbing only).
Planning arms (`--arm`): `fixed` one command · `scripted` authored sequence on a time budget · `oracle` authored
sequence switched by simulator truth · `revisable` / `revisable-truth` the same, rewritten to the destination the
current instruction names · `native` π0.5² plans its own subtasks · `commit` π0.5² plans, each planned subtask is
committed until the verifier confirms it (`CommitVerifyPlanner`, `--max-commit`). Authored sequences are in tasks.py.
Interaction lane (`--interaction`): `none` (one lane) · `continue` (always-continue baseline) · `rules` (keyword
stand-in for Jev's acts) · `jev` (Jev v1, `rojev.system2.jev`: `--chooser onejev` scores acts with OneJev from the
service JSON, `--chooser keywords` with the keyword stand-in; `--reviser canonical|rewrite`: a new task is one of the
trained task sentences S1 can run here (the run's instruction plus `--skills FILE`, one per line), never the user's
words; `--max-asks`, `--jev-calibration` (JSON {"act": …, "task": …} from calibrate_jev.py));
`--hold-recheck S` asks the lane again while a hold stands; `--record-interaction DIR` saves Jev's model decisions as
calibration cases; `--interaction-delay` / `--interaction-fail` inject S2
faults on that lane. Step-completion events (`--verify mjev|truth`): end-state checks of the running step
(`rojev.system2.verify`; `truth` = simulator truth of authored steps, a declared spoof), posted to the owner; with
`--arm native --policy remote --s1 pi052` π0.5²'s own subtask switches are step events too (XR-1 reports none): claims the step watcher confirms
with one check when `--verify` is on (only confirmed ones are progress), else unchecked events. The planning heartbeat defaults to
1 s, or 0.8 × lease when step events cover progress. What the robot says is printed and logged (utterances.jsonl).
`--say T:TEXT` (repeatable) sends a user instruction T seconds in; `--on-change review|revoke`.
Commitment: `--commit-policy fixed|phase`, `--speculative-predict --samples K`, `--sample-noise` (skills: target noise
per sample). Execution monitor: `--execution-monitor MODEL.json` / `--monitor-record OUT.npz`; `--block-at T --block-s
D` blocks the arm. `--viewer` opens a live MuJoCo window (harness/viewer.py); `--trace PROJECT` exports the run to
Phoenix (`make phoenix`).

  uv run python realtime.py --task LoadCondimentsInFridge --arm revisable --policy skills --interaction rules \\
      --say "8:wait" --say "11:go on" --viewer --trace rojev-live
Outputs: <out>/{config.json, ticks.jsonl, owner.jsonl, s1_requests.jsonl, commits.jsonl, decisions.jsonl,
utterances.jsonl, verify_checks.jsonl (with --verify: each check with simulator truth), frames/ (with --record-frames
S: every S seconds the cameras and the simulator state, for `sam_check.py track`; harness/tracking.py), summary.json,
view.mp4 (with --video: three cameras + command / lane / decision panel, 10 fps)}.
After setup the process freezes the objects it built (`gc.freeze`), so a full garbage collection never walks them
during the run (measured 45–96 ms per full pass otherwise: docs/research/2026-10-22-native-runtime.md).
"""
import argparse
import gc
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
from lerobot.envs.robocasa import RoboCasaEnv

from embodiment import ARM_DIMS, PANDA_OMRON, PandaOmronEffect, to_action_space
from harness import (FailingDecider, FrameCheck, FrameRecorder, InteractionCases, KeywordChooser, LiveView,
                     MirrorPublisher, RecordingCommit, RecordingDecider, RecordingPolicy, RecordingVerifier,
                     RevisableSequence, RewriteReviser, RuleBasedInteraction, SkillScriptPolicy, SlowDecider,
                     TruthVerifier, VideoRecorder)
from history import FrameHistory, HistoryPolicy
import render_context
from rojev.control import ControlLoop, ExecutionRuntime, Guard, TickRecord
from rojev.eval import AlwaysContinue, NativePlanner, ScriptedSequence, TruthSequence
from rojev.owner import INSTRUCTION_CHANGE_MODES, INTERACTION_TRIGGERS, StateOwner
from rojev.protocol import (ACK, Deviation, Observation, ObservationSnapshot, Operation, OpKind, UserInstruction,
                            Utterance)
from rojev.system1 import (COMMIT_POLICIES, ActionChunk, ExecutionMonitor, MonitorModel, Policy, RemotePolicy, S1Runner,
                           make_commit_policy)
from rojev.system2 import CommitVerifyPlanner, DecisionWorker, Verdict
from rojev.system2.jev import TEMPLATES, Calibration, CanonicalTask, JevInteraction, OneJevChooser
from rojev.system2.models import RemoteMJev, onejev
from rojev.system2.verify import MJevVerifier, OneJevVerifier, StepWatcher
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
    live viewer. Render contexts of replaced simulators are freed in their own GL context (`render_context`), else a
    garbage collection corrupts every later camera image."""

    def __init__(self, task: str, seed: int, horizon: int, blocked: tuple[float, float] | None = None):
        render_context.install()
        self.env = RoboCasaEnv(task=task, split="pretrain", obj_registries=("objaverse", "lightwheel"),
                               terminate_on_success=False, horizon=horizon)
        self.raw, _ = self.env.reset(seed=seed)
        self.success = False
        self.success_now = False
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
        self.success_now = bool(info.get("is_success", False))
        if self.success_now and not self.success:
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

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        return self.act_samples(obs, subtask, 1)[0]

    def act_samples(self, obs: Observation, subtask: str, k: int, timeout_s: float | None = None) -> list[ActionChunk]:
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
    p.add_argument("--arm", choices=["fixed", "scripted", "oracle", "revisable", "revisable-truth", "native", "commit"],
                   default="fixed")
    p.add_argument("--max-commit", type=float, default=25.0,
                   help="commit: release an unverified committed subtask after this long (s; ≈ 2 × p75 demo subtask)")
    p.add_argument("--subtask", help="fixed arm: text for S1 (default: the task instruction)")
    p.add_argument("--policy", choices=["remote", "skills", "wiggle"], default="skills")
    p.add_argument("--service", help="remote: JSON file with {url, token} (or `kaggle`)")
    p.add_argument("--s1", choices=["pi052", "xr1"], default="pi052",
                   help="remote S1: π0.5² (lerobot/pi052_robocasa) or Xiaomi-Robotics-1 RoboCasa365 "
                        "(4 frames, 2 apart)")
    p.add_argument("--latency", type=float, default=0.3, help="skills / wiggle: simulated inference time (s)")
    p.add_argument("--sample-noise", type=float, default=0.0, help="skills / wiggle: per-sample noise (m / rad)")
    p.add_argument("--interaction", choices=["none", "continue", "rules", "jev"], default="none")
    p.add_argument("--chooser", choices=["onejev", "keywords"], default="onejev", help="jev: who scores the acts")
    p.add_argument("--reviser", choices=["canonical", "rewrite"],
                   help="jev: REVISE = one of S1's trained task sentences (default with --policy remote) or the "
                        "step rewritten for the scripted S1 (default else)")
    p.add_argument("--max-asks", type=int, default=2)
    p.add_argument("--jev-calibration", type=Path,
                   help="jev: {act, task} Calibration JSON from calibrate_jev.py (default: T=1, declared margin)")
    p.add_argument("--skills", type=Path,
                   help="jev --reviser canonical: trained task sentences S1 can run in this scene, one per line "
                        "(added to the run's instruction)")
    p.add_argument("--hold-recheck", type=float, help="ask the interaction lane again after a hold stood this long (s)")
    p.add_argument("--record-interaction", type=Path,
                   help="jev: save each model decision as a calibration case here (calibrate_jev.py)")
    p.add_argument("--verify", choices=["none", "mjev", "onejev", "truth"], default="none",
                   help="step-completion events from MJev or OneJev end-state checks, or simulator truth (spoof)")
    p.add_argument("--verify-every", type=float, default=1.0)
    p.add_argument("--lockstep", action="store_true",
                   help="simulated time stops while any model thinks (S1 supply, S2 decisions), then one control "
                        "tick (how benchmarks are scored; --seconds is simulated time)")
    p.add_argument("--frame-check-every", type=float, default=1.0, metavar="S",
                   help="sample the camera frames every S seconds; a run with noise / blank frames is not scored")
    p.add_argument("--record-frames", type=float, metavar="S",
                   help="every S seconds save the cameras and the simulator state to <out>/frames (sam_check.py track)")
    p.add_argument("--done-threshold", type=float, default=0.7)
    p.add_argument("--confirmations", type=int, default=2)
    p.add_argument("--interaction-timeout", type=float, default=1.0)
    p.add_argument("--interaction-delay", type=float, default=0.0, help="fault: interaction-lane decider latency (s)")
    p.add_argument("--interaction-fail", action="store_true", help="fault: interaction-lane decider raises")
    p.add_argument("--say", type=utterance, action="append", default=[], help="T:TEXT user instruction at T seconds")
    p.add_argument("--interactive", action="store_true",
                   help="read user instructions from the terminal while the run goes (`quit` ends it)")
    p.add_argument("--input-file", type=Path,
                   help="interactive: read user instructions appended to this file instead of the terminal")
    p.add_argument("--live-view", type=float, metavar="S",
                   help="write <out>/live/now.png + status.json every S simulated seconds (what a watcher sees)")
    p.add_argument("--max-wall", type=float, metavar="S", help="end the run after S wall-clock seconds")
    p.add_argument("--end-after-success", type=float, metavar="S",
                   help="end the run S simulated seconds after the task's own check first holds")
    p.add_argument("--instruction", help="the overall task given by the user (default: the environment's sentence)")
    p.add_argument("--on-change", choices=list(INSTRUCTION_CHANGE_MODES), default="review")
    p.add_argument("--review-s", type=float, default=1.0)
    p.add_argument("--stop-s", type=float, default=0.3)
    p.add_argument("--refill", type=float, default=1.0)
    p.add_argument("--blend", type=float, default=0.2)
    p.add_argument("--lease", type=float, default=3.0)
    p.add_argument("--decide-every", type=float,
                   help="planning heartbeat (s); default 1.0, or 0.8 × lease when step events cover progress "
                        "(--verify, or --arm native with --policy remote: π0.5²'s own subtask switches)")
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
    needs_service = (a.interaction == "jev" and a.chooser == "onejev") or a.verify in ("mjev", "onejev")
    if needs_service and not a.service:
        p.error("--chooser onejev and --verify mjev|onejev need --service with onejev / mjev entries")
    if a.arm == "commit" and (a.verify == "none" or a.policy != "remote"):
        p.error("--arm commit needs a verifier (--verify mjev|onejev|truth) and S1's own planner (--policy remote)")
    a.reviser = a.reviser or ("canonical" if a.policy == "remote" else "rewrite")
    if a.record_interaction and a.interaction != "jev":
        p.error("--record-interaction needs --interaction jev")
    step_events = a.verify != "none" or (a.arm in ("native", "commit") and a.policy == "remote")
    a.decide_every = a.decide_every or (0.8 * a.lease if step_events else 1.0)
    try:
        a.commit = make_commit_policy(a.commit_policy, a.speculative_predict, **{
            k: v for k, v in (("samples", a.samples), ("tolerance", a.tolerance)) if v is not None})
    except ValueError as e:
        p.error(str(e))
    return a


XR1_HISTORY = (4, 2)


def make_policy(a, backend: RoboCasaBackend, history: FrameHistory) -> Policy:
    """`history` feeds a policy that conditions on past frames (Xiaomi-Robotics-1: 4 frames 2 ticks apart)."""
    if a.policy == "remote":
        remote = RemotePolicy(**service_addresses(a.service, "serve")[a.s1])
        return remote if a.s1 == "pi052" else HistoryPolicy(remote, history, *XR1_HISTORY, to_action_space)
    if a.policy == "skills":
        return SkillScriptPolicy(backend.read, a.latency, a.sample_noise, seed=a.seed)
    return WigglePolicy(a.latency, noise=a.sample_noise, seed=a.seed)


def make_planner(a, backend: RoboCasaBackend):
    """The planning-lane decider and the authored steps (for truth logging)."""
    if a.arm == "fixed":
        return FixedCommand(a.subtask or backend.env.task_description), []
    if a.arm == "native":
        return NativePlanner(), []
    if a.arm == "commit":
        return CommitVerifyPlanner(a.max_commit), []
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


def skills(a, instruction: str) -> list[str]:
    """The trained task sentences a revision may choose: the run's instruction, then `--skills` (one per line)."""
    extra = [line.strip() for line in a.skills.read_text().splitlines() if line.strip()] if a.skills else []
    return list(dict.fromkeys([instruction, *extra]))


def calibrations(a) -> dict[str, Calibration]:
    """Per question (`act`, `task`): the fitted calibration from `--jev-calibration`, else the declared defaults."""
    fitted = json.loads(a.jev_calibration.read_text()) if a.jev_calibration else {}
    return {q: Calibration.from_json(json.dumps(fitted[q])) if q in fitted else Calibration() for q in ("act", "task")}


def make_jev(a, instruction: str) -> JevInteraction:
    chooser = (OneJevChooser(onejev.KeptClient(service_addresses(a.service, "serve")["onejev"]["url"]), list(CAMS))
               if a.chooser == "onejev" else KeywordChooser())
    calibration = calibrations(a)
    reviser = (CanonicalTask(chooser, skills(a, instruction), calibration["task"]) if a.reviser == "canonical"
               else RewriteReviser())
    return JevInteraction(chooser, reviser, calibration["act"], max_asks=a.max_asks)


def make_interaction(a, instruction: str):
    if a.interaction == "none":
        return None
    decider = {"continue": AlwaysContinue, "rules": RuleBasedInteraction, "jev": lambda: make_jev(a, instruction)}[a.interaction]()
    if a.record_interaction:
        run = f"{a.task}-s{a.seed}-{time.strftime('%H%M%S')}"
        decider = InteractionCases(decider, list(CAMS), a.record_interaction, run)
    if a.interaction_delay:
        decider = SlowDecider(decider, a.interaction_delay)
    if a.interaction_fail:
        decider = FailingDecider(decider, INTERACTION_TRIGGERS)
    return decider


def make_verifier(a, backend: RoboCasaBackend, steps, now) -> RecordingVerifier:
    """The end-state checker, wrapped so every check is logged with the simulator's truth (verify_checks.jsonl)."""
    checks = {s.subtask: s.done for s in steps}

    def truth(step: str) -> bool | None:
        return backend.truth(checks[step]) if step in checks else None

    if a.verify == "mjev":
        inner = MJevVerifier(RemoteMJev(**service_addresses(a.service, "serve")["mjev"]))
    elif a.verify == "onejev":
        inner = OneJevVerifier(onejev.KeptClient(service_addresses(a.service, "serve")["onejev"]["url"]))
    else:
        inner = TruthVerifier(truth)
    return RecordingVerifier(inner, truth, now)


class Console:
    """A person in the loop: lines typed on the terminal (or appended to `follow`, a file read as it grows), read on a
    daemon thread, become user instructions in the order typed (`drain`, on the run loop's thread); `quit` (or end of
    terminal input) asks the run to stop."""

    def __init__(self, follow: Path | None = None):
        self.lines: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.quit = threading.Event()
        self.log: list[dict] = []
        self.follow = follow
        threading.Thread(target=self._read, name="console", daemon=True).start()
        where = f"append lines to {follow}" if follow else "type an instruction and Enter at any time"
        print(f"[you] {where}; `quit` ends the run", flush=True)

    def _source(self):
        if self.follow is None:
            yield from sys.stdin
            return
        self.follow.touch()
        with self.follow.open() as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    time.sleep(0.2)

    def _read(self) -> None:
        for line in self._source():
            text = line.strip()
            if text.lower() == "quit":
                break
            if text:
                self.lines.put(text)
        self.quit.set()

    def drain(self) -> list[str]:
        out = []
        while not self.lines.empty():
            out.append(self.lines.get())
        return out


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


def watch(owner: StateOwner, backend: RoboCasaBackend, speaker: "Speaker") -> dict:
    """The live status a watcher reads (`LiveView`): the task in force, what the robot does, what it said."""
    op = owner.desired.operation if owner.desired is not None else None
    return {"task": owner.instruction, "doing": None if op is None else op.kind.value,
            "step": None if op is None else op.subtask,
            "holding": None if owner.hold is None else owner.hold.operation.kind.value,
            "robot_said": [u["text"] for u in speaker.log[-3:]],
            "task_check_now": backend.success_now, "task_done_at": backend.success_t}


def planner_of(worker: DecisionWorker):
    """The planning decider inside its recording wrapper."""
    return worker.decider.inner


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
    instruction = a.instruction or backend.env.task_description
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

    history = FrameHistory((XR1_HISTORY[0] - 1) * XR1_HISTORY[1] + 1)

    def on_tick(r: TickRecord) -> None:
        history.add(r.after.observation)
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

    loop = ControlLoop(backend, guard, PANDA_OMRON, realtime=not a.lockstep, on_tick=on_tick)
    history.add(loop.latest().observation)
    policy = RecordingPolicy(make_policy(a, backend, history), loop.now)
    commit = RecordingCommit(a.commit, loop.now)
    runner = S1Runner(runtime, policy, loop.latest, loop.now, refill_s=a.refill, commit=commit,
                      on_step=lambda e: watcher.claim(e) if watcher is not None else owner.post(e),
                      on_plan=lambda e: owner.post(e), deadline=not a.lockstep)
    planner, steps = make_planner(a, backend)
    planning = RecordingDecider(planner, loop.now, "planning")
    worker = DecisionWorker(planning, lambda prop: owner.post(prop), loop.now, name="planning")
    responder_decider = make_interaction(a, instruction)
    interaction = RecordingDecider(responder_decider, loop.now, "interaction") if responder_decider else None
    responder = (DecisionWorker(interaction, lambda prop: owner.post(prop), loop.now, name="interaction")
                 if interaction is not None else None)
    speaker = Speaker()
    owner = StateOwner(runtime, guard, worker, loop.latest, loop.now, instruction, lease_s=a.lease,
                       decide_every_s=a.decide_every, on_instruction_change=a.on_change, review_s=a.review_s,
                       interaction=responder, interaction_timeout_s=a.interaction_timeout, say=speaker.say,
                       hold_recheck_s=a.hold_recheck)
    watcher = (StepWatcher(make_verifier(a, backend, steps, loop.now), runtime.active_goal, loop.latest, owner.post, loop.now,
                           list(CAMS), every_s=a.verify_every, done_threshold=a.done_threshold,
                           confirmations=a.confirmations)
               if a.verify != "none" else None)
    config = {"task": a.task, "instruction": instruction, "arm": a.arm, "planner": planner.name, "s1": a.s1,
              "interaction": interaction.name if interaction else None, "policy": a.policy, "seed": a.seed,
              "sequence": [s.subtask for s in steps], "say": a.say, "latency_s": a.latency, "sample_noise": a.sample_noise,
              "seconds": a.seconds, "refill_s": a.refill, "blend_s": a.blend, "lease_s": a.lease,
              "decide_every_s": a.decide_every, "on_change": a.on_change, "review_s": a.review_s, "stop_s": a.stop_s,
              "interaction_timeout_s": a.interaction_timeout, "interaction_delay_s": a.interaction_delay,
              "interaction_fail": a.interaction_fail, "commit": repr(a.commit),
              "chooser": a.chooser if a.interaction == "jev" else None,
              "reviser": a.reviser if a.interaction == "jev" else None, "max_asks": a.max_asks,
              "jev_calibration": str(a.jev_calibration) if a.jev_calibration else None,
              "skills": skills(a, instruction) if a.interaction == "jev" and a.reviser == "canonical" else None,
              "hold_recheck_s": a.hold_recheck,
              "verify": None if watcher is None else {"every_s": a.verify_every, "done_threshold": a.done_threshold,
                                                      "confirmations": a.confirmations},
              "execution_monitor": str(a.execution_monitor) if a.execution_monitor else None,
              "blocked": {"at": a.block_at, "s": a.block_s} if a.block_at is not None else None,
              "control_hz": PANDA_OMRON.control_hz}
    frames = (FrameRecorder(loop.latest, backend.read, list(CAMS), out / "frames", every_s=a.record_frames)
              if a.record_frames else None)
    config["record_frames_s"] = a.record_frames
    frame_check = FrameCheck(loop.latest, list(CAMS), every_s=a.frame_check_every)
    config["lockstep"] = a.lockstep
    config["interactive"] = a.interactive or a.input_file is not None
    config.update({"input_file": str(a.input_file) if a.input_file else None, "live_view_s": a.live_view,
                   "max_wall_s": a.max_wall, "end_after_success_s": a.end_after_success})
    config["frame_check_every_s"] = a.frame_check_every
    (out / "config.json").write_text(json.dumps(config, indent=2))
    driven = () if a.lockstep else (loop, runner)
    threads = driven + (worker, owner, frame_check) + tuple(t for t in (responder, watcher, frames) if t is not None)
    pending = sorted(a.say)
    console = Console(a.input_file) if a.interactive or a.input_file else None
    live = LiveView(out / "live", a.live_view) if a.live_view else None
    ended = "seconds"
    gc.collect()
    gc.freeze()
    t0_ns, w0 = time.time_ns(), time.perf_counter()
    for th in threads:
        th.start()
    try:
        while (loop.now() if a.lockstep else time.perf_counter() - w0) < a.seconds and loop.error is None:
            while pending and loop.now() >= pending[0][0]:
                owner.post(UserInstruction(pending.pop(0)[1]))
            if console is not None:
                for text in console.drain():
                    console.log.append({"t": round(loop.now(), 3), "text": text})
                    owner.post(UserInstruction(text))
                if console.quit.is_set():
                    ended = "quit"
                    break
            if a.max_wall is not None and time.perf_counter() - w0 >= a.max_wall:
                ended = "wall"
                break
            if a.end_after_success is not None and backend.success_t is not None \
                    and loop.now() >= backend.success_t + a.end_after_success:
                ended = "success"
                break
            if live is not None and live.due(loop.now()):
                capture = loop.latest()
                live.write(loop.now(), [capture.observation.images[c] for c in CAMS], watch(owner, backend, speaker))
            if a.lockstep:
                if any(lane is not None and lane.busy for lane in (worker, responder)):
                    time.sleep(0.002)
                    continue
                runner.step()
                loop.run_ticks(1)
            else:
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
    if console is not None:
        write_jsonl(out / "typed.jsonl", console.log)
    if watcher is not None:
        write_jsonl(out / "verify_checks.jsonl", watcher.verifier.log)
    if frames is not None:
        frames.save()
    summary = summarize(a, config, out, loop, runtime, runner, owner, worker, responder, monitor, model, backend,
                        steps, wall, watcher, frames)
    summary["frame_check"] = frame_check.report()
    summary["ended"] = "error" if loop.error is not None else ended
    if not summary["frame_check"]["valid"]:
        print(f"[realtime] WARNING: camera frames not real ({summary['frame_check']}); this run is not scored",
              flush=True)
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
              wall, watcher, frames) -> dict:
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
                "claims": watcher.stats.claims, "unconfirmed": watcher.stats.unconfirmed,
                "last_error": watcher.stats.last_error},
            "frames_recorded": None if frames is None else len(frames.states),
            "truth_now": {s.subtask: backend.truth(s.done) for s in steps},
            "s1_requests": st.requests, "s1_accepted": st.accepted, "s1_errors": st.errors, "s1_last_error": st.last_error,
            "s1_latency_est_s": None if runner.latency_s() is None else round(runner.latency_s(), 3),
            "s1_floor_last": st.floors[-1] if st.floors else None, "s1_commit_reasons": st.commit_reasons,
            "s1_forced": st.forced, "s1_planner_steps": st.planner_steps, "s1_timeouts": st.timeouts,
            "commit": None if not isinstance(planner_of(worker), CommitVerifyPlanner) else {
                "reentries": planner_of(worker).reentries, "expired": planner_of(worker).expired,
                "verified": sum(k.checked for k in owner.keyframes)},
            "monitor": None if monitor is None or model is None else {
                "threshold": round(model.threshold, 3), "max_score": round(monitor.max_score, 3),
                "uncovered_ticks": monitor.uncovered,
                "deviations": [{"t": d.t, "cmd": d.cmd_id, "score": d.score} for d in monitor.deviations]},
            "s2_planning": {"decisions": worker.stats.decisions, "errors": worker.stats.errors},
            "violations": sum(bool(r["violations"]) for r in rows)}


if __name__ == "__main__":
    raise SystemExit(main())
