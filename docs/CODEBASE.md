# rojev: Code Map

A guide for reading and reviewing the code; documents are listed in [`README.md`](README.md). The SDK is the `rojev/` package, split along the architecture's boundaries ([`architecture.md`](architecture.md)); robot- and model-specific code lives beside it in `executors/` and `services/`, each with its own uv environment. Current stage: v3 (Jev × Helix), v1 = the full System-1 / System-2 loop from open-source frozen parts ([`plan/v3-v1-spec.md`](plan/v3-v1-spec.md), tracked in [`plan/v3-roadmap.md`](plan/v3-roadmap.md)).

**Contents:** [1 Big picture](#1-big-picture) · [2 Package map](#2-package-map) · [3 Review path](#3-review-path) · [4 Files](#4-files) · [5 Who writes what](#5-who-writes-what) · [6 Extension points](#6-extension-points) · [7 Configuration and pins](#7-configuration-and-pins) · [8 Tests and tools](#8-tests-and-tools) · [9 Verified](#9-verified)

---

## 1. Big picture

System 2 (Jev = OneJev + MJev, or a baseline) chooses what to do; System 1 (π0.5²) does it; the control layer owns what is physically active; the state owner is the only writer of intent between them. Every part runs on its own thread; nothing waits for a model. Simulation runs on the Mac; the models run as services on a GPU.

```
 user ── UserInstruction ─────────────────────────────▶ owner.StateOwner ── AcceptedGoal / Revoke / renew ──┐
                                                          ▲ DecisionProposal      ▲ ExecutionAck              │
 system2.DecisionWorker ── Decider (Jev | eval baseline) ─┘                       │                           ▼
        ▲ ObservationSnapshot (newest)                       control.ExecutionRuntime ◀── TimedChunk ── system1.S1Runner ── Policy (π0.5² service)
        │                                                          │ tick(t) → action                                   ▲ newest Capture
 control.ControlLoop ◀── control.Guard ◀───────────────────────────┘                                                    │
        │ backend.step(action) → observation → latest Capture ──────────────────────────────────────────────────────────┘
        ▼
 RoboCasa Franka (executors/robocasa-pi052)  |  any rojev.robot.Robot via control.RobotBackend (SO-101 sim, later hardware)
```

Authority: advice is evidence; a proposal is not a command; a command is active only once the execution runtime has applied its first action. Every message carries the session epoch.

---

## 2. Package map

Bottom to top; each package imports only those listed as its dependencies (enforced by `tests/test_architecture.py`).

```
rojev/
  protocol/        shared contracts: observation.py (Observation, Capture) · commands.py · decisions.py        deps: —
  robot/           robot layer: Robot, clocks, limits, ActionSpec; sim/ = MuJoCo backend (robot, world, GL)    deps: —
  transport/       remote model calls: HTTP/3 (pinned TLS), HTTP/1.1 keep-alive, shared server Endpoint      deps: —
  control/         real-time layer: execution.py (ExecutionRuntime), guard.py, loop.py, robot_backend.py     deps: protocol, robot
  system1/         policy.py (Policy, ActionChunk), commit.py (CommitPolicy), wire.py, remote.py (RemotePolicy), runner.py (S1Runner)  deps: protocol, robot, control, transport
  system2/         decider.py, worker.py, jev/ (Jev v1 interaction), verify/ (end-state checks), models/  deps: protocol, observability, transport
  owner/           state_owner.py (StateOwner): single writer of intent                                     deps: protocol, control, system2
  eval/            deciders.py: baselines scripted / oracle / native / always-continue (evaluation only)                      deps: protocol, system2
  observability/   tracing.py (Phoenix), run_trace.py (post-run span tree)                                 deps: —
  legacy_v2/       QUARANTINED v2 chunk-synchronous supervisor (arm D, B/E, candidate filter); delete after T8  deps: protocol, system2, eval
executors/robocasa-pi052/  π0.5² service, RoboCasa backend + PandaOmron ActionSpec, live runner, scenario harness, GPU hook
services/onejev, services/mjev, services/sam2  S2 model services (own envs, GPU hooks)
scripts/         gpu_setup.sh: one command from the Mac to a serving GPU box
executors/robocasa/        earlier SmolVLA executor env (v2 G0)
services/mjev/             MJev-Spatial service + checks (own env)
services/sam2/             SAM 2.1 masks + streaming tracks (own env)
examples/ tests/ playground/ scenes/ config/ vendor/ (git-ignored)
```

---

## 3. Review path

Read in this order; each step adds one layer of the loop.

| # | Read | To understand |
|---|---|---|
| 1 | [`protocol/`](../rojev/protocol/__init__.py): [`observation.py`](../rojev/protocol/observation.py) → [`commands.py`](../rojev/protocol/commands.py) → [`decisions.py`](../rojev/protocol/decisions.py) | Everything that crosses between layers, and the authority rules |
| 2 | [`robot/action_spec.py`](../rojev/robot/action_spec.py) | How any robot plugs in: its action vector as an `ActionSpec` |
| 3 | [`control/execution.py`](../rojev/control/execution.py) `ExecutionRuntime.tick` | What is physically active: leases, revoke, irreversible switches, skipped actions, blending |
| 4 | [`control/guard.py`](../rojev/control/guard.py) → [`control/loop.py`](../rojev/control/loop.py) | What keeps the robot safe, and how it keeps its own clock |
| 5 | [`system1/policy.py`](../rojev/system1/policy.py) → [`system1/commit.py`](../rojev/system1/commit.py) → [`system1/runner.py`](../rojev/system1/runner.py) | How S1 is kept supplied with chunks, how much of each is committed, placed at observation time |
| 6 | [`system2/decider.py`](../rojev/system2/decider.py) → [`system2/worker.py`](../rojev/system2/worker.py) | How a decider's verdict becomes a stamped proposal |
| 7 | [`owner/state_owner.py`](../rojev/owner/state_owner.py) | How proposals and instructions become commands; why stale ones are refused |
| 8 | [`eval/deciders.py`](../rojev/eval/deciders.py) | The baselines Jev must beat |
| 9 | [`executors/robocasa-pi052/realtime.py`](../executors/robocasa-pi052/realtime.py) `main` | The whole v1 loop wired together in RoboCasa |
| 10 | [`system1/wire.py`](../rojev/system1/wire.py) → [`system1/remote.py`](../rojev/system1/remote.py) → [`executors/robocasa-pi052/serve.py`](../executors/robocasa-pi052/serve.py) | How a chunk is fetched from π0.5² on a GPU |
| 11 | [`robot/base.py`](../rojev/robot/base.py) → [`robot/sim/mujoco_robot.py`](../rojev/robot/sim/mujoco_robot.py) → [`control/robot_backend.py`](../rojev/control/robot_backend.py) | The hardware-shaped robot interface, the SO-101 simulator, and how it joins the loop |
| 12 | [`system2/models/`](../rojev/system2/models/__init__.py) → [`system2/jev/`](../rojev/system2/jev/__init__.py) → [`system2/verify/`](../rojev/system2/verify/__init__.py) | The OneJev and MJev clients; Jev v1 answering a change; end-state checks posting step events |

---

## 4. Files

### `rojev/protocol/` (shared contracts; depends on nothing)
| File | What it does | Look at |
|---|---|---|
| [`observation.py`](../rojev/protocol/observation.py) | What every layer sees: one observation; an observation stamped with its robot-clock time | `Observation`, `Capture` |
| [`commands.py`](../rojev/protocol/commands.py) | Owner ↔ S1 runner ↔ execution runtime; each carries `epoch` | `OpKind`, `Operation`, `Transition`, `AcceptedGoal`, `Revoke`, `TimedChunk`, `AckStatus`, `ExecutionAck` |
| [`decisions.py`](../rojev/protocol/decisions.py) | Owner ↔ System 2 (and the monitor): what a decider sees and proposes; progress keyframes; failures; deviations | `UserInstruction`, `ObservationSnapshot`, `Keyframe`, `StepEvent`, `Turn`, `Utterance`, `Hold`, `DecisionProposal`, `DecisionFailure`, `Deviation`, `Precondition`, `Advice` |

### `rojev/robot/` (robot layer; simulator = one backend)
| File | What it does | Look at |
|---|---|---|
| [`types.py`](../rojev/robot/types.py) | Timestamped samples and commands in canonical units | `Stamp`, `JointState`, `CameraFrame`, `CameraInfo`, `RobotObservation`, `JointCommand`, `CommandResult` |
| [`base.py`](../rojev/robot/base.py) | `Robot` (read / write through the safety filter, serialized across threads), `Clock`, `Rate` (drops missed ticks) | `Robot.write`, `Rate.sleep` |
| [`limits.py`](../rojev/robot/limits.py) | Joint limits from the model plus sourced extras; clamps and reports every violation | `JointLimits.from_model`, `SafetyFilter.apply` |
| [`effect.py`](../rojev/robot/effect.py) | What an action should do to the measured state (execution-monitor plug-in): tracked quantities + commanded change; `JointTargetEffect` for any joint-target robot | `EffectModel`, `JointTargetEffect` |
| [`action_spec.py`](../rojev/robot/action_spec.py) | A robot's action vector for the control layer: bounds, hold value (`KEEP` = repeat last), switch threshold, irreversible flag; `joint_target_spec` for any `Robot` | `ActionSpec.hold`, `irreversible_change`, `limit`, `joint_target_spec` |
| [`mock.py`](../rojev/robot/mock.py) | No physics: commands become state | `MockRobot` |
| [`so101.py`](../rojev/robot/so101.py) | SO-101 factory from `config/so101.yaml` | `simulated` |
| [`sim/mujoco_robot.py`](../rojev/robot/sim/mujoco_robot.py) | MuJoCo backend: time modes `lockstep` / `paced` / `realtime` (physics thread), cameras with rate/latency/calibration, privileged truth for evaluators | `MujocoRobot`, `SimClock`, `RealtimeSimClock`, `SimGroundTruth`, `camera_info` |
| [`sim/gl.py`](../rojev/robot/sim/gl.py) | One thread owns all OpenGL contexts of a simulated robot (macOS CGL blocks across threads) | `GLThread.call` |
| [`sim/world.py`](../rojev/robot/sim/world.py) | World + vendor robot composition (solver options kept), contacts, segmentation pixel counts | `compose`, `contacts_with`, `visible_pixels` |

### `rojev/transport/` (remote model calls; depends on nothing)

| File | What it does | Key symbols |
|---|---|---|
| [`__init__.py`](../rojev/transport/__init__.py) | `open_transport(url, token, cert_pem)`: `h3://` → HTTP/3, `http(s)://` → HTTP/1.1; exports the servers | `open_transport` |
| [`endpoint.py`](../rojev/transport/endpoint.py) | Server side every service shares: admission (token, declared length) before the body; at most one execution per request id (replay cache) | `Endpoint` (`admit`, `handle`), `ReplayCache`, `REQUEST_ID`, `new_request_id`, `parse_length` |
| [`h3.py`](../rojev/transport/h3.py) | HTTP/3 over QUIC: persistent client connection (one retry with the same request id on `ConnectionLost`, never on `ServiceError`; `health` through the same path), CUBIC, session resumption; server admits on headers, 413 + STOP_SENDING past the limit, handler off the QUIC loop; self-signed certificate pinned as the only trust anchor | `H3Transport`, `H3Server(endpoint, …)`, `make_certificate` |
| [`http.py`](../rojev/transport/http.py) | HTTP/1.1 over one kept TCP connection (one replay-safe retry; `timeout_s` bounds the whole call); threaded server for an `Endpoint` | `HttpTransport`, `HttpServer` |

### `rojev/control/` (real-time layer)
| File | What it does | Look at |
|---|---|---|
| [`execution.py`](../rojev/control/execution.py) | One action per tick. Only current epoch / newer command / unexpired lease; active on first applied action; revoke drops queued actions; review keeps motion but freezes irreversible switches; irreversible switches only while running a chunk outside review; elapsed actions skipped; segments blended; decelerating hold (`stop_s`); `rejected` vs `chunk_rejected` ACKs; starvation measured (`starved_ticks`); `queued_from(cmd, t)` = the command's queued actions from t (read-only) | `ExecutionRuntime.accept`, `submit`, `revoke`, `review`, `renew`, `tick`, `target`, `queued_from`, `Mode` |
| [`guard.py`](../rojev/control/guard.py) | Watchdog ("starved", keeps the command), owner heartbeat ("owner lost") and runtime errors (drop the commands: `dropped` ACK), stop | `Guard.step` |
| [`loop.py`](../rojev/control/loop.py) | Steps a `Backend` at its control rate on its own thread: the robot's clock if it has one, else the wall clock; lag reported; newest observation in a latest-value slot; `TickRecord` carries the observation the action was applied to and the one that followed | `ControlLoop`, `Backend`, `TickRecord` |
| [`robot_backend.py`](../rojev/control/robot_backend.py) | Any `rojev.robot.Robot` as a loop backend (joint targets through its safety filter; its clock paces the loop) | `RobotBackend` |

### `rojev/system1/` (fast visuomotor policy)
| File | What it does | Look at |
|---|---|---|
| [`policy.py`](../rojev/system1/policy.py) | Policy contract: observation + subtask text → action chunk; empty subtask = the policy's own planner; `info` = what it reports; `act_samples` = K candidates; a policy that `accepts_continuation` is given the actions it runs while it thinks (real-time chunking) | `Policy`, `ActionChunk`, `Continuation` |
| [`commit.py`](../rojev/system1/commit.py) | How much of a chunk to commit: whole · through the first slow point · K-sample majority consensus (`speculative_predict`, opt-in); never below the runner's latency-derived floor; per-step strict majority on irreversible switches, ties keep the value in effect (the robot's at the first step, from `last`) | `majority_switches`, `make_commit_policy`, `FixedCommit`, `PhaseCommit`, `ConsensusCommit`, `Commitment` |
| [`monitor.py`](../rojev/system1/monitor.py) | Execution monitor (opt-in): per tick, measured vs predicted change of the tracked state over a window; linear map from this and the previous window's commands (lag-aware), residual scale ≥ the effect model's resolution; split-conformal threshold, split by run; command directions never exercised are not scored; one `Deviation` per excursion | `ExecutionMonitor.observe`, `fit_monitor`, `MonitorModel`, `Recording` |
| [`wire.py`](../rojev/system1/wire.py) | Versioned npz wire (v3, no pickle): K samples per request, optional continuation, codecs `raw` / `webp90` (default) | `VERSION`, `DEFAULT_CODEC`, `ActRequest`, `encode_request`, `decode_request`, `encode_response`, `decode_response` |
| [`service.py`](../rojev/system1/service.py) | The server half of a remote S1: any `Policy` behind the wire (`/act`, `/reset`, `/health`), one model lock, idle shutdown; wrapped by `Endpoint` | `PolicyService`, `serve_policy` |
| [`remote.py`](../rojev/system1/remote.py) | A remote S1 through `rojev.transport`; K samples in one request; forwards a continuation; network and round-trip timings; passes the server's `info` through | `RemotePolicy`, `act_samples` |
| [`runner.py`](../rojev/system1/runner.py) | Keeps the target command supplied: measures request latency; asks the policy (with the command's instruction) when less than `max(refill_s, latency + dt)` is queued; passes the floor ⌈2·latency/dt⌉ + 1 to the commitment policy; passes the action last applied to the commitment policy; submits the committed prefix at the observation time; `refill_now(cmd_id)` forces a fresh chunk only for the command it supplies; counts `refill_ignored`, `floor_over_chunk`; posts π0.5²'s own subtask switches as unchecked `StepEvent` claims (`on_step`); each request's deadline = one chunk duration (`deadline_s`, `timeouts`); a continuation (queued actions from the capture, ⌈L/dt⌉ delay steps) to a policy that accepts one (`continuations`) | `S1Runner.step`, `floor`, `lead_s`, `refill_now`, `S1Stats` |

### `rojev/system2/` (slow decision layer)
| File | What it does | Look at |
|---|---|---|
| [`decider.py`](../rojev/system2/decider.py) | Decider contract: snapshot → verdict (operation, reason, scores, preconditions, evidence) | `Decider`, `Verdict` |
| [`worker.py`](../rojev/system2/worker.py) | Runs a decider on its own thread (one per lane); newest request wins; stamps the `DecisionProposal` (epoch, request id, instruction revision, base command, expiry); a decider error becomes a `DecisionFailure` | `DecisionWorker`, `S2Stats` |
| [`planning.py`](../rojev/system2/planning.py) | Commit and verify for a self-planning S1: commit its planned subtask, release on a checked step event or the declared bound, never re-commit a verified step | `CommitVerifyPlanner` |
| [`models/onejev.py`](../rojev/system2/models/onejev.py) | OneJev call traced as an OpenInference span (images, state, answers, logits); qev's client over one kept connection | `system_one`, `KeptClient` |
| [`models/images.py`](../rojev/system2/models/images.py) | Lossless image / 1-bit mask encoding shared by the S2 clients | `png_b64`, `mask_from_b64` |
| [`models/mjev.py`](../rojev/system2/models/mjev.py) | MJev-Spatial client over one kept `rojev.transport` connection | `RemoteMJev.ask` |
| [`models/sam2.py`](../rojev/system2/models/sam2.py) | SAM 2.1 client: masks from point prompts; streaming tracks (frame-numbered steps, retry-safe) | `RemoteSam2.segment`, `track_start`, `track_step`, `track_end`, `PointPrompt`, `Mask`, `Track` |
| [`jev/acts.py`](../rojev/system2/jev/acts.py) | Legal interaction acts per situation (one per distinct effect; no retry), question budget, act → operation, the `Reviser` / `Revision` contract for REVISE, robot speech templates | `candidates`, `situation_of`, `effect`, `may_ask`, `operation`, `Revision`, `Reviser`, `TEMPLATES` |
| [`jev/task_choice.py`](../rojev/system2/jev/task_choice.py) | The second question: which of S1's trained task sentences a revision means (or none → ask); the user's words never become the task | `CanonicalTask`, `NONE`, `TASK_QUESTION` |
| [`jev/calibration.py`](../rojev/system2/jev/calibration.py) | Temperature scaling + abstention margin, fitted offline: T by NLL, margin = smallest reaching a target accuracy on decided cases | `Calibration`, `fit_temperature`, `fit_margin`, `fit_calibration` |
| [`jev/chooser.py`](../rojev/system2/jev/chooser.py) | Who scores one question's options: one OneJev Choice over instruction, step, progress, dialogue, hold, images | `Chooser`, `OneJevChooser`, `situation`, `ACT_QUESTION` |
| [`jev/interaction.py`](../rojev/system2/jev/interaction.py) | Jev v1 interaction decider: wait budget, dwell, act question, calibrate, abstain, task question on revise, budgets, operation | `JevInteraction` |
| [`verify/checks.py`](../rojev/system2/verify/checks.py) | End-state check per step (done / possible / blocked) for any step text: MJev in one request, or OneJev with the identical question | `Verifier`, `MJevVerifier`, `OneJevVerifier`, `STATES` |
| [`verify/watcher.py`](../rojev/system2/verify/watcher.py) | Thread checking the running step; confirmed P(done) → one checked `StepEvent` per command; confirms claims (π0.5²'s subtask switch) with one check | `StepWatcher`, `claim`, `confirm` |

### `rojev/owner/`, `rojev/eval/`, `rojev/observability/`
| File | What it does | Look at |
|---|---|---|
| [`owner/state_owner.py`](../rojev/owner/state_owner.py) | Single writer of instruction (revision), desired command and active record (from ACKs). On an instruction change: `review` (default; reaffirm on CONTINUE) or `revoke`; two lanes (interaction: instruction / deviation / hold, priority, deadline, escalation to planning, deferred planning replayed; planning: the rest); standing holds answering changes; dialogue + utterances (`say`); step events; accepts only proposals answering its own current requests; CONTINUE renews leases; progress keyframes when a running step is moved past; falls back to the running command if a pending one fails or is dropped; heartbeat to the guard | `StateOwner.handle`, `tick`, `request_decision`, `_on_instruction`, `_on_deviation`, `_on_step`, `_on_failure`, `_on_proposal`, `_apply`, `_update_hold`, `_keep_hold`, `_record_progress`, `_issue`, `_on_ack` |
| [`eval/deciders.py`](../rojev/eval/deciders.py) | Baselines: `scripted` (time budget), `oracle` (simulator truth), `native` (S1's own planner), `always-continue` (interaction lane). Never shipped | `ScriptedSequence`, `TruthSequence`, `NativePlanner`, `AlwaysContinue`, `next_after_latest_done` |
| [`eval/scorecard.py`](../rojev/eval/scorecard.py) | Revision-benchmark scorecard (T8): `Injection` / `Case` / `RunRecords` → `score_run` per run, `aggregate` per arm, `render_markdown` / `render_json`; robot-agnostic, reads record dicts only | `score_run`, `aggregate`, `Injection`, `Case`, `RunRecords` |
| [`observability/tracing.py`](../rojev/observability/tracing.py) | OpenTelemetry → local Phoenix | `setup` |
| [`observability/run_trace.py`](../rojev/observability/run_trace.py) | One run as a Phoenix span tree after the run: commands (ACKs, reviews, keyframes, deviations), S1 requests + commitments, S2 decisions per lane with outcome, user utterances | `export_run` |

### `rojev/legacy_v2/` (QUARANTINED; only `workflow.py`, E9/E13 playground scripts and their tests use it)
| File | What it does |
|---|---|
| [`arms.py`](../rojev/legacy_v2/arms.py) | v2 per-chunk arm contract (`SubtaskSource`, `DecisionContext`, `Decision`) |
| [`onejev_arm.py`](../rojev/legacy_v2/onejev_arm.py) | Arm D: MJev judges candidates, one OneJev choice |
| [`candidates.py`](../rojev/legacy_v2/candidates.py) | v2 candidate filter over π0.5²'s 21,326 subtasks (BM25, transitions, object overlap) |
| [`baselines.py`](../rojev/legacy_v2/baselines.py) | v2 arms B (fixed) and E (truth) |

### `executors/robocasa-pi052/` (π0.5² + RoboCasa integration; own env)
| File | What it does | Look at |
|---|---|---|
| [`realtime.py`](../executors/robocasa-pi052/realtime.py) | The v1 live loop from its entrypoint: S1 `remote` / `skills` / `wiggle`; arms `fixed` / `scripted` / `oracle` / `revisable(-truth)` / `native`; interaction lane `none` / `continue` / `rules` / `jev` (+ delay / failure faults, hold recheck); step events `--verify mjev|truth`; robot speech printed + utterances.jsonl; `gc.freeze` after setup; `--say T:TEXT`; commitment and monitor flags; `--block-at`; `--lockstep` (simulated time stops while S1 thinks; diagnostic); `--record-frames S` (cameras + state for `sam_check.py track`); `--viewer` (live MuJoCo), `--video` (cameras + decision panel), `--trace` (Phoenix); per-call logs | `RoboCasaBackend`, `make_policy`, `make_planner`, `make_jev`, `make_interaction`, `make_verifier`, `Speaker`, `main`, `summarize` |
| [`scenarios.py`](../executors/robocasa-pi052/scenarios.py) | Integration / regression matrix (step 5a): 19 cases run through `realtime.py`, invariants + per-case expectations, report.json, video and Phoenix per case; declared changes per case scored by the T8 scorecard (scorecard.md / .json, `--score-only RUN_DIR`) | `SCENARIOS`, `Scenario.changes`, `run_case`, `score_matrix` |
| [`harness/skills.py`](../executors/robocasa-pi052/harness/skills.py) | SPOOF S1 without a GPU: pick / place / press from simulator poses, chunk planned forward from the measured hand | `SkillScriptPolicy` |
| [`harness/deciders.py`](../executors/robocasa-pi052/harness/deciders.py) | SPOOF deciders: keyword interaction acts (for Jev), keyword scores inside the real Jev (for OneJev), step rewrite (for the trained-task choice), truth end-state checks (for MJev), revisable planner; S2 fault wrappers | `RuleBasedInteraction`, `KeywordChooser`, `RewriteReviser`, `TruthVerifier`, `RevisableSequence`, `SlowDecider`, `FailingDecider` |
| [`harness/recording.py`](../executors/robocasa-pi052/harness/recording.py) | Per-call logs by wrapping `Policy`, `CommitPolicy`, `Decider` (core unchanged) | `RecordingPolicy`, `RecordingCommit`, `RecordingDecider`; `InteractionCases`: each question a Jev decision asked saved as a calibration case (+ `labels.jsonl`) |
| [`harness/invariants.py`](../executors/robocasa-pi052/harness/invariants.py) | Run invariants: no stale action, irreversible switches (from the robot's `ActionSpec`) held in review / hold, monotonic activation, changes answered, no violations | `check`, `Run` |
| [`harness/viewer.py`](../executors/robocasa-pi052/harness/viewer.py) | Live MuJoCo viewer as a mirror process (shared-memory state, own `mjpython`) | `MirrorPublisher` |
| [`harness/video.py`](../executors/robocasa-pi052/harness/video.py) | Run video: three cameras + decision panel, encoded after the run | `VideoRecorder` |
| [`campaign.py`](../executors/robocasa-pi052/campaign.py) | Test campaign: scripted (happy / faulty), fuzz and live users talking to the v1 loop, each run judged per line and as a whole; report.md / .json | `CASES`, `fuzz_cases`, `run_case`, `cmd_live`, `cmd_say`, `cmd_report` |
| [`harness/users.py`](../executors/robocasa-pi052/harness/users.py) | Simulated users: intents, phrasings, typos, timing; replayable fuzz sessions | `Line`, `FuzzUser`, `say`, `noisy` |
| [`harness/judge.py`](../executors/robocasa-pi052/harness/judge.py) | Per user line: the state it was heard in, the answer, allowed or not (pass / unsafe / safe miss); per run summary | `judge_run`, `judge_lines`, `allowed` |
| [`harness/live_view.py`](../executors/robocasa-pi052/harness/live_view.py) | What a live user watches: cameras + status written atomically during the run | `LiveView` |
| [`gpu/hook.sh`](../executors/robocasa-pi052/gpu/hook.sh), [`gpu/hub.py`](../executors/robocasa-pi052/gpu/hub.py) | π0.5² component hook for `scripts/gpu_setup.sh` (gated tokenizer on the Mac, weights on the box) | |
| [`calibrate_monitor.py`](../executors/robocasa-pi052/calibrate_monitor.py) | Fits the execution monitor on recorded nominal runs (refuses runs behind real time or with mixed policies); prints gain, exercised directions, threshold, false alarms / min | `main` |
| [`calibrate_jev.py`](../executors/robocasa-pi052/calibrate_jev.py) | Fits one question's `Calibration` on a fit set and ships it only if the held-out set's decided accuracy reaches the target (re-scored by OneJev on the GPU) | `labelled`, `decided_accuracy`, `main` |
| [`label_jev.py`](../executors/robocasa-pi052/label_jev.py) | Terminal labelling of recorded Jev cases: opens the cameras, shows what was said and the legal options, writes the chosen act into `labels.jsonl` | `main`, `rows`, `save` |
| [`done_check_ab.py`](../executors/robocasa-pi052/done_check_ab.py) | MJev vs OneJev on "is this step done?": `build` sets kitchen states and labels them with the task's own completion tests, saves simulator masks (Mac), `score` asks both checkers (GPU) | `build`, `score`, `metrics` |
| [`harness/frame_check.py`](../executors/robocasa-pi052/harness/frame_check.py) | Camera-frame integrity during a run: noise / blank frames make the run invalid (not scored) | `FrameCheck`, `frame_problem` |
| [`history.py`](../executors/robocasa-pi052/history.py) | Observation history for policies that condition on past frames: a tick-fed buffer, matched by identity to the observation the runner sends; frames as `<cam>@k` images + stacked states | `FrameHistory`, `HistoryPolicy` |
| [`sam_check.py`](../executors/robocasa-pi052/sam_check.py) | SAM 2.1 against simulator truth: `frames` (masks from correct points, MJev pointing, 2D inside rule on still frames), `track` (a recorded live run streamed through tracking) | `score`, `track`, `track_report` |
| [`harness/tracking.py`](../executors/robocasa-pi052/harness/tracking.py) | Simulator truth masks and 2D geometry; `FrameRecorder` (cameras + state during a live run, scored afterwards) | `truth_masks`, `inside_share`, `FrameRecorder` |
| [`gpu_suite.py`](../executors/robocasa-pi052/gpu_suite.py) | The v1 GPU test suite in one sequential command: π0.5² baseline, K-sample latency / diversity / memory, OneJev latency + live Jev runs (calibration cases recorded), MJev checks vs truth, commitment ablation (T8a) + gate run; suite.json / suite.md | `STEPS`, `realtime`, `main` |
| [`embodiment.py`](../executors/robocasa-pi052/embodiment.py) | PandaOmron `ActionSpec` (12-D; deltas hold at 0; gripper and base mode kept; gripper irreversible); monitor plug-in: end-effector position vs OSC position + torso commands | `PANDA_OMRON`, `PandaOmronEffect`, `ARM_DIMS` |
| [`serve.py`](../executors/robocasa-pi052/serve.py) | π0.5² service on a GPU: `Pi052Policy` on the core's `PolicyService` behind `Endpoint` (admission, replay cache) on `HttpServer` (loopback) and, with `--h3-port`, `H3Server`; empty subtask → its own planner, planned once per request; K draws in one forward batch (`Draws`); `--rtc H` real-time chunking on the request's continuation (switch snapping: `embodiment.py` `to_action_space`) | `Pi052Policy.act_samples`, `Pi052Policy.guidance`, `Draws`, `main` |
| [`tasks.py`](../executors/robocasa-pi052/tasks.py) | Authored sequences and per-subtask truth for 3 diagnosis tasks (evaluation only); budget rationale in [`features/history/v2-levers.md`](features/history/v2-levers.md) §7 | `TaskSpec`, `_picked`, `_placed` |
| [`drive.py`](../executors/robocasa-pi052/drive.py) | π0.5² on LeRobot's own language runtime (own planner or fixed subtask); history: [`features/history/v2-pi052-executor.md`](features/history/v2-pi052-executor.md) | `load_policy`, `main --planner` |
| [`workflow.py`](../executors/robocasa-pi052/workflow.py) | v2 chunk-synchronous live runner (arms B/D/E, uses `legacy_v2`) | `main`, `LiveView`, `service_addresses` |
| [`extract_subtask_vocab.py`](../executors/robocasa-pi052/extract_subtask_vocab.py) | π0.5²'s training subtask strings and timelines (E8; [`features/history/v2-pi052-executor.md`](features/history/v2-pi052-executor.md)) | |
| [`kaggle/kaggle_run.py`](../executors/robocasa-pi052/kaggle/kaggle_run.py), [`kaggle/push.py`](../executors/robocasa-pi052/kaggle/push.py) | Kaggle 2×T4 bring-up (serve / armd / smoke), bundle, push, service addresses, pull (v2; [`features/history/v2-pi052-executor.md`](features/history/v2-pi052-executor.md)) | `MODE`, `services_stages`, `push`, `service` |

### `services/` (S2 model services, own envs), `scripts/`, and `executors/robocasa/` (v2 G0)
| File | What it does |
|---|---|
| [`services/mjev/serve_mjev.py`](../services/mjev/serve_mjev.py) | MJev-Spatial-4B service: half-precision trunk (bf16 on A100), fp32 readout, exact `PatchLinear` patch embedding; HTTP/1.1 keep-alive, threaded |
| `services/onejev/` | OneJev 1.6 4B via `qev serve` (PyTorch backend), own env; loopback only (qev has no auth) |
| [`executors/robocasa-xr1/serve.py`](../executors/robocasa-xr1/serve.py), [`xr1_inputs.py`](../executors/robocasa-xr1/xr1_inputs.py) | Xiaomi-Robotics-1 RoboCasa365 checkpoint (pinned `config/xr1_weights.txt`) as a remote S1 on our wire (never its pickle socket); inputs ported from its evaluation client (4 frames 2 ticks apart, 0.95 crop, end-effector-first 14-value state); own env (torch 2.8, transformers 4.57.1, flash-attn 2.8.3 from its sdist, fetching the prebuilt kernels; readiness imports them) |
| [`services/sam2/serve_sam2.py`](../services/sam2/serve_sam2.py) | SAM 2.1 Hiera Tiny (transformers, pinned `config/sam2_weights.txt`): `/segment` and streaming `/track/*` (session state on CPU, idle sessions closed, errors → 500); own env, GPU hook |
| `services/*/gpu/hook.sh`, `executors/*/gpu/hook.sh` | Component hooks for `scripts/gpu_setup.sh`: OneJev, MJev, SAM 2; π0.5², XR-1 |
| `scripts/gpu_setup.sh`, `scripts/gpu_box.sh`, `scripts/hf_pins.py` | One command from the Mac: fresh GPU box → driver, uv, code, vendors at pins, envs, pinned weights, systemd units, tunnels, one service JSON per box (`docs/features/gpu-setup.md`) |
| [`check_precision.py`](../services/mjev/check_precision.py), [`speed_check.py`](../services/mjev/speed_check.py), [`profile_mjev.py`](../services/mjev/profile_mjev.py), [`patch_embed_check.py`](../services/mjev/patch_embed_check.py), [`make_testdata.py`](../services/mjev/make_testdata.py) | Precision vs fp32, speed levers, profile, patch-embedding fix, test scenes (E11–E12) |
| [`executors/robocasa/`](../executors/robocasa/rollout.py) | SmolVLA env: `checkpoint.py`, `rollout.py`, `check_policy_load.py`, `capability_table.py` |

---

## 5. Who writes what

| Object | Written by | Read by |
|---|---|---|
| `UserInstruction` | user / scenario | `StateOwner` (revision + revoke) |
| `ObservationSnapshot` | `StateOwner.request_decision` | `DecisionWorker` → decider |
| `Verdict` → `DecisionProposal` | decider → `DecisionWorker` | `StateOwner` (accepts, or refuses with a reason) |
| `AcceptedGoal`, `Revoke`, lease renewal | `StateOwner` | `ExecutionRuntime`; `S1Runner` (via `target()`) |
| `ActionChunk` | S1 `Policy` (π0.5² service) | `S1Runner` |
| `TimedChunk` | `S1Runner` (chunk + command id + observation time) | `ExecutionRuntime.submit` |
| `ExecutionAck` | `ExecutionRuntime` (`seq`; `rejected` = command, `chunk_rejected` = one chunk) | `StateOwner` (active record) |
| action per tick, `TickResult` | `Guard.step` | backend (robot / simulator), logs |
| `Capture` | `ControlLoop` (latest-value slot) | `S1Runner`, `StateOwner` → snapshots |
| `SimGroundTruth`, `tasks.py` truth | simulator | evaluators and baselines only, never model input |

---

## 6. Extension points

**A robot** (any arm): a `rojev.robot.Robot` needs nothing else.

```python
from rojev.control import ControlLoop, ExecutionRuntime, Guard, RobotBackend
from rojev.robot import joint_target_spec

backend = RobotBackend(robot, task="pick up the cube")
spec = joint_target_spec(robot, switches={"gripper": 0.8}, irreversible=("gripper",))
runtime = ExecutionRuntime(spec, epoch=1, last=backend.current_command())
loop = ControlLoop(backend, Guard(runtime), spec)
```

| To add | Where |
|---|---|
| Robot | a `rojev.robot.Robot` (must pass `tests/test_robot_contract.py`) → `RobotBackend` + `joint_target_spec`; or any `Backend` (observe / step) with its own `ActionSpec` (like `executors/robocasa-pi052/embodiment.py`) |
| S1 policy | a `rojev.system1.Policy` (local) or a service speaking `system1/wire.py` (remote) |
| S2 decider (Jev, a new arm) | a `rojev.system2.Decider`: `decide(ObservationSnapshot) -> Verdict`; deployable in `system2/`, truth-reading baselines in `eval/` |
| New message | the right `protocol/` module (`commands` for execution, `decisions` for S2), with `epoch` |

---

## 7. Configuration and pins

| File | What it holds |
|---|---|
| [`pyproject.toml`](../pyproject.toml) / `uv.lock` | SDK env: `qev` editable from `vendor/OneJev`; Phoenix in `dev`; pytest `testpaths = tests` |
| `executors/*/pyproject.toml`, `services/*/pyproject.toml` | Separate envs (RoboCasa/LeRobot, MJev, OneJev) whose pins conflict with the SDK and each other |
| [`config/so101.yaml`](../config/so101.yaml) | SO-101 values the MJCF lacks, each with its source |
| `config/*.commit`, `config/onejev_weights.txt`, `config/mjev_weights.txt`, `config/toolchain.txt` | Pinned vendor sources, OneJev 1.6 4B and MJev-Spatial-4B weight revisions, host toolchain |
| [`Makefile`](../Makefile) | `phoenix`, `qev`, `sim`, `sim-viewer`, `test` |
| `graphify-out/` (git-ignored) | Code-only Graphify graph (`graphify extract . --code-only`; refresh with `graphify update .`) |

---

## 8. Tests and tools

| File | Covers |
|---|---|
| [`tests/test_architecture.py`](../tests/test_architecture.py) (3) | Package layering: every import points down the stack; nothing depends on `legacy_v2`; no loose modules |
| [`tests/test_control_execution.py`](../tests/test_control_execution.py) (33) | Execution runtime + guard: activation, skip, revoke vs queued release, irreversible gating, rejections, ACK order, leases (also while starved), transitions and takeover, rate check, blending, watchdog, owner loss (commands dropped, a new one moves), runtime error, PandaOmron plug-in, starvation counted only for an active moving command, blends stay in bounds and are not violations |
| [`tests/test_system1_commit.py`](../tests/test_system1_commit.py) (43) | Commitment policies: unit rules, randomized invariants (300 candidate sets × 5 policies × 2 robot kinds), factory strictness; deterministic reach-and-grasp through the real runtime + runner: latency-derived floor and lead time, zero starvation across latencies and policies, outlier sample, disputed grasp (10 seeds), floor-ignoring policy starves, unbatched samples; even-K ties, first-step tie with and without the robot's switch, plurality is not a majority, reference majority implementation |
| [`tests/test_system1_monitor.py`](../tests/test_system1_monitor.py) (52) | Execution monitor on a simulated lagged planar arm through the real runtime + loop: full recording from the pre-action state, lag-aware gain fit, online = offline scores, conformal false-alarm guarantee on held-out runs (10 calibrations), blocked arm (20 seeds) and pushed arm (20 seeds) detected within the window, unasked block raises nothing, one deviation per excursion, unexercised directions, resolution floor, refusals |
| [`tests/test_transport.py`](../tests/test_transport.py) (16) | HTTP/3 and HTTP/1.1 (behind a lossy TCP proxy) on loopback: one kept connection, `info` passed through, wrong token / unpinned certificate refused, retries never run a request twice (`/act`, `/health`), stale-connection retry, admission before the body (401 / 413, handler never runs, connection usable); wire v2 K samples, codecs |
| `executors/robocasa-pi052/tests/test_harness.py` (7) | Harness logic without the simulator: interaction rules (resume from the owner's hold), Jev with keyword scores, revisable planner, invariant checks, calibration cases saved and labelled |
| `executors/robocasa-pi052/tests/test_scenarios_scorecard.py` (5) | Scenario changes declared at their `--say` / `--block-at` times, goal texts in their utterance, arm label, mismatch raises, `--score-only` on a synthetic matrix |
| `executors/robocasa-pi052/tests/test_label_jev.py` (1) | Labelling CLI writes the chosen act, skip leaves it empty |
| `executors/robocasa-pi052/tests/test_campaign.py` (5) | Fuzz sessions replayable per seed, typo noise, allowance by state, state read before the answer, unsafe answers |
| `executors/robocasa-pi052/tests/test_serve.py` (3) | π0.5² service routing behind `Endpoint` + `HttpServer` with a stand-in policy; `Draws` K-row batch with a fake model |
| [`tests/test_system1_runner.py`](../tests/test_system1_runner.py) (14 + 2 `timing`) | Control loop + S1 runner, deterministic (simulated time): continuous supply, actions due during inference skipped (exact index), lead time, too-slow policy, revoke incl. a chunk in flight, policy errors, instruction conditioning, refill-now only for the supplied command, floor over chunk counted, S1-planner subtask switch posted as a step event; **SO-101 MuJoCo lockstep under the same runtime**; `timing`: pacing and SO-101 real-time budgets (one control period) |
| [`tests/test_owner.py`](../tests/test_owner.py) (41) | Owner rules (valid proposal, seven kinds of stale or unknown proposal, review vs revoke on an instruction change, reaffirm without a pause, review timeout, CONTINUE renews, rejected chunk vs command, COMPLETE, deviation of the active command vs stale deviations; lanes: routing, planning waits while interaction pends, superseded requests, unanswered change holds on failure / timeout / refusal and is decided on the recheck, one-lane mode; keyframes: recorded with the decision's observation, not for aborts / waits / retries / unstarted steps, kept across deviation and revision; worker failure; holds through heartbeats, resume, recheck, one-lane race; dialogue + utterances; step events + D8 ordering; dropped commands; reaffirm always immediate; continue `renewed`) and live scripted / native (review and revoke) / oracle / two-lane arms |
| [`tests/test_system2_jev.py`](../tests/test_system2_jev.py) (17) | Jev v1 with a stub chooser: candidates per situation, budgets, act → operation, abstention, dwell, wait budget, temperature and margin fit, unanswered hold decided on recheck, model inputs; owner + Jev stop → heartbeats → go on |
| [`tests/test_eval_scorecard.py`](../tests/test_eval_scorecard.py) (27) | Scorecard: validation, goal-swap metrics, answer rule (heartbeats never count), act mapping, pause, ask precision, unanswered hold, edge cases, ECE, aggregation and renderers |
| [`tests/test_system2_verify.py`](../tests/test_system2_verify.py) (5) | Step watcher confirmation streaks, one event per command, no check without a step, failures counted; MJev request shape |
| [`tests/test_robot_contract.py`](../tests/test_robot_contract.py) (18, mujoco + mock) | Robot contract, cameras, calibration, time modes |
| [`tests/test_robot_sim_world.py`](../tests/test_robot_sim_world.py) (4) | World composition |
| `tests/test_legacy_v2_*.py` (11) | Quarantined v2 arms, candidate filter |
| [`tests/test_install.py`](../tests/test_install.py) (10) | Install gate ([`features/history/phase0-install.md`](features/history/phase0-install.md)) |
| [`examples/scripted_trajectory.py`](../examples/scripted_trajectory.py) | SO-101 scripted trajectory through `rojev.robot`, traced; optional OneJev checkpoints with a truth evaluator |
| [`playground/onejev_bench/`](../playground/onejev_bench/run_bench.py), [`playground/candidates/`](../playground/candidates/coverage.py), [`playground/archive/`](../playground/archive/e3_prompt_variants.py) | Experiments E13, E9/E9b, E3/E6 (`templates.py` to be adopted for Jev v1) |

```bash
uv run pytest -q          # SDK tests (217; deterministic)
cd executors/robocasa-pi052 && uv run python scenarios.py   # live matrix (19 cases; --viewer, --trace, --policy remote)
uv run pytest -m timing   # wall-clock budgets, run alone (2)
make phoenix              # traces at http://localhost:6006
make sim                  # SO-101 scripted trajectory (ARGS="--time-mode realtime")
cd executors/robocasa-pi052 && uv run python realtime.py --task MicrowaveThawingFridge --arm oracle --policy wiggle
```

---

## 9. Verified

| Path | Status |
|---|---|
| Package layering (no upward or sideways imports, legacy quarantined) | ✓ `test_architecture.py`, 2026-10-20 |
| Execution runtime + guard rules | ✓ 23 tests |
| Control loop + S1 runner; same runtime on the SO-101 MuJoCo backend | ✓ 8 tests |
| State owner + S2 worker + baselines (scripted, oracle, native) | ✓ 10 tests; live RoboCasa with the stand-in policy and a change of mind: real-time factor 1.00, max lag ≤ 9.6 ms, 0 errors, 0 violations |
| SO-101 sim realtime mode | ✓ tests + scripted run |
| π0.5² service on Kaggle T4 + v2 workflow (arms B/D/E) | ✓ 2026-10-18 (chunk-synchronous); v2 workflow smoke-run after the restructure ✓ |
| MJev service fp16 (1.3 s / 3 questions on T4) | ✓ E12f |
| π0.5² live in the v1 loop (incl. its own-planner mode) | ✗ needs GPU access |
| Review mode + decelerating hold (step 4.1) | ✓ tests; live: no pause after a change of mind vs 0.3 s stop with `revoke` |
| Commitment policy (step 4.2) | ✓ tests + simulated grasp; live stand-in; **not validated with π0.5²** (T8a) |
| Execution monitor (step 4.3) | ✓ tests; live RoboCasa: blocked arm detected in 0.3 s; calibrated on stand-in runs only (recalibrate with π0.5²) |
| Two lanes + progress keyframes (step 4.4) | ✓ tests; live RoboCasa: change of mind answered by the interaction lane at once, 40/40 ticks in motion after it, keyframe recorded |
| Fault suite; Jev v1; tracker | ✗ |
