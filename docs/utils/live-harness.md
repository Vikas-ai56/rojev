# Live harness: RoboCasa runs, scenario matrix, spoofs

Code: `executors/robocasa-pi052/` — `realtime.py` (one run), `scenarios.py` (matrix), `harness/` (invariants, skills,
deciders, recording, frame_check, tracking, viewer, video, live_view), `embodiment.py` (PandaOmron `ActionSpec` +
effect model), `tasks.py` (authored steps + truth checks). Tests: `executors/robocasa-pi052/tests/` (36 passed
2026-10-24; `test_harness.py` 9, `test_scenarios_scorecard.py` 5).
Related: [`v3-runtime.md`](../features/v3-runtime.md) · [`s1-supply-and-commitment.md`](s1-supply-and-commitment.md) ·
[`execution-monitor.md`](../features/execution-monitor.md) · [`remote-models.md`](../features/remote-models.md) · [`system2-map.md`](../features/system2-map.md) · campaign: `executors/robocasa-pi052/campaign.py`.

## Rule: the core never bends to the simulator (decisions 2026-10-21)
- Stand-ins, faults, recording, viewer, video and the matrix live in the executor and use only core interfaces
  (`Policy`, `CommitPolicy`, `Decider`, `Backend`, `TickRecord`).
- Every stand-in is declared a SPOOF in its module docstring; physics stays real (no privileged attach).

## `realtime.py`: one run
- Wiring: `RoboCasaBackend` (LeRobot `RoboCasaEnv`) → `ExecutionRuntime(PANDA_OMRON)` seeded still, gripper open →
  `Guard` → `ControlLoop`; `S1Runner` over `RecordingPolicy` + `RecordingCommit`; planning `DecisionWorker` (+ optional
  interaction worker) over `RecordingDecider`; `StateOwner`; optional `ExecutionMonitor` (deviation → owner + refill) and
  `StepWatcher`. Per-part detail: [`system2-map.md`](../features/system2-map.md) §1.
- `RoboCasaBackend.read(fn)` evaluates the kitchen under a lock between two steps; latches RoboCasa's own success
  (`success_t`); `--block-at` zeroes the arm dims inside `step` (fault).
- Time modes (decision 2026-10-23):
  - **lockstep** (`--lockstep`) = the simulator evaluation mode: one thread runs the owner, S1 supply and one control
    tick, and no tick while either S2 lane is busy or its answer is unhandled, so model computation takes zero simulated time; `--seconds` is simulated;
  - **lockstep with model delay** (`--lockstep --model-delay`, `harness/model_delay.py`) = honest scoring: each S1
    request and S2 decision costs simulated time equal to its measured wall time, simulator stepping costs none; the
    runner and lanes run on threads, the loop never ticks past an open call's start + wall time so far, and an answer
    returns once simulated time reaches start + its wall time (`model_delay.jsonl`). Latency, deadlines and slow-model
    faults then act as on hardware, independent of simulator speed (decision 2026-11-06);
  - **real time** (without `--lockstep`) = reported alongside: control at 20 Hz on the wall clock, every part on its
    own thread; measures the latency a robot will have (E24: XR-1 over the WAN cannot keep up).
- Setup heap frozen (`gc.collect(); gc.freeze()`) before threads start (full GC pass 45–96 ms otherwise).

| Flags | Default | Meaning |
|---|---|---|
| `--task`, `--seed`, `--seconds` | LoadCondimentsInFridge, 1000, 20 | run length: simulated in lockstep, wall in real time |
| `--arm` (+ `--subtask`, `--max-commit`) | `fixed` | `fixed` · `scripted` · `oracle` · `revisable` · `revisable-truth` · `native` · `commit` (25 s) |
| `--policy`, `--service`, `--s1` | `skills`, —, `xr1` | `skills` scripted S1 · `remote` (`--s1 xr1\|pi052`; xr1 gets 4 frames 2 ticks apart) · `wiggle` plumbing |
| `--latency`, `--sample-noise` | 0.3 s, 0 | skills / wiggle: simulated inference time, per-sample noise |
| `--interaction`, `--interaction-timeout` | `none`, 1.0 s | `none` (one lane) · `continue` · `rules` (keyword spoof) · `jev` |
| `--chooser`, `--reviser`, `--skills`, `--max-asks`, `--jev-calibration` | `onejev`, `canonical` if remote else `rewrite`, —, 2, — | Jev: act scorer (`keywords` = spoof); REVISE = trained sentence / rewritten step; extra trained sentences (run instruction always included); question budget; `{act, task}` JSON |
| `--hold-recheck S`, `--record-interaction DIR` | off | ask the lane again while a hold stands; save Jev's questions as calibration cases |
| `--verify`, `--verify-every`, `--done-threshold`, `--confirmations` | `none`, 1.0 s, 0.7, 2 | step events from `mjev` / `onejev` checks or `truth` (spoof) |
| `--interaction-delay`, `--interaction-fail` | 0, off | S2 faults on the interaction lane (need a lane) |
| `--say T:TEXT` (repeatable), `--instruction` | —, environment's sentence | scripted user line at T s; the overall task |
| `--interactive`, `--input-file F` | off | typed lines (terminal, or appended to F) become instructions; `quit` ends |
| `--live-view S` | off | `live/now.png` + `status.json` every S simulated s (for a live user) |
| `--max-wall S`, `--end-after-success S` | off | end on wall time; end S simulated s after first success |
| `--lockstep` | off | see time modes |
| `--model-delay` | off | with `--lockstep`: measured model delay added back (see time modes) |
| `--frame-check-every S`, `--record-frames S` | 1.0, off | frame integrity sampling; cameras + sim state for `sam_check.py track` |
| `--on-change`, `--review-s`, `--stop-s`, `--refill`, `--blend`, `--lease`, `--decide-every` | `review`, 1.0, 0.3, 1.0, 0.2, 3.0, 1.0 (0.8 × lease with step events) | owner / runtime parameters |
| `--commit-policy`, `--speculative-predict --samples K`, `--tolerance` | `fixed`, off | commitment (an inapplicable parameter is an error) |
| `--execution-monitor M`, `--monitor-record OUT`, `--block-at T --block-s D` | off, off, off / 2 s | monitor model / nominal recording / blocked-arm fault |
| `--viewer`, `--video`, `--trace PROJECT`, `--out` | off | mirror window · `view.mp4` · Phoenix export after the run · run directory |

- Refused combinations: `remote` without `--service`; S2 faults without a lane; `onejev` chooser or verifier without
  `--service`; `commit` without a verifier and `remote`; `--record-interaction` without `jev`.
- Outputs (`results/runs/realtime/<id>/` unless `--out`): `config.json`, `ticks.jsonl`, `owner.jsonl`,
  `s1_requests.jsonl`, `commits.jsonl`, `decisions.jsonl` (both lanes: op, reason, `scores`, `task`, `decide_ms`),
  `utterances.jsonl`, `typed.jsonl` (interactive), `verify_checks.jsonl` (`--verify`), `frames/`, `live/`, `view.mp4`,
  `summary.json` (timing, commands, refused, keyframes, lanes, ACKs, holds, said, step events, S1 stats, monitor,
  `frame_check`, `ended`: seconds · wall · success · quit · error).
- `--trace` builds spans from the records after the run (`run_trace.export_run`); nothing is traced on the control path.

## Integrity and measurement
- `harness/frame_check.FrameCheck` (always on): noise (neighbour-pixel difference > 25) or blank (std < 3) frames →
  summary `frame_check.valid` false, warning, `frames_real` fails, the run is not scored. Thresholds from 963 real
  frames (≤ 7.4, std ≥ 26.8) and the 2026-10-22 corrupted run (render-context bug, fixed in `render_context.py`).
- `--record-frames` (`harness/tracking.FrameRecorder`): nothing rendered live (blank inside the live process);
  `sam_check.py track` restores each state in a fresh process and scores SAM 2.1 against truth masks.

## `harness/invariants.py`: what every run must keep
- `no_stale_action`: no chunk action of a command after its `revoked` / `expired` / `dropped` ACK.
- `switches_held`: no irreversible switch (`ActionSpec`; PandaOmron: gripper) changes under review or holding.
- `active_monotonic`: activated command ids strictly increase.
- `changes_answered`: every instruction gets a command, reaffirm or continue before the next one (if ≥ 2 s remain).
- `no_violations`: no incoming action clamped by the bounds.
- `frames_real`: every sampled camera frame was real.

## `scenarios.py`: the matrix (20 cases)
- Each case = one `realtime.py` subprocess, MicrowaveThawingFridge seed 1000, `revisable` arm, real time (scripted S1
  on the Mac); checked against the invariants and its own expectations.
- `uv run python scenarios.py [--only NAME…] [--viewer] [--policy remote --service S.json] [--monitor M.json]
  [--trace P] [--no-video]`; `--score-only DIR` re-scores a past matrix. Report: `results/scenarios/<stamp>/report.json`;
  exit 1 if a case fails (scores never change the exit code).
- Scorecard feed ([`scorecard.md`](scorecard.md)): each case declares one `Change(kind, expected, goal, old_goal)` per
  `--say` / `--block-at`, paired in order (mismatch = error). Arm = `--interaction`, else `--arm`, plus `+revoke`,
  `+s2-slow`, `+s2-fail`. A run is scored only if its summary schedule equals the case's and its frames are valid.

| Case | Setup | Expectations | Declared changes (t: kind → act) |
|---|---|---|---|
| nominal | 16 s | first two steps in order, a keyframe, 0 starved | — |
| speculative / -phase | K = 3, noise 0.01 (+ phase policy) | a consensus rule fired / a commitment shorter than the chunk; 0 starved | — |
| harmless | rules, "nice work" 5 s | interaction-lane `continue`, no new command from it; moving 2 s after | 5: harmless → continue |
| revise | rules, "…on the plate" 4 s, "no, …microwave" 14 s | place step → plate; in-step revision → microwave by the interaction lane | 4: goal_swap → continue · 14: goal_swap → revise |
| stop / stop-stays | rules, "stop" 6 s / 5 s | ABORT; holding after (stop-stays: standing hold, no moving command, to the end) | stop → abort |
| wait-resume / jev-wait-resume | rules / jev + keyword chooser; "wait" 5, "go on" 9 | GUARDED_WAIT, then the interrupted step (jev: two ACKs) | 5: pause → wait · 9: resume → revise |
| ask / jev-ask-answer | rules / jev; "put it over there" 5, "on the plate" 9 | ASK (jev: `clarify` said), then SWITCH_TO | 5: ambiguous → ask · 9: goal_swap → revise |
| jev-wait-budget | jev, `--hold-recheck 2`, vague line 5 s | asked once, `waiting_for_user` once, holding to the end (D7) | 5: ambiguous → ask |
| revoke-baseline | rules, `--on-change revoke`, "nice work" | revoke recorded, hold ticks 5–7 s | 5: harmless → continue |
| one-lane | no interaction lane, "wait" | the planning lane answers | 5: pause → wait |
| s2-slow | rules, delay 1.6 s > timeout 1.0 s | timeout, `unanswered` said, planning never answers, holding | 5: pause → wait |
| s2-fail | rules, decider raises | `decision failed`, unanswered hold within 1 s, planning never answers, holding | 5: pause → wait |
| s1-slow | S1 latency 0.9 s | floor ≥ 37, 0 starved | — |
| step-events | `revisable-truth` + `--verify truth`; needs `--policy remote` | a step event; a keyframe with the checker's evidence | — |
| blocked | rules stand-in, monitor on, arm blocked 6–8 s; needs `--monitor` | a deviation, RETRY from the interaction lane (the stand-in's own act) | 6: deviation → revise |
| jev-blocked | jev + keyword chooser, same block | a deviation, GUARDED_WAIT from the interaction lane, never a retry (Jev offers continue / wait on a deviation) | 6: deviation → wait |

## Spoofs (declared) and faults
- `harness/skills.py` `SkillScriptPolicy`: reads object / fixture poses, grasp and contact from the simulator; a
  proportional controller planned forward from the measured hand (50 OSC-delta actions); skills pick, place, press
  start / stop, else hold still. Limits: orientation never steered, the base never drives (`info["reach_m"]`), grasps
  press into contact.
- `harness/deciders.py`: `RuleBasedInteraction` (keywords → ABORT / GUARDED_WAIT / resume / ASK / revise; deviation →
  RETRY) for Jev's acts; `KeywordChooser` for OneJev's scores inside the real `JevInteraction`; `RewriteReviser` for
  the trained-task choice (the scripted S1 needs step text); `TruthVerifier` for end-state checks; `RevisableSequence`
  for a re-planning planner; `SlowDecider` / `FailingDecider` inject S2 faults.
- Faults: S1 latency (`--latency`), S2 delay / failure, blocked arm.

## Recording, viewer, video
- `harness/recording.py`: wrappers append one dict per call (robot clock); `InteractionCases` saves Jev's questions as
  calibration cases. `harness/viewer.py`: the viewer is its own `mjpython` process reading shared memory (GLFW and
  Cocoa both need the main thread); physics only. `harness/video.py`: raw frames in the run, H.264 after it.

## Measured
- **2026-10-25** (`results/scenarios/20261025-hygiene`; scripted S1, real time, no video, monitor `results/monitor/skills-20261021`; after the lockstep owner
  fix and the scenario updates): **17/19 run cases pass**, `step-events` skipped. Real-time factor 1.000–1.001,
  0 starved, 0 violations. `harmless` now passes (interaction-lane `continue`, no command from it).
- `blocked` and `jev-blocked` fail the same way: no deviation raised (max score 2.15 vs threshold 2.54, as on
  2026-10-22); scripted grasps press into contact, so the nominal calibration already holds blocked-like motion. Their
  decision expectations are unexercised until the monitor gets contact-aware calibration (no threshold tuning).
- `step-events` runs only with `--policy remote`: the scripted S1 has completed no step in any recorded run.
  Earlier matrices (2026-10-21 13/14, 2026-10-22 17/19, 2026-10-24 16/17) are superseded by the 10-25 run.
- Bugs the matrix found, fixed at their owner: runtime blend pushed actions past a bound (core: incoming action checked,
  blend projected); in-run video encoding stalled the loop (harness: encode after); `RemotePolicy` dropped chunk `info`.

## Not validated
- The matrix with a model S1 (`--policy remote`); model runs with users are in `executors/robocasa-pi052/campaign.py`.
- `blocked` with a contact-aware monitor; `step-events` with an S1 that completes steps.
