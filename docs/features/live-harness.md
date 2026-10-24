# Live harness: RoboCasa runs, scenario matrix, spoofs (step 5a, 2026-10-21)

Code: `executors/robocasa-pi052/` — `realtime.py` (entrypoint), `scenarios.py` (matrix), `harness/` (invariants, skills, deciders, recording, viewer, video), `embodiment.py` (PandaOmron `ActionSpec` + effect model), `tasks.py` (authored steps + truth checks). Tests: `executors/robocasa-pi052/tests/test_harness.py` (7), `tests/test_scenarios_scorecard.py` (5; no simulator).
Core it drives: [`v3-runtime.md`](v3-runtime.md). S1 supply / commitment: [`s1-supply-and-commitment.md`](s1-supply-and-commitment.md). Monitor: [`execution-monitor.md`](execution-monitor.md). π0.5² service and transport: [`remote-models.md`](remote-models.md).

## Rule: the core never bends to the simulator (decisions 2026-10-21)
- Scripted S1, keyword interaction acts, revisable planner, fault injection, recording, viewer, video and the matrix live here, using only core interfaces (`Policy`, `CommitPolicy`, `Decider`, `Backend`, `TickRecord`).
- Every stand-in is declared as a SPOOF in its module docstring; physics stays real (no privileged attach).

## `realtime.py`: one live run
- Wires: `RoboCasaBackend` (LeRobot `RoboCasaEnv`, pretrain split) → `ExecutionRuntime(PANDA_OMRON, stop_s, blend_s)` seeded with `START` (still, gripper open) → `Guard` → `ControlLoop(realtime=True)`; `S1Runner` over `RecordingPolicy` + `RecordingCommit`; planning `DecisionWorker` (+ optional interaction worker) over `RecordingDecider`; `StateOwner`; optional `ExecutionMonitor` (deviation → `owner.post` + `runner.refill_now`). `--say` utterances posted on the robot clock.
- `RoboCasaBackend`: `read(fn)` evaluates the kitchen under a lock between two steps; latches RoboCasa's `is_success`; `--block-at` zeroes the arm dims inside `step` (fault); mirrors each step to the viewer.

| Flag | Default | Choices / meaning |
|---|---|---|
| `--task`, `--seed`, `--seconds` | LoadCondimentsInFridge, 1000, 20 | authored sequences in `tasks.py`: MicrowaveThawingFridge, MoveFridgeToFreezer, LoadCondimentsInFridge |
| `--arm` | `fixed` | `fixed` one command (`--subtask`) · `scripted` · `oracle` · `revisable` (time budget) · `revisable-truth` · `native` |
| `--policy` | `skills` | `skills` scripted S1 · `remote` π0.5² (`--service` JSON or `kaggle`) · `wiggle` plumbing only |
| `--latency`, `--sample-noise` | 0.3 s, 0 | skills / wiggle: simulated inference time, per-sample target noise |
| `--interaction` | `none` | `none` one lane · `continue` (`AlwaysContinue`) · `rules` (keyword spoof) · `jev` (`JevInteraction`); `--interaction-timeout` 1.0 s |
| `--chooser`, `--reviser`, `--skills`, `--max-asks`, `--jev-calibration` | `onejev`, `canonical` (remote) / `rewrite` (else), none, 2, none | jev: who scores acts (`keywords` = spoof), REVISE = a trained task sentence / the step rewritten, extra trained sentences (one per line; the run's instruction is always one), question budget, `{act, task}` calibration JSON |
| `--hold-recheck S` | off | ask the interaction lane again while a hold stands (wait budget) |
| `--verify`, `--verify-every`, `--done-threshold`, `--confirmations` | `none`, 1.0 s, 0.7, 2 | step events from `mjev` end-state checks or `truth` (spoof) |
| `--interaction-delay`, `--interaction-fail` | 0, off | S2 faults on the interaction lane (need a lane) |
| `--say T:TEXT` (repeatable), `--on-change` | —, `review` | user instruction at T s; `review` or `revoke` |
| `--review-s`, `--stop-s`, `--refill`, `--blend`, `--lease`, `--decide-every` | 1.0, 0.3, 1.0, 0.2, 3.0, 1.0 s | runtime / owner parameters |
| `--commit-policy`, `--speculative-predict --samples K`, `--tolerance` | `fixed`, off | commitment (strict factory: an inapplicable parameter is an error) |
| `--execution-monitor M.json`, `--monitor-record OUT.npz` | off | score against a fitted model / record a nominal run for `calibrate_monitor.py` |
| `--block-at T --block-s D` | off, 2 s | blocked-arm fault |
| `--viewer`, `--video`, `--trace PROJECT`, `--out` | off | mirror window · `view.mp4` · Phoenix export after the run · run directory |

- Outputs (`results/runs/realtime/<task>_<arm>_<policy>_s<seed>_<stamp>/` unless `--out`): `config.json`, `ticks.jsonl` (per tick: source, mode, cmd, skipped, held, violations, hold reason, review, starved, step / lag ms, action, monitor score), `owner.jsonl` (every owner event), `s1_requests.jsonl`, `commits.jsonl`, `decisions.jsonl`, `utterances.jsonl` (what the robot said, rendered), `summary.json` (real-time factor, lag, step p50 / p95, chunk / hold / starved ticks, commands, refused, keyframes, lane stats, ACK counts (incl. `dropped`), holds, said, step events, verifier stats, truth per step, S1 latency estimate and floor, monitor, violation ticks), `view.mp4` with `--video` (three cameras + command / lane / decision panel, 10 fps).
- `--trace`: `rojev.observability.run_trace.export_run` builds the span tree from the records after the run; nothing is traced on the control path.

- Setup heap frozen (`gc.collect(); gc.freeze()`) before the threads start (research brief 2026-10-22: full GC pass 45–96 ms).

### Frame check (2026-10-23)
- Always on: `harness/frame_check.FrameCheck` samples the camera frames every `--frame-check-every` s (default 1); noise (mean neighbour-pixel difference > 25) or blank (std < 3) frames → summary `frame_check.valid` false, a warning, the `frames_real` invariant fails and `scenarios` does not score the run. Thresholds from 963 real frames (difference ≤ 7.4, std ≥ 26.8) and the 2026-10-22 corrupted run.

### Delay and measurement flags (2026-10-22)
- `--s1 pi052|xr1` (with `--policy remote`): which remote S1; `xr1` wraps it in `history.HistoryPolicy` (4 frames, 2 ticks apart, fed by the tick hook).
- `--lockstep`: simulated time stops while any model thinks: main thread `runner.step()` then `loop.run_ticks(1)`, and no tick while an S2 lane is busy, so S2 deadlines in simulated seconds do not depend on simulator speed (`--seconds` = simulated time). `--interactive` / `--instruction TEXT`: a person types instructions during the run (`typed.jsonl`); `--input-file F` reads them from a file as it grows (a live user agent); `--live-view S` writes `live/now.png` + `status.json` every S simulated seconds; `--max-wall S` ends the run on wall time; `--end-after-success S` ends it S simulated seconds after the task's own check first holds (summary `ended`: seconds · wall · success · quit · error). Diagnostic only: separates π0.5²'s own skill from the cost of real-time execution. Real time stays the default.
- `--record-frames S`: `harness/tracking.FrameRecorder` saves cameras + flattened simulator state every S s to `frames/frames.npz`; nothing rendered live (the offscreen renderer returns blank images inside the live process). `sam_check.py track <run>` restores each state in a fresh process, renders truth masks, streams the frames through SAM 2.1 (`frames/tracks.jsonl`, `track_report.json`).
- `sam_check.py frames <done-check folder>`: SAM 2.1 masks from correct (simulator) points, MJev pointing, 2D "placed inside" on still frames (`sam_report.json`).

## `scenarios.py`: the matrix (19 cases; 5 added 2026-10-22)
- Each case = a `realtime.py` subprocess on MicrowaveThawingFridge seed 1000, `revisable` arm, checked against the five invariants and its own expectations.
- `uv run python scenarios.py` (all, headless, scripted S1) · `--only stop --viewer` · `--policy remote --service S.json` · `--trace PROJECT` · `--monitor M.json` (needed by `blocked`, else skipped) · `--no-video`.
- Report: `results/scenarios/<stamp>/report.json` (per case: ok, invariant findings, expectations, run dir, real-time factor, starved ticks, `scorecard`) + one run directory per case; exit 1 if any case fails (scores never change the exit code).

### Scorecard (T8, 2026-10-22): [`scorecard.md`](scorecard.md)
- `Scenario.changes`: one `Change(kind, expected, goal, old_goal, phase)` per `--say` / `--block-at`, in argument order, declared by the case's design; time = its argument's. `injections()` pairs them (count or trigger mismatch = error); cases without changes are not scored.
- Arm = the case's `--interaction` value, else its `--arm` (`rules`, `jev`, `revisable` for one-lane).
- After the matrix: each case with changes and a recorded run is scored (`score_run`), aggregated per arm; `scorecard.md` + `scorecard.json` beside `report.json`; per-injection scores in the case's report entry; per-arm table printed.
- A run whose summary `say` / `blocked` differs from the scenario's schedule is not scored (`scorecard.skipped` says why, printed as `[score skip]`): past runs of since-edited cases are never scored under the new declaration.
- `--score-only RUN_DIR`: re-scores an existing matrix directory (its `report.json` + run directories), rewrites `report.json` with the scores, runs nothing.

| Case | Changes declared (t s: kind → expected) |
|---|---|
| harmless, revoke-baseline | 5: harmless → continue |
| revise | 4: goal_swap plate ← microwave → continue (current step unaffected, later step rewritten) · 14: goal_swap microwave ← plate → revise |
| stop / stop-stays | 6 / 5: stop → abort |
| wait-resume, jev-wait-resume | 5: pause ("wait") → wait · 9: resume ("go on") → revise (resume = switch_to) |
| ask, jev-ask-answer | 5: ambiguous → ask · 9: goal_swap plate ← microwave → revise |
| jev-wait-budget | 5: ambiguous → ask |
| one-lane, s2-slow, s2-fail | 5: pause ("wait") → wait |
| blocked | 6: deviation → revise (retry) |

| Case | Setup | Expectations |
|---|---|---|
| nominal | authored steps on the time budget, 16 s | first two steps in order, a keyframe, 0 starved |
| speculative | K = 3, noise 0.01 | a consensus rule fired, 0 starved |
| speculative-phase | K = 3 + phase | a commitment shorter than the chunk, 0 starved |
| harmless | rules, "nice work" at 5 s | interaction-lane reaffirm, chunk ticks for 2 s after |
| revise | "actually … on the plate" at 4 s, "no, … microwave" at 14 s | place step goes to the plate; in-step revision to the microwave by the interaction lane |
| stop | "stop" at 6 s | ABORT from the interaction lane, hold ticks for 3 s after |
| wait-resume | "wait" 5 s, "go on" 9 s | GUARDED_WAIT, then the interrupted step resumed |
| ask | "put it over there" 5 s, "on the plate" 9 s | ASK, then a SWITCH_TO after the answer |
| revoke-baseline | `--on-change revoke`, "nice work" | revoke recorded, hold ticks in 5–7 s |
| one-lane | no interaction lane, "wait" | the planning lane answers |
| s2-slow | interaction delay 1.6 s > timeout 1.0 s | interaction timeout, `unanswered` said, planning never answers the change, holding after |
| s2-fail | interaction decider raises | `decision failed`, unanswered hold within 1 s, planning never answers, holding after |
| s1-slow | S1 latency 0.9 s | floor ≥ 37, 0 starved |
| stop-stays | rules, "stop" at 5 s, 15 s | ABORT, a standing hold, no moving command after it, hold ticks to the end (D4) |
| jev-wait-resume | jev + keyword scores, "wait" 5 s, "go on" 9 s | GUARDED_WAIT, interrupted step resumed, two `ack`s |
| jev-ask-answer | jev, "put it over there" 5 s, "on the plate" 9 s | `clarify` said, SWITCH_TO after the answer |
| jev-wait-budget | jev, `--hold-recheck 2`, unanswered vague reference | asked once, `waiting_for_user` once, holding to the end (D7) |
| step-events | `revisable-truth` + `--verify truth`, heartbeat 2.5 s; runs only with `--policy remote` | a step event; a keyframe with the checker's evidence |
| blocked | monitor on, arm blocked 6–8 s | a deviation, RETRY from the interaction lane |

## `harness/invariants.py`: what every run must keep
- `no_stale_action`: no chunk action of a command after its `revoked` / `expired` / `dropped` ACK.
- `switches_held`: no irreversible switch of the robot's `ActionSpec` (PandaOmron: gripper) changes on a tick under review or holding.
- `active_monotonic`: activated command ids strictly increase.
- `changes_answered`: every instruction followed by a command, reaffirm or continue before the next instruction or the end (if ≥ 2 s remain).
- `no_violations`: no incoming action clamped by the bounds.

## Spoofs (declared) and faults
- `harness/skills.py` `SkillScriptPolicy` (scripted S1): reads object / fixture poses, grasp and contact from the simulator; hand-written proportional controller planned forward from the measured hand (50 OSC-delta actions); K samples differ only by target noise; latency is a sleep. Skills: pick, place (fridge shelves, microwave tray, plate), press start / stop; anything else holds still (e.g. "close the microwave door").
- Its physical limits: orientation never steered; the base never drives, so targets beyond ~0.85 m (`info["reach_m"]`) stay unfinished; a 15 cm flat steak is hard to grasp top-down (reach and contact are real); grasps press into contact.
- `harness/deciders.py`: `RuleBasedInteraction` (keywords → ABORT / GUARDED_WAIT / resume / ASK / revise SWITCH_TO; deviation → RETRY; else CONTINUE; resumes the owner's `hold.interrupted`) stands in for Jev's acts; `KeywordChooser` (the same keywords as act probabilities) stands in for OneJev's scores inside the real `JevInteraction`; `RewriteReviser` stands in for π0.5²'s re-planning on REVISE; `TruthVerifier` (tasks.py completion checks as P(done) 1 / 0) stands in for MJev's end-state checks; `RevisableSequence` (authored steps rewritten to the named destination, advanced by budget or truth) stands in for a re-planning planner; `SlowDecider`, `FailingDecider` inject S2 faults. Oracle truth via `tasks.py`.
- Faults: S1 latency (`--latency`), S2 delay / failure, blocked arm (backend).

## Recording, viewer, video
- `harness/recording.py`: `RecordingPolicy`, `RecordingCommit`, `RecordingDecider` implement the wrapped interface and only append one dict per call (robot-clock times); the core runs unchanged.
- `harness/viewer.py` `MirrorPublisher`: robosuite cameras need a GLFW window on the main thread, and `mjpython` (passive viewer) takes the main thread for Cocoa, so the viewer runs in its own `mjpython` process. The sim saves the compiled `.mjb` and copies time, qpos and mocap poses into shared memory each step; the viewer only reads.
  - Fix: `mjpython` replaces dyld's default fallback path, hiding the system zlib some wheels link against → launched with `/usr/local/lib:/usr/lib` restored.
  - Fix: Python 3.12 registers an attached shared-memory block with the viewer's resource tracker (unlinked on exit) → the viewer unregisters it; only the publisher unlinks.
  - Shows physics state only, not robosuite-side visual changes.
- `harness/video.py` `VideoRecorder`: frames kept raw during the run, composed and H.264-encoded in `close()` after it; encoder start-up holds the GIL ~0.2 s (measured 236 ms step, 3 starved ticks, a corrupted S1 latency sample). ~6 MB per second of run.

## `tests/test_harness.py` (6; re-run 2026-10-22: 6 passed)
- Destination parsing and rewrite; utterance → act mapping; wait / ask remember and resume the interrupted step; revisable planner follows instruction and budget; each invariant catches what it promises and passes a clean run.

## `tests/test_scenarios_scorecard.py` (5; 2026-10-22: 5 passed)
- Every `--say` / `--block-at` has a declared change at its time; swap goal texts appear in their utterance; arm label; a mismatched declaration raises; `--score-only` on a synthetic matrix (scores, skips a schedule mismatch, writes both files).

## Measured: matrix with scripted S1 (2026-10-21, Mac, no GPU; `results/scenarios/skills-20261021-1335*`)
- **2026-10-22 (19 cases, scripted S1, `--no-video`, monitor `results/monitor/skills-20261021/monitor.json`): 17/19** (`results/scenarios/20261022-155022`). New `stop-stays`, `jev-wait-resume`, `jev-ask-answer`, `jev-wait-budget` pass. `blocked` fails as before (contact-blind monitor). `step-events` failed for its setup: the watcher ran 38 checks with 0 errors but no step was ever done; the scripted S1 has completed no step in any recorded run (a standalone "press the start button" run left the microwave off). It now runs only with `--policy remote`.
- 2026-10-21: **13/14 pass.** First pass 11/13 (+ `blocked` skipped): `speculative` and `speculative-phase` starved 3 and 2 ticks at 2.0 s after a 253 / 261 ms step at 0.1 s (video encoding in the run). After encoding moved off the run, both pass (real-time factor 0.954 / 0.958, 0 starved). Other 11 cases: real-time factor 0.983–0.992, 0 starved, no invariant findings.
- **`blocked` fails** (monitor from 7 recorded scripted-S1 nominal runs, window 0.5 s, segment 5 s, α 0.1, 12 calibration segments, `results/monitor/skills-20261021/`): no deviation; max score 2.15 vs threshold 2.54. Cause: scripted grasps press into contact, so "nominal" calibration already contains blocked-like motion (residual scale 2.9 cm vs 0.3 cm in stand-in runs). Needs contact-aware calibration (step 6); no threshold tuning (decisions 2026-10-21).
- Bugs the matrix found, fixed at their owner:
  - Runtime blend pushed an action past a bound and reported it as an upstream violation (10 of 13 cases at the time) → core: incoming action checked, blend projected onto the bounds.
  - Video encoding during the run stalled the loop 236 ms and corrupted the S1 latency estimate → harness: encode after the run.
  - `RemotePolicy` dropped the chunk's `info` → core fix.
- π0.5² through the same entrypoint: [`v3-runtime.md`](v3-runtime.md) (two runs, 2026-10-21). Matrix with `--policy remote`: not run yet.
