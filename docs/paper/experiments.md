# Experiments record (exact numbers; every number cites the file it came from)

## Read this first
- **Stack now (v3 runtime, v1 loop):** S1 = Xiaomi-Robotics-1 RoboCasa365 (5B, BF16, `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365` @3a6d029, `config/xr1_weights.txt`) · S2 = Jev v1 interaction layer on OneJev 1.6 4B (`OmniJev/OneJev1.6-4B` @7d55a87, `config/onejev_weights.txt`, served by qev) + XR-1's native planner (whole-task sentence) · MJev out of the loop since 2026-10-22.
- **Hardware:** RoboCasa simulator on a MacBook Air M5 (India); models on one A100 40 GB box over SSH tunnels (`docs/features/remote-models.md`, `docs/features/gpu-setup.md`).
- **Evaluation mode:** lockstep (simulated time stops while S1 or S2 computes); real time reported alongside (`docs/decisions.md` 2026-10-23).
- **Status column:** `valid` = stands as measured · `superseded by X` = still true, but the design it informed is gone · `withdrawn: corrupted frames` = live `realtime.py` result taken before the 2026-10-22 render-context fix (E22); not evidence until re-run.
- `docs/decisions.md` is authoritative for status. "n" = frames, lines or episodes.

## Index
| id | date | question | result (one line) | status | source |
|---|---|---|---|---|---|
| E1 | 10-16 | Does OneJev 0.8B use the image? | P(green box) 0.984 shown vs 0.140 hidden | superseded by 4B decision (E13) | `docs/features/history/phase0-install.md` |
| E2 | 10-16 | Does instruction text bias answers? | Instruction raised P(box visible) on hidden-box frames (0.003 → 0.138) | superseded by v3 (task / user line kept apart) | this file, git history |
| E3 | 10-16 | Prompt variants for a false premise | Two-step (visibility on neutral state → decision) 4/4; option-order debias no effect | superseded by v3 | `playground/archive/e3_prompt_variants.py` |
| E4 | 10-17 | Does viewpoint change the answer? | Overhead camera: P(box) 0.03, no false approach | superseded by v3 | `docs/features/robot-interface.md` |
| E5 | 10-16 | OneJev readout mass | 3–8 % on answer letters (image questions); by design, probabilities are relative | valid (applies to all OneJev scores) | this file, git history |
| E6 | 10-17 | Loopback communication cost | Transport 0.41 ms vs decision 73 ms median | superseded by E14 / E29 (WAN) | `playground/archive/e6_ipc_bench.py` |
| E7 | 10-17 | SmolVLA as RoboCasa executor | 4/20 after fixing 3 eval mismatches (cameras, split, task names) | superseded by π0.5² → XR-1 | this file, git history |
| E8 | 10-17 | π0.5² subtask vocabulary | 21,326 distinct subtasks in 32,043 episodes | superseded by v3 (no subtask selection) | `data/vocab/pi052_robocasa_subtasks.json` |
| E9 | 10-17 | Candidate shortlist coverage@K | Held-out tasks: 25.2 % of needed subtasks never seen as strings; @10 = .398 | superseded by v3 | `results/runs/e9_coverage_20261017-174343/` |
| E9b | 10-17 | Template + slot-filling builder | Reach 75 → 89 %, ranking +0.7 pt @10; not adopted | superseded by v3 | `results/logs/e9b.log` |
| E10 | 10-18 | Live workflow, oracle arm E | 0/3; "grasped" ≠ "picked up"; π0.5² dropped the steak | superseded by v3; frames never checked | `results/runs/workflow/` |
| E11 | 10-18 | MJev bf16 vs fp32 on T4 | Top-1 12/12, max Δp 0.029; 24.5 s / request | superseded by E17 (MJev out) | `results/kaggle/armd_20261018-interactive/` |
| E12 | 10-18 | MJev speed levers | Slow Conv3d patch embed (20.6 of 23.8 s); as matmul: fp16 1,280 ms (19×) | superseded by E17 | `results/kaggle/armd_20261018-interactive/mjev_speed_patched.json` |
| E13 | 10-18 | OneJev next-subtask choice, 0.8B vs 4B | 4B 40 % vs 0.8B 22.5 % (n 40); image ablation confounded | superseded by v3; → 4B adopted | `results/runs/e13_onejev_bench_*` |
| E14 | 10-21 | Remote S1 transport over the WAN | Keep-alive + WebP-90 round trip p50 559 / p90 597 ms; best path | valid | `docs/features/remote-models.md` |
| E15 | 10-21 | Runtime scenario matrix, stand-in S1 | 13/14 pass; real-time factor 0.954–0.992, 0 starved | valid (stand-in ignores images) | `docs/features/live-harness.md` |
| E16 | 10-22 | GPU suite, π0.5² + Jev live | Owner bugs found (review lease < Jev latency; utterance became task text), fixed; model behaviour (18 subtask flips in 60 s) | bugs valid; behaviour withdrawn: corrupted frames | `docs/decisions.md` 10-22 entries |
| E17 | 10-22 | Done-check A/B on simulator-labelled frames | MJev 0.50 / 0.45, OneJev 0.52 / 0.52 balanced accuracy → MJev out | valid (still frames) | `results/done-check/20261022*/` |
| E18 | 10-22 | Commit-and-verify live | Subtask switches 19 → 2, nothing verified | withdrawn: corrupted frames | `docs/decisions.md` 10-22 |
| E19 | 10-22 | π0.5² planner memory A/B | Switches 16 / 10 off vs 21 / 11 on, no success | withdrawn: corrupted frames | `results/memory-ab/20261022` |
| E20 | 10-22 | SAM 2.1 as a completion signal | Still frames: mask IoU 0.64; live tracking IoU 0.86–0.96 | still frames valid; live withdrawn | `docs/decisions.md` 10-22 |
| E21 | 10-22 | π0.5² on composite tasks (+ oracle) | 0/14 episodes; oracle 0/4 | withdrawn: corrupted frames | `results/delay-ab/` |
| E22 | 10-22 | Render-context bug; frame check | Live frames were noise (neighbour diff 70–84 vs 5); check added | valid | `docs/decisions.md` 10-22, 10-23 |
| E23 | 10-22 | XR-1 parity in our harness | 18/20 vs their 20/20; first pre-fix run withdrawn | valid (fixed run) | `results/xr1-parity/20261022-fixed` |
| E24 | 10-23 | Lockstep vs real time | Lockstep smoke: success 10.5 s; real time: factor 0.81, 651 starved ticks | valid | `results/v1-smoke/`, `results/campaign/20261024-134629/F13` |
| E25 | 10-23 | Interactive runs 1–5 (XR-1 + Jev) | Remarks became tasks; trained tasks not offered; finished task resumed → v1 testing stopped | valid | `results/v1-interactive/` |
| E26 | 10-23 | First fitted Jev calibration | T 0.44, margin 0.10 on 30 cases | withdrawn 10-24: no held-out set | `results/jev-calibration/20261023/` |
| E27 | 10-24 | v1 test campaign (scripted, fuzz, live users) | 82 lines: 55 right, 2 unsafe, 24 safe misses, 1 unanswered | valid | `results/campaign/20261024-134629/analysis.md` |
| E28 | 10-24 | Effect-based hesitation, offline replay | Unsafe "ruko" now asks; two stops now stop; one regression | valid (offline) | `docs/decisions.md` 10-24 |
| E29 | 10-24 | OneJev latency breakdown; kept connection | 1269 → 604 ms median, identical probabilities | valid | `docs/features/remote-models.md` |
| E30 | 10-24 | Live rerun: effect hesitation, kept connection, late answers | stop/go-on ×2 4/4; "ruko" asks; late answers applied; idle connection dropped after ~5 s | valid | `results/campaign/20261024-rerun/` |
| E31 | 10-30 | RynnValue-4B (frozen) on the E17 done-check pairs | P(success) ranks the done step first on identical images 0.82 (OneJev 0.46, MJev 0.28); written verdict never "Yes" (0/108) | valid (still states, 1 camera) | `results/rynnvalue-check/20261030/` |

## E17 — Done-check A/B: neither model can tell a finished step (2026-10-22)
- Setup: kitchen states set in the simulator (object placed, door closed, appliance on), labelled by the task's own completion test; 2 of 80 states dropped (predicate failed), never relabelled; 222 pairs (108 done, 114 not). Two question forms: "is this step done?" and yes / no postconditions. Still frames, 0 of 444 images corrupt.

| checker (top answer) | "done?" bal. acc / recall / false done | postconditions bal. acc / recall / false done | p50 ms |
|---|---|---|---|
| MJev | 0.50 / 0.00 / 0.00 | 0.45 / 0.22 / 0.32 | 57 |
| OneJev, views glued | 0.52 / 0.06 / 0.01 | 0.52 / 0.42 / 0.38 | 350 |
| OneJev, separate views | 0.52 / 0.07 / 0.03 | 0.57 / 0.36 / 0.22 | 142 |

- Meaning: no separation to put a threshold on; MJev never says "done" at all. MJev is not loaded by default; commit-and-verify releases only on its time bound until a measured completion signal exists.
- Open: 256 × 256 multi-camera views may be the limit, not the models; teleported placements may differ from robot-made ones.
- Source: `results/done-check/20261022/report{,_post}.json`, `results/done-check/20261022-separate/report{,_post}.json`; `docs/decisions.md` 2026-10-22 "MJev out of the v1 loop".

## E21 — π0.5² composite failure and the oracle (2026-10-22) — WITHDRAWN
- Claimed: π0.5² (`pi052_robocasa`) 0/14 composite episodes (real time, lockstep, full budget); with an oracle planner 0/4 (picks 2/4, places 0/4); compared with RoboCasa365's π0.5 composite rate 7.1 %.
- Status: withdrawn with every pre-fix `realtime.py` result (memory A/B, delay A/B incl. RTC, full-budget lockstep, oracle, SAM live, XR-1's first run). "The checkpoint is the limit" and "delay is not the cause" are unverified. Not re-run; the switch to XR-1 stands on XR-1's own published number (57.28 % on target50) and E23.
- Source: `docs/decisions.md` 2026-10-22 "π0.5² … cannot execute our composite tasks (EVIDENCE WITHDRAWN …)".

## E22 — Render-context bug and the frame-integrity check (2026-10-22 / 23)
- Cause: robosuite frees a replaced simulator's MuJoCo render context in whatever GL context is current, deleting the live simulator's framebuffer and textures; `realtime.py` runs `gc.collect()` before `gc.freeze()`, so the policy saw noise from the first step. Reproduced with a stand-in policy, no network, lockstep and real time.
- Signal: neighbour-pixel difference 70–84 on corrupted frames vs 5 on real ones. Still-frame sets (done-check, SAM check) never collect: 0 of 444 images corrupt.
- Fix: `executors/robocasa-pi052/render_context.py` frees the context in its own GL context (GLFW / CGL only); test `executors/robocasa-pi052/tests/test_render_context.py`.
- Check on every run (`harness/frame_check.FrameCheck`, every 1 s): noise = neighbour difference > 25, blank = std < 3 → run invalid, not scored. 963 real frames: difference ≤ 7.4, std ≥ 26.8.
- Source: `docs/decisions.md` 2026-10-22 "Camera images in live runs were corrupted…", 2026-10-23 "Every run checks its camera frames…".

## E23 — XR-1 parity in our harness (2026-10-22, re-run after the fix)
- Setup: XR-1's published target50 cases (their seeds and horizons), 4 tasks × 5 seeds, lockstep, refill 0.05 s; 3 atomic-seen + 1 composite-seen task; Composite-Unseen never run here.

| task | ours | theirs |
|---|---|---|
| CloseToasterOvenDoor | 4/5 | 5/5 |
| TurnOnElectricKettle | 5/5 | 5/5 |
| PickPlaceCounterToCabinet | 5/5 | 5/5 |
| RinseSinkBasin | 4/5 | 5/5 |
| **total** | **18/20** | **20/20** |

- XR-1 request wall time (Mac ↔ A100): p50 511 ms, p90 731 ms (1,020 requests); server ~40 ms preprocess + 275–290 ms inference, 10.2 GB.
- Meaning: our runtime reproduces XR-1's skill to within 2 episodes; S1 can be trusted on these tasks, so loop failures point at S2.
- Source: `results/xr1-parity/20261022-fixed/*/summary.json`, `s1_requests.jsonl`; `docs/features/system1-map.md` §12; `docs/research/2026-10-23-v2-plan.md`; server numbers `docs/features/remote-models.md`.

## E24 — Lockstep vs real time (2026-10-23 / 24)
- Lockstep smoke (XR-1 + Jev, CloseToasterOvenDoor s107): with only S1 paused, OneJev's 1.1–1.5 s wall became 1.75–1.85 s simulated against a 1.0 s deadline: both answers late, "go on" revised into the task, no success. With S2 paused too: wait at 3.05 s, resume at 6.05 s, success at 10.5 s.
- Real time (campaign F13, WashLettuce s1657, 60 s budget): real-time factor 0.806, max lag 10.8 s, 651 of 985 ticks starved, 13 S1 timeouts (deadline 0.80 s), 36 of 64 S1 answers accepted, S1 latency estimate 0.9 s vs a 0.8 s chunk; no success. Jev answered in 0.90 / 0.85 s (both right).
- Meaning: over this link XR-1 cannot supply chunks in real time; lockstep is the only fair simulator comparison until S1 runs next to the robot. Lockstep hides S1 and S2 latency, so it overstates real-world performance.
- Source: `results/v1-smoke/20261023-toaster{,-2}/summary.json`; `results/campaign/20261024-134629/F13/summary.json`; `docs/decisions.md` 2026-10-23 "Lockstep also stops simulated time while S2 decides".

## E25 — Interactive runs 1–5 and the stop decision (2026-10-23)
- Setup: a person typing to XR-1 + Jev (OneJev), PickPlaceCounterToCabinet s407, lockstep, ≤ 120 s simulated; frame check valid in all six summaries.

| run | task success | what it exposed |
|---|---|---|
| 1 (+ replay) | 17.1 s (replay 19.05 s) | a remark became the task; a repeated question went unsaid (fixed: owner re-says) |
| 2 | no | goals outside XR-1's repertoire → erratic motion |
| 3, 4 | 66.3 s; no | hesitation between equivalent acts; "close the cabinet door" (a trained task) not offered as an option; typos sent raw; "make it fast pls" revised into the task |
| 5 (E26 calibration) | 36.75 s | after success the robot kept executing, was "resumed" into the finished task, the base circled |

- Decision (2026-10-23): no more v1 test rounds; build the capability signal (REVISE chooses among trained task sentences, built 2026-10-24) and a completion signal first.
- Source: `results/v1-interactive/run*/summary.json`; `docs/features/system2-map.md` §9; `docs/decisions.md` 2026-10-23 "v1 interactive testing stops…".

## E26 — Jev calibration 2026-10-23: fitted, then withdrawn
- Fit: T = 0.441, abstain margin 0.103 (target accuracy 0.8), 30 labelled decisions (runs 2–4 + run-1 replay); NLL 0.636 → 0.516; 87 % of decisions kept at 81 % accuracy. Target 0.9 would need margin 0.72 (abstain every other turn).
- Withdrawn 2026-10-24: accuracy measured on the cases the threshold was fitted on; one scene; the act set changed since. Run 5 showed it made wrong reads decisive. Rule now: one calibration per question, shipped only if held-out runs (split by run / scene) reach the target; until then T = 1, margin 0.2 declared.
- Source: `results/jev-calibration/20261023/calibration.json`; `docs/decisions.md` 2026-10-23 and 2026-10-24 calibration entries.

## E27 — v1 test campaign (2026-10-24)
- Setup: XR-1 + Jev (act + task questions, T = 1 / margin 0.2) + native planner, lockstep, 4 runs at a time on one A100; tasks and seeds XR-1 solved in its own evaluation. Each line judged by the robot's state when heard: pass · unsafe (wrong answer that moves the robot) · safe miss · unanswered.

| users | lines | pass | unsafe | safe miss | unanswered |
|---|---|---|---|---|---|
| scripted (27 runs) | 39 | 26 | 1 | 11 | 1 (F12, by design) |
| fuzz (8 runs) | 32 | 24 | 0 | 8 | 0 |
| live (2 sessions) | 11 | 5 | 1 | 5 | 0 |
| **all** | **82** | **55 (67 %)** | **2** | **24** | **1** |

| intent | lines | right | main miss |
|---|---|---|---|
| stop | 9 | 3 | abort vs wait split → ask ×4; "ruko" → revise (unsafe) |
| resume | 10 | 7 | after success → resumed the finished task (live, unsafe) |
| new task (listed) | 7 | 3 | act abstained on clear requests; task question tied on shared words (0.05 apart) |
| unlisted / gibberish / negation / vague | 8 | 8 | — |
| pause | 5 | 3 (+1 by design) | a question while pausing → ask |
| status ("are you done?") | 2 | 0 | → abort (stopped a working robot) |

- Findings: (1) an unknown stop word can start a task: two weak guesses compound (act revise p 0.45, then task p 0.69); (2) completion gap reproduced live; (3) abstaining between two holds asks instead of stopping; (4) no act answers a status question; (5) margin 0.2 loses correct revises (0.48 vs 0.34).
- Worked: never revised on unlisted, gibberish, negated or vague lines; never sent the user's words to XR-1; typos mapped to the trained sentence; decider failure → safety hold.
- Jev decision time (wall): one question p50 961 / p90 1191 / max 2104 ms (n 37); two questions p50 2121 ms (n 8). Baselines without speech 4/4. Frames valid in every run.
- Caveats: one run per scripted case; XR-1 sampling differs run to run, so task success is not comparable across cases; late-answer path not triggered.
- Source: `results/campaign/20261024-134629/analysis.md`, `report.md`; `docs/features/campaign.md`.

## E28 — Effect-based hesitation, offline replay of E27 (2026-10-24)
- Change: act probabilities summed per effect (move · hold · switch); abstention compares the top two effects, not labels.
- Trigger cases: "stop" split abort 0.49 / wait 0.41 → asked although both hold; "ruko" revise 0.45 vs hold 0.38.
- Replay of all 82 recorded lines: the unsafe "ruko" now asks; two stops now stop; one remark keeps holding; regression: one vague line while already held now stops instead of waiting.
- Source: `docs/decisions.md` 2026-10-24 "Jev hesitates between effects…".

## E29 — OneJev decision latency and the kept connection (2026-10-24)
- Setup: Mac in India → A100 over the SSH tunnel; 3 cameras, 256 × 256, JPEG q95 (~26 kB each).

| part | ms |
|---|---|
| qev server: render + one forward pass (~74 tokens, ~20 ms per image) | 75–140 |
| one round trip (kept connection) | ~270–345 |
| image upload (3 images vs none) | ~215 |
| extra round trip of a new connection per request | ~650 |

- `KeptClient` (one kept HTTP connection): median 1269 → 604 ms per decision, probabilities identical (6 recorded cases).
- Rejected: JPEG q85 / q75, WebP q90 / q80 changed the top option in 1 / 6 cases; only WebP q80 clearly faster (428 vs 533 ms).
- Source: `docs/features/remote-models.md` "OneJev decision latency"; `docs/decisions.md` 2026-10-24 "OneJev over one kept connection".

## E30 — Live rerun 2026-10-24 (effect hesitation, kept connection, late answers)
- Setup: XR-1 + Jev/OneJev, lockstep, the four cases the hesitation change affects (H05, F06, F08, F10) four at a time;
  F16 alone in real time with the interaction deadline forced to 0.3 s. `results/campaign/20261024-rerun/report.md`.

| Case | Before (E27) | After |
|---|---|---|
| F06 stop / go on ×2 | 2 / 4 (stops asked) | **4 / 4** |
| F10 "ruko" | revise to the faucet task (unsafe) | **ask** |
| H05 "stop" | ask | wait (hold 0.94; robot holds, no question; judged a miss: abort expected) |
| F08 after success: stop, go on | ask, ask | wait, **resume (unsafe: completion gap, open)** |
| F16 real time, 0.3 s deadline | — | both changes timed out → owner hold → **late answers applied** at +0.7 s / +0.9 s, not re-decided |
- Jev in the loop with `KeptClient`: 548–1232 ms (median ~930), not the 604 ms of E29: the server drops a connection
  idle for ~5 s (health 263–302 ms after ≤ 4 s idle, 665–960 ms after 6–10 s), and decisions are seconds apart.
- Meaning: effect hesitation removes the unsafe "ruko" and the stop/ask splits live; the late-answer rule works;
  the speed gain needs a longer server keep-alive (open).

## E31 — RynnValue-4B as a frozen done signal (2026-10-30)
- Setup: `Alibaba-DAMO-Academy/RynnValue-4B` @2e78c2c (`config/rynnvalue_weights.txt`), int8 weight-only on the Mac
  GPU (mps), `services/rynnvalue/check_done.py`. E17's 222 (frame, step) pairs (108 done), the step as the
  instruction, left agentview only. `chain`: the seed's states up to the labelled one as an ordered sequence (1–4
  frames); `still` (E17's single frame) stopped at 15 answers. Smoke on the model's own example video first: grounding
  right (Match Yes 0.91–0.997 on the true instruction, 0.0004 on a wrong one).

| measure (chain, n = 222, 0 invalid) | RynnValue | OneJev (E17, views glued) | MJev (E17) |
|---|---|---|---|
| AUROC, done vs not-done step on identical images | **0.82** (P(success)) · 0.71 (remaining time) | 0.46 | 0.28 |
| AUROC, all pairs | 0.75 (P(success)) · 0.82 (remaining time) | 0.53 | 0.34 |
| "Success: Yes" (its own verdict): recall / false done | 0.00 / 0.00 | 0.06 / 0.01 | 0.00 / 0.00 |
| P(success) cut at 10 % false done (fitted on this set) | recall 0.46 (P > 0.026) | — | — |
| median latency per question (Mac GPU) | 30.4 s | 0.35 s (A100) | 0.06 s (A100) |
- Inputs differ: E17 asked one still frame with all three views; here a 1–4 frame sequence from one camera. Same
  pairs, same truth.
- By step type (all pairs, P(success)): place 0.83, close 0.73, press 0.64. Max P(success) 0.148: the written verdict
  never flips, the probability carries the signal.
- Remaining time falls with chain length (median 3.67 → 2.33 → 1.98 → 1.49 s for 1–4 frames), and longer chains are
  later states with more steps done, so its all-pairs 0.82 is partly length; the identical-image comparison removes it.
  The repeated-first-frame control (`--mode repeat`) was stopped at 1 of 48 answers.
- Meaning: the first frozen model with a done signal on these frames, but not a release signal as is (no "Yes", a
  threshold fitted on the test set, still states set by the simulator, one camera, 30 s on the Mac). Not in the v1
  loop; a base candidate for the v2 critic next to Robometer-4B.
- Also found: on mps, 8 frames at 640 × 360 gave non-finite logits ("!!!!" text, negative remaining time) while one
  256 × 256 frame matched across attention kernels; every answer now records `valid`. PyTorch's macOS CPU path has no
  fast bf16 matmul (generic `bf16_dot_with_fp32_arith`), so the CPU is not an option at bf16.
- Source: `results/rynnvalue-check/20261030/{answers.jsonl,report.json,run.log}`; E17 numbers recomputed from
  `results/done-check/20261022/scores.jsonl` (P(done), same pairs).

## Superseded findings still worth knowing (one line each)
- E2/E3: an instruction in the prompt raises the model's *visibility* answers; evidence questions must not share the instruction's state (now: `task` and `user_just_said` kept apart).
- E7: three LeRobot eval mismatches (camera mapping, scene split, task names vs sentences) silently invalidate VLA runs; eval settings must come from the checkpoint.
- E12: MJev's latency was one Conv3d on T4 (20.6 of 23.8 s), exact as a matrix multiply (fp16 1,280 ms per 3-question request).
- E13: the image ablation shuffled options per variant, so it shows no image benefit, not that OneJev ignores images.
