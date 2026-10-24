# Failure cases: what broke, why, and what changed (as of 2026-10-24)

These are the failures we found while building and testing the v1 S1/S2 loop: XR-1 (earlier π0.5²) as S1, Jev on
OneJev as S2, the state owner, and the execution runtime. For each one we name the layer that owned the cause and
the structural fix. Evidence cites `docs/decisions.md` by entry date (`dec MM-DD`), `docs/paper/experiments.md` by id
(`E..`), or the file named.

**Status:** **fixed** = change in place, verified by a test or a run · **mitigated** = harm bounded, cause remains ·
**open** = known, not fixed. "Withdrawn" marks evidence taken on corrupted frames before the render fix (E22).

## 1. Evaluation integrity
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| Live camera frames were noise from the first step (neighbour-pixel diff 70–84 vs 5 on real frames); silent for a day | robosuite frees a replaced simulator's render context in whichever GL context is current, which deletes the live framebuffer; our `gc.collect()` set it off | executor · `executors/robocasa-pi052/render_context.py` frees it inside the context being destroyed | reproduced with a stand-in policy and no network; `executors/robocasa-pi052/tests/test_render_context.py`; dec 10-22 | fixed (GLFW/CGL only) |
| Nothing checked what the policy actually saw | no integrity check in the live process | executor · `harness/frame_check.py` marks a run invalid and unscored on noise (diff > 25) or a blank frame (std < 3) | 963 real frames: diff ≤ 7.4, std ≥ 26.8; frames valid in every campaign run; dec 10-23 | fixed |
| Conclusions drawn from corrupted runs: π0.5² 0/14 composite, oracle 0/4, memory and delay/RTC A/Bs, SAM live, XR-1's first 1/6, GPU-suite behaviour | same render bug | all pre-fix `realtime.py` results withdrawn. The move to XR-1 rests on its published 57.28 % and E23 | E16, E18–E21; dec 10-22 | mitigated (not re-run) |
| Same speech case, different XR-1 outcome (H01-W vs H02 actions differ at t = 0.1 s); WashLettuce is 84 % in XR-1's own evaluation | sampled actions; server noise comes from a global torch RNG seeded once at import (`docs/research/2026-10-22-xr1-integration.md` §a) | evaluation: each line judged on its own; success compared only with repeats per case | campaign `analysis.md` "Runs" | open (no repeats yet) |
| Our judge mis-scored lines: it read the robot's state after the answer, and counted a "revision" to the task already in force as a change | it read owner events in the wrong order, and the owner logs nothing when the sentence is already in force | eval · `harness/judge.py`: state taken from events before the line; revision read from the decider's record; a same-task revise judged by its effect. All runs re-judged | `executors/robocasa-pi052/tests/test_campaign.py`; `analysis.md` | fixed |
| SmolVLA eval silently wrong: docs swap two cameras, wrong scene split, sentences instead of task names (1/5 vs 0/5) | eval settings copied from docs and CLI defaults instead of the checkpoint | settings read from each checkpoint's `policy_preprocessor.json`; own rollout loop | E7 (4/20 after the fixes); dec 10-17 | fixed (v2; kept as a gate in `docs/model-bringup/checklist.md`) |
| "OneJev ignores the image" (blank ≈ real) | option order shuffled per variant, which confounded the ablation | ablation redone: fixed order, matched counterfactuals | E13; dec 10-19 | fixed (claim retracted) |

## 2. Real time
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| A 105 s network stall starved the robot for the rest of the run | S1 requests had no deadline | core · `rojev/system1/runner.py`: deadline = one chunk's duration; on timeout, re-request on the newest frame | dec 10-22 | fixed |
| That deadline then starved lockstep runs (217 of 292) | a wall-clock deadline in a mode where simulated time waits for the model throws away good answers | core · `S1Runner(deadline=False)` in lockstep (`realtime.py`) | `docs/features/system1-map.md` §5; parity runs: 31 requests per 450 ticks, no starvation | fixed |
| Under lockstep, OneJev's 1.1–1.5 s wall became 1.75–1.85 s simulated against a 1.0 s deadline; "go on" was revised into the task; no success | lockstep paused S1 only | executor · `realtime.py`: no tick while either S2 lane is `busy` | smoke test: wait 3.05 s, resume 6.05 s, success 10.5 s; E24; dec 10-23 | fixed |
| A late interaction answer was refused and the change decided again: two decisions per change in real time | the answer's base command was the owner's own placeholder hold | owner · `rojev/owner/state_owner.py` `_as_decided`: applied to the situation it was decided on | `tests/test_owner.py`; E30 F16 (0.3 s deadline): both late answers applied | fixed |
| OneJev took 1269 ms per decision, of which the server was 75–140 ms | qev's client opened a new connection per request (~650 ms extra WAN round trip) | S2 · `rojev/system2/models/onejev.py` `KeptClient` | median 604 ms, probabilities identical (6 cases); E29. In the loop ~930 ms: the server drops connections idle ~5 s (E30) | mitigated (server keep-alive open) |
| Same mistake earlier on S1: new connection 680 ms network vs 255 ms kept | per-request `urllib` | transport · `rojev/transport/` kept connection + WebP-90 | E14: round trip p50 559 / p90 597 ms | fixed |
| A retried `/act` ran π0.5² (and its planner state) twice | the connection was lost after the server had already run the request | transport · `X-Request-Id` + `ReplayCache` in `Endpoint` | `tests/test_transport.py` (proxy that loses replies); dec 10-22 | fixed |
| Video encoding stalled the loop for 236 ms and skewed the S1 latency estimate; a full GC pass took 45–96 ms | GIL contention in the control process | encode after the run; `gc.freeze()` before threads start | dec 10-21, 10-22 | fixed |
| XR-1 in real time (F13): real-time factor 0.806, 651 of 985 ticks starved, 13 timeouts, 36 of 64 chunks accepted, no success | ~0.9 s S1 latency vs a 0.8 s chunk; the checkpoint is synchronous (no prefix input) | none in code. Lockstep is the simulator mode; real time is reported alongside | E24; dec 10-23 | open (S1 next to robot, RTC or async training) |
| Jev too slow for its 1.0 s deadline: one question p50 961 / p90 1191 ms; two questions p50 2121 ms | act and task questions are two requests | — | `analysis.md` "Runs" | open |

## 3. Interaction (Jev)
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| Remarks and "make it fast pls" became XR-1's task; typos reached S1 | REVISE sent the user's raw words as the task (`S1Replans`) | S2 · `rojev/system2/jev/task_choice.py` `CanonicalTask`: a second typed question over trained sentences + `none`; `S1Replans` removed | campaign: user words never sent to S1; "trun on teh sink fuacet" → "Turn on the sink faucet."; dec 10-24 | fixed |
| "close the cabinet door" (a trained task) could not be chosen | no option named it | same fix: the run's trained sentences are the options (`--skills`) | runs 3–4 (E25) | fixed; feasibility in the current state open |
| Clear new tasks abstained (revise 0.48 vs continue 0.34) | margin 0.2 is declared, not calibrated | cases recorded for a held-out fit | H06, H07, live; `analysis.md` | open |
| Retry and other same-effect acts split probability | for XR-1, retry is the same sentence as continue | S2 · `acts.py`: situation table, one act per effect, retry removed; a deviation offers only continue or wait | dec 10-24 | fixed |
| "stop" split between abort 0.49 / wait 0.41, so the robot asked instead of stopping (4 of 9 stops) | abstention compared labels, not effects (both hold) | S2 · `acts.effect`: probabilities summed per effect (move · hold · switch) | E28 offline replay of 82 lines: two stops now stop; E30 live: F06 4 / 4 | fixed |
| "ruko" (unknown stop word) → revise p 0.45 (margin 0.22) → "Turn on the sink faucet." p 0.69: **unsafe** | two weak guesses compound; the label margin just cleared 0.2 | same effect rule: revise 0.45 vs hold 0.38 → ask | E28: now asks; E30 live: asks. Regression: one vague line while held now stops instead of waiting | fixed |
| "are you done?" mid-task → abort p 0.47; stopped a working robot | no act answers a question | — | F07; `analysis.md` finding 4 | open |
| After success, "go on" resumed the finished task and the base circled: **unsafe** | nothing in S2 knows a task is done; XR-1 reports no completion; the planning lane never says COMPLETE | — | run 5 (success 36.75 s), live L2; dec 10-23 | open (completion signal + owner rule) |
| Task question tied lettuce vs faucet (0.05 apart) on "wash the lettuce in the sink" | the candidate sentences share words ("sink") | — | live session; `analysis.md` finding 6 | open |
| A repeated question or "waiting" went unsaid | the owner spoke a template only when it issued a new command | owner · `state_owner.py` `_apply`: a templated reply is said again on renew | run 1 (E25); `docs/features/system2-map.md` §3 | fixed |
| "go on" while held matched the act named "continue", which there meant "keep holding" | act names were shown to the model | S2: neutral option ids; options worded as what the user wants | `docs/features/jev-interaction.md` | fixed |
| A fitted calibration (T 0.441, margin 0.103, 30 cases, one scene) made wrong reads decisive (run 5) | accuracy measured on the fitting set; the act set changed afterwards | tooling · `calibrate_jev.py --fit … --held-out …` writes a calibration only if held-out accuracy reaches the target; one per question | E26; dec 10-23, 10-24 | fixed (no fitted value yet: T = 1, margin 0.2 declared) |
| Hold recheck of 3 s re-says "waiting" 3 s after a question | campaign setting | — | `analysis.md` finding 7 | open |

## 4. Owner and runtime
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| Every change went unanswered: the 1 s review lease expired before OneJev's 1.15 s answer | two deadlines for one wait | owner · `review_window_s` = the interaction deadline | GPU suite; dec 10-22 | fixed |
| "wait" and "go on" became π0.5²'s task text | the owner adopted every utterance as the task | owner: only an accepted proposal carrying `task` changes the task text | GPU suite; dec 10-22 | fixed |
| After "stop", the planner's next heartbeat restarted the task (D4) | each decider was trusted to respect holds | owner: standing `Hold`; routine proposals refused while it stands | dec 10-22; `tests/test_owner.py` | fixed |
| An unread "stop" could move the robot | an unanswered change escalated to the planning lane | owner: hold `UNANSWERED`, say so; planning never answers a change (reverses dec 10-21) | dec 10-22 | fixed |
| The robot stayed held forever after a safety hold | the hold kept the command active but refused its chunks; the owner kept renewing it | runtime · `AckStatus.DROPPED` ends the command; the owner re-decides | found in docs review; unit tests; dec 10-22 | fixed |
| Planning triggers dropped while a change was pending | deferral discarded them | owner: the latest trigger is replayed after the decision | dec 10-22 | fixed |
| The runtime reported its own blend overshoot (−1.006) as an upstream violation (10 of 13 live cases) | clamp check ran after the blend | runtime · `rojev/control/execution.py`: check input, blend, then project without reporting | dec 10-21 | fixed |
| Rendering from the loop thread deadlocked (macOS CGL); a starved command outlived its lease; S1 saw stale task text | thread and lease ownership not explicit | one GL thread; leases end commands in any mode; `AcceptedGoal.instruction` | pre-briefing review; dec 10-20 | fixed |
| Consensus commitment livelocked and starved; a fixed `min_commit` of 10 was unsafe (RoboCasa needed 13) | truncation to zero; a constant standing in for measured latency | S1 · `rojev/system1/commit.py` majority rules, never empty; floor ⌈2L/dt⌉ + 1 from measured L | dec 10-20, 10-21 | fixed (stand-ins only) |
| Execution monitor: 0.17 false alarms per segment at α = 0.05; cannot tell contact from a fault (2.15 vs 2.54) | calibrated on the runs it was fitted on; kinematics only | S1 · `rojev/system1/monitor.py`: split by run, lag-aware map. Contact still blind | dec 10-21 | fixed / open (contact) |

## 5. S1 integration
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| Near miss: XR-1 state read base-first instead of end-effector-first | LeRobot's `agent_pos` layout (ee, base, gripper) differs from the checkpoint's (ee, gripper, base) | executor · `executors/robocasa-xr1/xr1_inputs.py` `state14`; caught by reading the env source | `docs/model-bringup/checklist.md` §2; parity 18/20 vs their 20/20 (E23) | fixed |
| XR-1 needs 4 frames 2 ticks apart; the integration brief proposed core changes (`Observation.history`, wire v4) | model-specific input; the server sees one frame per chunk, and a server-side buffer breaks retries | executor · `history.py` `FrameHistory` + `HistoryPolicy` on the unchanged wire; newest frame matched by identity | dec 10-22; E23 | fixed |
| π0.5²: 486 of 600 ticks reported violations, all on gripper and base mode (≤ 1.018) | the adapter did not map switch outputs into the action space | executor · `embodiment.py` `to_action_space` (0.5 threshold, as RoboCasa's gym wrapper); applied on the robot side for XR-1 | dec 10-21; campaign: only continuous XR-1 outputs ≤ 1.055 clamped (6–18 ticks per run) | fixed |
| XR-1 env never provisioned: flash-attn 2.8.3 wheel named `+cu12torch2.8…`, metadata 2.8.3, so `uv sync --frozen --check` failed | prebuilt wheel's file name and metadata disagree | executor env · built from the sdist with `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE`; readiness imports `flash_attn_2_cuda` (`executors/robocasa-xr1/pyproject.toml`, `gpu/hook.sh`) | dec 10-23 | fixed |
| π0.5² planner flipped freezer ↔ fridge 18× in 60 s, each switch logged as progress | memoryless subtask head (trained on task + image only) | S2 · `StepEvent.checked`: a switch is a claim; `CommitVerifyPlanner` pins the subtask. Cut switches 19 → 2, but verified nothing and pinned a wrong first pick | dec 10-22 (withdrawn: E16, E18) | open (no completion signal) |
| π0.5² planner memory gave false notes ("I placed it in the freezer") | head not trained on memory; note written from the plan, not observed state | off by default (`serve.py --planner-memory` opt-in) | 16 / 10 switches off vs 21 / 11 on (withdrawn: E19) | mitigated |

## 6. Models and checks
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| MJev took ~24 s per request on a T4; two guessed fixes (precision, FLA) changed nothing | slow bf16 Conv3d patch embedding on sm_75 (20.6 of 23.8 s) | S2 service: `PatchLinear` exact matmul + startup check; fp16 → 1.3 s | E12; dec 10-18 | fixed (MJev later removed) |
| MJev failed the done-check gate: balanced accuracy 0.50 / 0.45; never says "done" | — (OneJev 0.52 / 0.52 too) | MJev out of the loop; commit-and-verify releases on its time bound | E17 (222 sim-labelled pairs); dec 10-22 | open (completion signal) |
| OneJev got cameras glued side by side, so no view boundaries | one concatenated 768 × 256 picture | S2 · `onejev.camera_views`: each camera its own image, one request | dec 10-22 | fixed (gain unmeasured) |
| SAM 2.1: a bad first mask never recovers (IoU 0.00); re-acquisition 3/11; 2D "placed inside" 0.68 balanced (freezer 90 % false "done") | no grounding, no first-mask check, no depth | measurement only, not in the loop | E20 (live part withdrawn); dec 10-22 | open |

## 7. Infrastructure
| failure (symptom) | root cause | fix (layer · file) | evidence | status |
|---|---|---|---|---|
| SAM tracking ran out of GPU memory after 177 frames (7.5 GB) | session frames held on the GPU | service · `services/sam2`: session state on CPU; idle sessions closed after 600 s | dec 10-22 | fixed |
| `profile.py` / `bisect.py` broke imports | script names shadowed stdlib modules | renamed (neither file remains) | `docs/model-bringup/search-directions.md` | fixed |
| Fresh-box setup ran the driver, two env builds and two weight downloads one after another | serial orchestration | infra · `scripts/gpu_setup.sh`: parallel jobs synchronised by marker files; services start in order (GPU memory projected) | dec 10-24; `--dry-run` against `lambda` | fixed |
| π0.5² unit died at import on the first fresh box | hook did not ship `embodiment.py` | hook `files()` now includes it | `docs/features/gpu-setup.md` | fixed |
| `pkill -f` killed its own SSH shell; nvidia-utils 570 mismatched the 580 module; openrsync `-R --delete` removed a local `.venv` | tooling quirks | systemd units; utils by the built module's version; trees synced one by one | `docs/features/gpu-setup.md` pitfalls | fixed |
| OneJev (no auth) bound 0.0.0.0 for minutes; `serve.py` buffered bodies before checking the token | upstream defaults; checks after buffering | `--host 127.0.0.1` mandatory, tunnel only; `Endpoint.admit` runs on the headers | dec 10-16, 10-22 | fixed |

## Lessons
- **Check what the model sees, in the live process, on every run.** One GL teardown bug voided a day of results.
  A two-number frame check now catches it within a second.
- **A fix in one mode can break another.** The one-chunk deadline that saved real time starved 217 of 292 in
  lockstep. Deadlines belong to the clock that matters, and every mode needs its own run.
- **Pausing the world for one model only moves latency.** Lockstep must stop time for S1 and S2 alike, and it still
  overstates the real robot. Report real time next to it.
- **Profile before fixing.** MJev's latency had two plausible causes, and both were wrong. OneJev's 1.27 s was
  mostly connection setup: the same mistake we had already fixed for S1.
- **Free text must not cross a boundary it was not trained for.** User words became S1's prompt, and utterances
  became the task. Typed choices over what the receiver can actually do fixed both.
- **Hesitate on consequences, not labels.** "stop" vs "wait" (same effect) asked needlessly. A weak "revise" moved the
  robot. Summing probabilities per effect fixes both directions.
- **One owner per fact.** Most owner bugs were two components each holding half of one fact: lease vs deadline,
  planner vs hold, utterance vs task.
- **Evaluation code fails like product code.** A calibration fitted on its own cases, a judge reading state after the
  answer, and a shuffled-option ablation each gave confident wrong conclusions. Use held-out splits and test the
  judge.

## Source notes
- `217 of 292`: "requests starved" in `docs/features/system1-map.md` §5, "starved ticks" in `docs/model-bringup/checklist.md` §6.
- Listed new tasks: `docs/features/jev-interaction.md` says 6 / 14 right; `analysis.md` gives 3 / 7 (scripted + live) and 4 / 6 (fuzz), i.e. 7 / 13.
- `docs/features/gpu-setup.md` calls `raw` the code's codec default; `rojev/system1/wire.py` has `DEFAULT_CODEC = "webp90"` (dec 10-22).
- π0.5² subtask flips: 18× (dec 10-22 claim entry) vs 19 → 2 (MJev-out entry). Both come from GPU-suite runs that E16 marks withdrawn.
