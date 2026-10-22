# Workflow levers: every place output can be changed (registry)

Single list of the choices that shape output. Update the **Current** column whenever a value changes, and log the change in `docs/decisions.md`.
Cost of a change: **config** = edit a value · **refit** = rebuild from data (minutes, CPU) · **swap** = different model (re-validate) · **train** = fine-tune (GPU, data).
✱ = not chosen yet.

## 1. Task input
| Lever | Current | Where | Cost | Symptom it fixes |
|---|---|---|---|---|
| Instruction text given to OneJev / the filter | user's task string, verbatim | supervisor loop | config | wrong similar tasks (filter B), wrong choices |

## 2. Scene extraction (MJev-Spatial)
| Lever | Current | Where | Cost | Symptom it fixes |
|---|---|---|---|---|
| Model / precision | MJev-Spatial-4B @0657319 (code @df37e9e), base Qwen3.5-4B @851bf6e; fp16 trunk + vision, fp32 readout on T4 #1, validated vs fp32 (E11/E12f: 12/12, Δp ≤ 0.031) | cloud service | swap | wrong evidence |
| MJev speed | fp16 (1.3 s / 3-question request, E12f); 3 cameras (2 change answers, E12); image-token cap irrelevant at 256²; cause = patch-embedding Conv3d (20.6 s, E12d) → exact matrix-multiply form (4.8 ms, E12e) | MJev env / `serve_mjev.py` | config | decision latency |
| Evidence template | "Robot step: '<s>'. Which is true right now?" → done / possible / blocked, all 3 cameras | `onejev_arm.py` | config | wrong evidence wording |
| Judged subtasks | current + 2 recent + top M = 5 candidates | `onejev_arm.py` | config | right next step never judged |
| Answer type per question | choice / number / point | supervisor | config | — |
| Cameras + resolution | 3 views at 256² (= π0.5² training images) | sim / request | config | object not visible in chosen view |
| State neutrality | evidence asked without the instruction (E2) | supervisor | config | instruction-biased evidence |

## 3. Candidate filter (`rojev/legacy_v2/candidates.py`, v2)
| Lever | Current | Cost | Symptom it fixes |
|---|---|---|---|
| Fitting data | π0.5² training timelines, 32,043 episodes (E8) | refit | vocabulary ≠ executor's |
| Normalisation | lowercase, no punctuation/articles, `SYNONYMS={fridge: refrigerator}` | config | paraphrase duplicates |
| Stopwords (B, C) | 25 fixed function words | config | — |
| Source weights (A, B, C) | (1, 1, 1) | config | ranking (E9) |
| Similar tasks for B | top 10 by BM25 (k1 = 1.5, b = 0.75) | config | — |
| History depth for A | last subtask only | config | loops / repeated subtasks |
| Shortlist size K | ✱ 20–50 (E9: new tasks 47% @20, 55% @50) | config | right subtask missing |
| Canonical wording | most frequent phrasing per meaning | refit | — |
| Templates + slot filling | parked (E9b) | refit | unseen object combinations (25% of new-task subtasks) |

## 4. Decision (OneJev)
| Lever | Current | Where | Cost | Symptom it fixes |
|---|---|---|---|---|
| Model | OneJev 1.6 4B (from 2026-10-19; 0.8B dropped after E13) | Mac, qev + llama.cpp | swap / train | wrong choices with right shortlist |
| Decision question | one Choice: continue current / K = 20 candidates / 2 recent (redo) / task complete; argmax | `onejev_arm.py` | config | wrong next subtask, early/late switching |
| Decision inputs | composite of 3 cameras (one image) + task, previous, current, chunks on current, evidence table | `onejev_arm.py` | config | — |
| Option order | shuffled per query, logged (PriDe) | supervisor | config | position bias |
| Control options | continue / retry / reobserve / done / target_missing / abort | supervisor | config | — |
| Thresholds | argmax for now; ✱ τ_done, τ_fail, k consecutive, R retries (tuned on calibration episodes) | supervisor | config | over/under-steering |
| Speculation | on for arm D (π0.5² "continue" chunk computed during the decision) | `workflow.py` | config | latency |

## 5. Decision points and timing
| Lever | Current | Cost | Symptom it fixes |
|---|---|---|---|
| When arm D decides | every chunk boundary (50 steps = 2.5 s) | config | late detection of done/failure |
| Timing variant | T1 paused (sim waits for models) | config | latency effects (T2/T3 later) |
| Switch rule | S1 finish the current chunk | config | acting on a stale subtask |

## 6. Executor (π0.5²)
| Lever | Current | Where | Cost | Symptom it fixes |
|---|---|---|---|---|
| Checkpoint | `lerobot/pi052_robocasa` (PandaOmron, sim data) | cloud | swap / train | right subtask, failed motion |
| Precision | bf16 on T4 #0 (probe rel. err 0.29%) | cloud | config | — |
| Chunk / executed steps | 50 / 50 | checkpoint config | config | slow reaction mid-chunk |
| Flow steps | 10 | checkpoint config | config | action quality vs speed |
| Subtask text | sent verbatim; prompt `"{subtask}, State: …"` | adapter | config | wording the VLA ignores |
| Memory head | off | `drive.py` | config | — |
| Action dispatch | one action per env step | `drive.py` | config | skipped actions |

## 7. Simulation and evaluation
| Lever | Current | Cost |
|---|---|---|
| Task set | any of 300 pretrain tasks (arm D); 3 diagnosis tasks with subtask ground truth (`tasks.py`) | config |
| Split / objects | pretrain; objaverse + lightwheel | config |
| Seeds / episodes per arm | ✱ (start 1000; ≥ 380 per arm for significance) | config |
| Max steps per episode | 2320 / 1520 / 2370 (≈ 1.5 × p90 demo length); other tasks 2000 | config |
| Arm B budget | 5 / 5 / 4 chunks per subtask (p75 demo subtask duration) | config |
| Arm B/E sequences + wording | `tasks.py` (most frequent training phrasing) | config |
| Ground-truth checks | RoboCasa success primitives per subtask (`tasks.py`) | config |
| Arms | A own planner · B fixed · C generative VLM · D OneJev · E oracle | config |

## Order of fixes
1. Attribute each failure to one stage first (trace): wrong evidence → shortlist missing target → wrong choice → failed execution (arm E isolates the executor).
2. Change **config** levers before **swap** before **train**. Training is for a model whose knowledge is wrong, not a wrong setting.

## 9. v3 runtime (`rojev/control/`, `rojev/system1/`, `rojev/owner/`)
| Lever | Current | Where | Cost | Symptom it fixes |
|---|---|---|---|---|
| Chunk blend (inertialization) | `blend_s` 0.2 s, quintic decay, continuous dims only | `ExecutionRuntime` | config | jerk at command/chunk switches |
| S1 refill horizon | `refill_s` 1.0 s: lower bound of the lead time; the next chunk is requested when less than max(`refill_s`, L + dt) is queued | `S1Runner` / `realtime.py --refill` | config | first-request gap (too low; later leads follow measured L) or acting on older observations (too high) |
| Commitment policy | `fixed` (whole chunk); `phase`: through the first slow point after the floor (`slow_fraction` 0.3); never below the floor | `make_commit_policy` / `--commit-policy` (no flag for `slow_fraction`, `max_commit`) | config | reacting late near contact (fixed) |
| Speculative predict | off; on: K = `samples` 3 (use odd K), `tolerance` 0.5 × median speed, `smooth` 3 | `ConsensusCommit` / `--speculative-predict --samples --tolerance` | K × S1 latency today (sequential `act_samples`; batching = step 7, not built) → floor ≈ K× | committing to an uncertain or outlier chunk; wrong gripper decision |
| Execution monitor | off; on: window 0.5 s (+ previous window for lag), segment 5 s, α 0.1 per segment (≤ 1.2 false alarms / min); calibration split by run; runs below real-time factor 0.98 (`--min-realtime-factor`) or mixing policies refused; PandaOmron resolution 0.1 mm | `--execution-monitor`, `--monitor-record`, `calibrate_monitor.py --window-s --segment-s --alpha` | recalibration runs | blocked arm unnoticed (off); alarm noise (α too high); contact-blind (open, step 6) |
| Commitment floor + lead time | derived, not configured: floor ⌈2L/dt⌉ + 1 (whole chunk before the first sample), lead max(`refill_s`, L + dt), L = largest of the last 20 measured latencies (`latency_window`) | `S1Runner` | none | starvation (reported as `starved_ticks`) |
| ~~`min_commit`~~ | **removed 2026-10-21** (replaced by the derived floor; `make_commit_policy` rejects it) | — | — | — |
| S1 transport | HTTP/1.1 keep-alive over the SSH tunnel (`http://` url written by `gpu_setup.sh`); `h3://` + pinned `cert_pem` available, not default | service JSON `url` → `RemotePolicy` / `open_transport` | config | network time and tail per request |
| Frame codec | decided default `webp90` (decisions 2026-10-21), but `RemotePolicy`'s code default is `raw`: set `"codec": "webp90"` in the service JSON entry | `RemotePolicy(codec=)` / `wire.CODECS` | config (re-check actions vs sampling noise) | request size, loss tail |
| Command lease | `lease_s` 3.0 s, renewed by every CONTINUE decision | `StateOwner` / `realtime.py --lease` | config | robot acting on a command nobody confirms |
| Decision heartbeat | `decide_every_s` 1.0 s (plus events: instruction change, rejected / expired command) | `StateOwner` / `--decide-every` | config | slow reaction (too long) or wasted S2 calls (too short) |
| Proposal expiry | 2.0 s after the decision | `DecisionWorker` | config | acting on an old decision |
| On instruction change | `review` (keep reversible motion, freeze irreversible switches) or `revoke` (stop-and-go) | `StateOwner` / `realtime.py --on-change` | config | robot stopping on every rephrase (revoke) |
| Review window | `review_s` 1.0 s, then decelerating hold | `StateOwner` / `--review-s` | config | acting too long on the old goal (too long) or pausing (too short) |
| Interaction lane | none (one lane); `continue` baseline; deadline `interaction_timeout_s` 1.0 s then the planning lane | `StateOwner(interaction=…)` / `--interaction`, `--interaction-timeout` | one more S2 worker | routine planning overriding the response to a change; an unanswered change (deadline) |
| Decelerating hold | `stop_s` 0.3 s in `realtime.py` (library 0 = instant) | `ExecutionRuntime` / `--stop-s` | config | jerky stops |
| Starvation watchdog | `watchdog_s` 0.5 s → hold | `Guard` | config | robot coasting on stale/no chunks |
| Owner heartbeat timeout | `owner_timeout_s` 2.0 s → hold + cancel pending | `Guard` | config | acting with no supervisor |
| Transition on a new command | `pause_hold` (hold now) or `immediate` (switch at new chunk) | `AcceptedGoal.transition` | config | unsafe or slow switches |
| Robot action spec | per robot: bounds, hold value, switches, irreversible dims (PandaOmron: gripper) | `executors/*/embodiment.py` | config | wrong hold / gripper released on wait |
