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

## 3. Candidate filter (`rojev/supervisor/candidates.py`)
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
| Model | OneJev 1.5 0.8B Q8_0 GGUF, deterministic | Mac, qev + llama.cpp | swap / train | wrong choices with right shortlist |
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
