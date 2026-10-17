# Method (design as decided; ✱ = to be measured)

## 1. Three levels
| Level | Example | Decided by | Rate |
|---|---|---|---|
| Task | "put the box in the fridge" | user | once |
| **Subtask** | open the refrigerator door → pick up the box → place the box in the refrigerator → close the refrigerator door | **OneJev** (replaces the VLA's own planner) | at subtask boundaries |
| Actions | 50-step chunks (2.5 s at 20 Hz) | frozen VLA low level (π0.5² [pi052_robocasa]) | every chunk |

- π0.5²'s low level is conditioned only on `"{subtask}, State: …"`. The generic task goes to OneJev, never to the low level.
- Injection: LeRobot's runtime subtask slot (`set_context("subtask", …)`, its direct-subtask mode); no fork, no subclass. π0.5²'s own planner stays available unchanged as the baseline.

## 2. Closed-loop supervisor
```
loop until done / abort / budget:
  observe   → camera images + robot state summary + completed-subtask history
  MJev      → spatial evidence (object located? door open? gripper holding?)      [qin2026mjevspatial]
  OneJev Q1 → "is the current subtask done?" (Noul), every chunk boundary
  OneJev Q2 → only if done/failed/uncertain: "what next?" (Choice over shortlist + control options)
  VLA       → executes the chosen subtask
```
- Order: MJev evidence first, then OneJev decides (decision 2026-10-17).
- Evidence questions use a neutral state (no instruction in the state). Reason: experiment E2, where instruction text biased both action and visibility answers.

## 3. Candidate builder (deterministic, no LLM)
Pool: 21,326 training subtasks (E8). Shortlist = top-K from A ∪ B ∪ C after D, plus control options.

| Source | Mechanism | Covers |
|---|---|---|
| A. Transition graph | P(next subtask \| last completed), from ordered subtask timelines of the training episodes | unmentioned intermediate steps |
| B. Similar training tasks | BM25 between the instruction and the 4,740 training task strings → their subtasks | task-implied prerequisites |
| C. Object match | subtasks mentioning objects in the task text or MJev evidence | the right objects |
| D. Canonicalization | one most-frequent phrasing per meaning | wording the low level saw most |

✱ **Status:** E9 builder is used as-is (measured, E9). Template slot-filling (E9b) raises reachability but not ranking; parked. The shortlist only needs recall; meaning/wording is OneJev's job.

✱ **Sweet spot K:** replay held-out training episodes; coverage@K = P(true next subtask ∈ shortlist). Choose the smallest K with ≥ 95% coverage; report the coverage curve.

## 4. Control options and thresholds
| Option | Fires when | Guard (over-steer) | Guard (under-steer) |
|---|---|---|---|
| continue | default while the subtask runs | — | per-subtask step budget |
| done → next | P(done) > τ_done for k consecutive checks | debounce: k checks + minimum dwell [zhang2026robocombine] | — |
| retry | P(failed) > τ_fail or no progress within budget | ≤ R retries per subtask | budget forces a decision |
| reobserve | evidence uncertain (high entropy) | ≤ N consecutive | forced decision after N |
| target_missing | P(target visible) < τ_vis on k consecutive checks, after reobserve | requires reobserve first | k-check confirmation |
| abort | target missing confirmed, or retries/global budget exhausted | — | hard safety stop |

✱ Thresholds are tuned on calibration episodes; sweeping them gives the **risk–coverage curve** (probabilistic headline). Option order shuffled and logged [zheng2023pride]; conformal variant [ren2023knowno].

## 5. Decision timing variants (`playground/timing/`)
- **Timing:**
  - T1 paused (sync);
  - T2 latency-injected (deterministic);
  - T3 async wall clock with version + age checks.
- **Switching:**
  - S1 finish the chunk (default);
  - S2 discard the queue (abort/success only);
  - S3 fewer actions per chunk;
  - S4 debounce.
- **Guarantees:** the low level never waits on the high level; an empty queue → hold; deadlines; one owner of subgoal state.

## 6. Arms (identical low-level weights, candidates, seeds)
| Arm | Subtask chooser |
|---|---|
| A | π0.5²'s own planner |
| B | fixed human-authored sequence |
| C | generative VLM, LeRobot planner port, Qwen3.5-0.8B (OneJev's parent) |
| D | OneJev (+ MJev evidence) |
| E | oracle subtask |

## 7. Metrics
- Ordered success, per-transition conversion, success@k.
- Selector accuracy (set-based); coverage@K.
- False completion; recovery success.
- Calibration (Brier, reliability bins) and risk–coverage.
- Latency per decision (MJev / OneJev separately) and staleness.
- Language-following controls.
- Paired seeds; confidence intervals.
