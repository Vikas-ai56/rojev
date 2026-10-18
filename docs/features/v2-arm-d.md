# Arm D: OneJev decides, MJev judges the candidates

Arm D is the deployable supervisor. It picks the next subtask for π0.5² from the **current images** at every 50-step chunk boundary. No per-subtask or per-object code.

## One decision
```
current images + task + subtasks so far
  │
  ├─1 candidates   shortlist from π0.5²'s training subtasks (E9 filter, K = 20)
  │
  ├─2 evidence     MJev (cloud GPU 1), one template for every judged subtask s:
  │                  "Robot step: '<s>'. Which is true right now?"  → done / possible / blocked
  │                judged = current + 2 most recent + top M = 5 candidates   (one batched request)
  │
  └─3 decision     OneJev (Mac), one Choice on the 3-camera image, evidence table in its state:
                     continue current | any candidate | redo a recent one | task complete
```
- **Why one template is enough.** Object missing → its "pick up …" is *blocked*. Door already open → "open …" is *done*. Object dropped → "pick up …" is *possible* again and "place …" *blocked*. No case-specific questions.
- **Why recent subtasks are judged.** Undone progress (a drop) shows up as a past subtask no longer *done*; OneJev can redo it. Nothing is trusted from the previous step.
- **Speed.** While steps 2–3 run, π0.5² (cloud GPU 0) already computes the "continue current" chunk; it is used if the decision is *continue* (speculation hit).

## Example (MicrowaveThawingFridge, chunk 6)
| Judged subtask | done | possible | blocked |
|---|---|---|---|
| pick up the steak (current) | 0.10 | 0.80 | 0.10 |
| place the steak in the microwave | 0.05 | 0.15 | 0.80 |

OneJev → *continue: pick up the steak*. (Illustrative numbers, not a measurement.)

## What is logged per decision (`chunks.jsonl`, Phoenix span `onejev.next`)
- `evidence`: the done/possible/blocked table; `judged`; `options` (shuffled order); `chosen`; `top5`.
- `saycan_top3`: P_OneJev(option) × P_MJev(possible), a second decision rule at no extra cost [ahn2022saycan].
- `evidence_vs_truth`: MJev's *done* judgement vs the simulator's ground truth, for judged subtasks that are steps of the task; episode summary has `mjev_done_accuracy`.
- `timings_ms`: `mjev`, `onejev`, π0.5² `infer`, `network`, `decide`, `wait_after_decide`; speculation hits/misses.

## How it is checked
- MJev was not trained on "done / possible / blocked". Its accuracy is measured against simulator ground truth (`evidence_vs_truth`), never assumed.
- If weak, the same simulator labels become training data (MJev and OneJev both take typed JSONL records); no rules are added.

## Settings (levers, `v2-levers.md`)
`--k` shortlist size (20) · `--m` judged candidates (5) · recent judged (2) · `--no-mjev` ablation · option shuffle seed = episode seed.

## Limits
- Only shortlisted subtasks can be chosen; the filter's coverage (E9) caps arm D.
- "possible" is fuzzy for open-ended steps ("transport the steak"); visible in the calibration numbers.

## Run
```
make qev                                                         # OneJev on the Mac
python executors/robocasa-pi052/kaggle/push.py push armd         # MJev (GPU 1) + π0.5² (GPU 0); add `validate` once
cd executors/robocasa-pi052 && uv run python workflow.py --task MicrowaveThawingFridge --arm D --service kaggle
```
Code: `rojev/supervisor/onejev_arm.py` · `rojev/perception/mjev.py` · `services/mjev/serve_mjev.py` · `executors/robocasa-pi052/workflow.py`.
