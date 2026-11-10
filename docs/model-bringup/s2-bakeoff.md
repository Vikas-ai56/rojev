# S2 bake-off: Embodied-R1.5 on real SO-101 data

Decision: `decisions.md` [2026-11-09]. Status: pipeline built and rehearsed on Linux; not run on a GPU yet.

## What is tested
- **Planner:** the next step from the skill list (RoboVQA question form + the allowed steps).
- **Done checker:** R1.5's process-detection form "`<step>`. is it satisfied?" (yes / no). Its Corrector prompt is not public.
- One model, no comparator: each figure is checked against pass bars fixed beforehand (`config/s2_pass_bars.yaml`).

## Case sets (`data/s2-cases/`, regenerated; labels committed in `data/s2-labels/`)
| Set | Source | Cases | Tests |
|---|---|---|---|
| `so101-steps` | `hungho77/…open_drawer_pick_red_cube_stack_on_blue_cube` (SO-101, Apache-2.0), 12 episodes, step ends from gripper plateaus, checked by eye | 120 next step (48 true ledger, 48 no ledger, 24 contradicting ledger), 108 status (60 done, 48 in progress) | Planner reads the scene, not just the list; contradicting ledger = silent-failure check; same frame asked about two steps with opposite truths |
| `armnetbench` | `armnet/armnetbench_v01_lerobot_so101` (SO-101, π0.5 rollouts, Apache-2.0), operator labels | 218 whole attempts (109 done, 109 failed), 8 tasks | Real failures; cameras `top` (fixed front), `overhead`, `wrist` |

- Built by `executors/so101-data`: `cases.py label / build`, `rollouts.py sample / build`.
- Frames: the asked frame alone, per camera: the newest frame, as the live planner and done checker get it. The
  planner's sense of the past is its step list (text), shown or withheld by the case variant.
- Next-step questions show the top camera; done-check questions are asked once per camera, and a yes from either
  camera counts as done (the live done checker uses the same rule).

## Run
1. `scripts/gpu_setup.sh HOST --services planner-vlm --skip-serve`: env, weights (`config/embodied_r1_weights.txt`), case sets (~85 MB) to `~/rojev-data/s2-cases/`.
2. `scripts/gpu_session.sh HOST --plan config/gpu_session_s2.txt`: one R1.5 load, 990 questions in vLLM batches, then single-question latency.
3. On the Mac: `uv run python score.py --set ../../data/s2-cases/so101-steps <results>/so101-steps --set ../../data/s2-cases/armnetbench <results>/armnetbench --out report.json` (in `executors/so101-data`).

## Scoring
| Measure | How | Why |
|---|---|---|
| Next-step accuracy per ledger variant | Exact match to an allowed step; anything else wrong | Unparsable = unusable by the runtime |
| False-done rate | Said done / truly not done (in progress or failed) | The dangerous error: the robot moves on |
| Missed-done rate | Not done / truly done; a malformed answer counts here | Costs time only |
| Balanced accuracy, per goal | 1 − mean of the two rates, averaged over goals with both truths | Done/not-done mixes differ; a goal with only failures cannot reward a task prior |
| Paired steps | Same frame, two steps, opposite truths: both right | Catches judging the image without reading the step |
| Latency | p50 / p95 single question; tokens for right vs wrong answers | Event-driven S2 budget; long reasoning on wrong answers (FailBench) |

- Per camera and "either camera says done".
- 95% intervals from resampling whole episodes; verdict `uncertain` when an interval crosses its bar.
- Dry run with a random stand-in: every accuracy bar fails (next step ≈ 0.3, balanced accuracy ≈ 0.5).

## Result: Embodied-R1.5 fails (2026-11-10, E34)
| Bar | Needed | R1.5 |
|---|---|---|
| Next step, true step list | ≥ 0.90 | **0.63** (no list 0.21; a list that wrongly claims a step: 0.00) |
| Done check, SO-101 steps, either camera | false done ≤ 0.10, balanced ≥ 0.75 | 0.38, **0.50** (chance) |
| Done check, ArmNetBench real failures, either camera | false done ≤ 0.10, balanced ≥ 0.75 | 0.11 (uncertain), **0.69** |
| Latency p50 | ≤ 3 s | 0.12–0.25 s |

- It answers in about 8 tokens with no reasoning, though the question asks it to think first; our wording, template
  and settings match its official example, so this is how it behaves.
- It always trusts the step list: when the list claims a step the image shows undone, it never redoes it.
- Its first-step errors repeat GR00T's on LIBERO: it skips opening the drawer and goes for the object.

## Known limits
- 12 episodes of one task: intervals about ±10–14 points.
- ArmNetBench: 218, not 300 (only 109 single-arm π0.5 successes); successes end at the operator's success time, failures run to the attempt's end (different frame spacing); `top` faces a wall in the eye-drops and cable tasks (`overhead` is kept for them); cable clip has no successes.
- Not FailBench's exact slice (its IDs are unreleased).
