# v2 workflow: live episodes, sim on the Mac, π0.5² in the cloud (2026-10-18)

## Split
```
Mac                                              Kaggle 2×T4 (notebook rojev-armd; rojev-pi052 = π0.5² only)
RoboCasa sim, 1 env, live window   ──HTTPS──►    tunnel 1 → serve.py        (π0.5², GPU 0)
arm D: OneJev (qev, local) + filter ──HTTPS──►   tunnel 2 → serve_mjev.py   (MJev-Spatial, GPU 1)
Phoenix traces + files
```
- Arm D overlaps the two GPUs: while MJev + OneJev decide, π0.5² computes the "continue current subtask" chunk (speculation); used on "continue", else discarded and re-requested. Hits/misses logged.
- One request per 50-step chunk: 3 × 256² RGB (lossless, compressed .npz) + 16-D state + task + subtask → 50 × 12-D actions.
- Synchronous: the sim waits for the answer (timing T1).
- Contract: `rojev.executor` (`Observation`, `ActionChunk`, `Executor`; `RemoteExecutor`; `wire`). A real robot later replaces the sim behind the same contract.

## Files
| File | Role |
|---|---|
| `rojev/executor/{base,wire,remote}.py` | executor contract, lossless wire format, HTTPS client (bearer token) |
| `rojev/supervisor/base.py` | arm contract every arm implements |
| `rojev/eval/baselines.py` | evaluation-only arms B, E (E reads simulator ground truth) |
| `executors/robocasa-pi052/serve.py` | π0.5² service: env obs → LeRobot preprocess → PI052 adapter (given subtask) → postprocess per action |
| `executors/robocasa-pi052/tasks.py` | task set: authored subtask sequences + ground-truth checks (RoboCasa success primitives) |
| `executors/robocasa-pi052/workflow.py` | Mac runner: live window, arm loop, Phoenix spans, logs, video |
| `executors/robocasa-pi052/kaggle/` | `push.py push serve` (π0.5² only) / `push.py push armd [validate]` (both GPUs); `push.py service <mode>` prints URLs + tokens |
| `rojev/supervisor/onejev_arm.py` | arm D (`v2-arm-d.md`) |
| `rojev/perception/mjev.py` | `RemoteMJev` client (PNG, bearer token) |
| `services/mjev/` | MJev env (torch 2.9, transformers 5.17.0), `serve_mjev.py`, `check_precision.py`, `testdata/` |

## Run
1. `make phoenix` (trace UI at http://localhost:6006).
2. `python executors/robocasa-pi052/kaggle/push.py push serve` (≈ 10 min to ready; idles out after 30 min without requests).
3. `cd executors/robocasa-pi052 && uv run python workflow.py --task MicrowaveThawingFridge --arm E --service kaggle`
   - Arm D: see `v2-arm-d.md`.
   - `--executor hold`: zero actions, no GPU (checks loop, window, logging).
   - Window keys: `q` abort, `p` pause.

## What is logged (per episode)
- **Live window:** 3 policy cameras; task; arm, step, chunk, status; subtask i/n + the arm's reason; timings; ground-truth checklist with the step each subtask was first done.
- **Phoenix** (project `rojev-workflow`, one trace per episode): `decide:<arm>` (context → decision) → `executor.act` (subtask, state, camera image sent, timings) → `sim.chunk` (steps, ground truth, success).
- **Files** `results/runs/workflow/<task>_<arm>_s<seed>_<time>/`: `run_config.json`, `chunks.jsonl` (one line per chunk), `summary.json`, `episode.mp4`.
- Timings per chunk: `preprocess`, `infer`, `postprocess` (server), `network`, `round_trip`, `request_kb`.

## Arms
| Arm | Subtask choice | Switch rule |
|---|---|---|
| B | authored sequence (`tasks.py`) | fixed budget: `chunks_per_subtask` chunks each |
| E | same sequence | ground truth now: next = after the latest subtask whose check holds (forward, skip, or back after a drop) |
| D | OneJev + MJev candidate judgements — see `v2-arm-d.md` | one OneJev choice per chunk: continue / switch / redo / complete |

Ground truth (`tasks.py`) is an evaluation instrument, never supervisor input: scoring, arm-E ceiling, and labels for training OneJev's done-detection. It is computed for **every** arm, every step → failure attribution: subtask chosen but never done = executor; done but arm kept going (B) or wrong choice (D) = arm.

## Tasks
- Arm D runs on **any** RoboCasa pretrain task (`--task`, 300 choices), scored by RoboCasa's task-success check; default step limit 2000 (`--max-steps`).
- `tasks.py` (evaluation only) adds per-subtask ground truth and arms B/E for three diagnosis tasks:

## Diagnosis tasks with authored sequences (doors start open in all three)
| Task | Sequence | B budget | Max steps |
|---|---|---|---|
| MicrowaveThawingFridge | pick up {meat} → place {meat} in the microwave → close the microwave door → press the start button | 5 chunks | 2320 |
| MoveFridgeToFreezer | pick up {item} → place {item} in the freezer → close the freezer door → close the refrigerator door | 5 | 1520 |
| LoadCondimentsInFridge | [move top-shelf distractors to the middle shelf] → pick up / place each condiment on the top shelf | 4 | 2370 |

- Budgets = p75 demo subtask duration ÷ 50 (184 / 230 / 243 steps); max steps ≈ 1.5 × p90 demo length (1011–1579).
- Wording = most frequent phrasing in π0.5²'s training subtasks; "place … on the middle shelf" is not in the vocabulary.

## Known weak points (found while building)
- "pick up" check = grasped AND no contact with its source fixture / container (grasp contact alone fires on the plate; E10 ep 1).
- π0.5²'s training subtask labels are noisy: "open the refrigerator door" appears although doors start open; junk labels ("place the chicken drumstick on the floor", "move the robot arm").
- Local sim speed: 26.9 steps/s incl. 3-camera rendering (M5); scene build 7–10 s.
