# v2 G0: frozen executor (RoboCasa + SmolVLA)

Plan: `docs/onejev_learned_policy_plan_v2.md` §5 G0.

## Done
- **0.1 Sources pinned:** `config/lerobot.commit` (0.6.2), `config/robocasa.commit` (1.0.1), `config/robosuite.commit` (1.5.2). All Apache/MIT-style `LICENSE` files are present (read before copying any code).
- **Executor env:** `executors/robocasa/` is its own uv project (`uv sync` there). It's separate because the pins conflict with the `rojev` env: robosuite needs `mujoco<3.10`, RoboCasa pins `mujoco==3.3.1` / `numpy==2.2.5`, LeRobot pins `transformers<5.6`. RoboCasa's `lerobot==0.3.3` pin is removed through `[[tool.uv.dependency-metadata]]`, the uv form of LeRobot's documented `--no-deps` route.
- **Resolved:** mujoco 3.3.1, numpy 2.2.5, torch 2.11.0 (**MPS available**), transformers 5.5.4, tianshou 0.4.10, gymnasium 1.4.0. `SmolVLAPolicy` and `lerobot.envs.robocasa` import cleanly. `setup_macros` has been run.

## Facts for the experiment
- The checkpoint is trained on `robocasa_target_human_unified` (50 tasks, 25,307 episodes, 20 fps) for 20k steps at batch 32. **No success rates are published.** LeRobot's CI runs only a 1-episode smoke test of 10 atomic tasks: CloseFridge, OpenCabinet, OpenDrawer, TurnOnMicrowave, TurnOffStove, CloseToasterOvenDoor, SlideDishwasherRack, TurnOnSinkFaucet, NavigateKitchen, TurnOnElectricKettle.
- **Inputs:** 3 cameras at 256×256 (left / wrist / right → `camera1/2/3` via `rename_map`) and a 16-number robot state. **Output:** 12-number action in [-1, 1] (base 4, control mode 1, end-effector position 3, rotation 3, gripper 1). The control-mode entry means an all-zero action is not a safe hold.
- `chunk_size = n_action_steps = 50`: 2.5 s of open-loop actions per chunk.

## Downloads (2026-10-17, per decisions)
| Item | Size | Status |
|---|---|---|
| `lerobot/smolvla_robocasa` @f7af588d11e3 | 0.91 GB | done |
| SmolVLM2-500M config/tokenizer only @7b375e1b73b1 | ~0 GB | done |
| RoboCasa assets (tex, tex_generative, fixtures_lw, objs_lw) | 3.05 GB (4.4 GB unpacked) | done |
| `Qwen/Qwen3.5-4B` @851bf6e shard 2 (vision) + configs | 4.01 GB | done |
| `mindlab-research/MJev-Spatial-4B` @0657319c58d3 | 16.45 GB | done |
| RoboCasa objaverse objects | 2.16 GB (6.1 GB unpacked) | done (target split needs its categories) |
| π0.5-robocasa | 9.35 GB | not downloaded (Kaggle later) |

## Verified
- `check_policy_load.py`: SmolVLA loads **strictly** (zero missing/unexpected keys) from the final weights with `load_vlm_weights=False`. 450.0M params (bf16 + fp32), MPS allocated 1.20 GB, process peak 2.92 GB, 16 s.

- **G0-A** (`lerobot-eval`, CloseFridge, 1 episode, MPS, final weights only): ran end to end. **0/1 success** (95% CI 0–79%), 900 steps at ~27 steps/s, episode 58.5 s, peak RAM 3.39 GB. Video `results/runs/g0a_smolvla_closefridge_20261017-140555/`. Frames upright; the arm reaches the fridge door but doesn't close it, consistent with policy failure rather than a camera/orientation bug.
- `lerobot-eval` also needs LeRobot's `dataset` extra (added via `uv add "lerobot[smolvla,dataset]"`); the documented install omits it.

- **macOS: parallel envs (`--eval.use_async_envs=true`) crash.** Worker processes die during the kitchen reset (macOS GL errors: `NSXPCSharedListener … Connection invalid`, `Fallback to SW fragment processing`). Run episodes sequentially in one process on the Mac; parallel envs only on Linux/Kaggle.
- G0-C runner: `executors/robocasa/capability_table.py` (one job per task, rows saved as each task finishes, time-budget guard, Ctrl-C safe, free-RAM/swap sampling).

- **Bug found: camera mapping.** LeRobot's RoboCasa docs/CI map eye_in_hand→camera2, right→camera3; `smolvla_robocasa` was **trained** with right→camera2, eye_in_hand→camera3 (its `train_config.json` / `policy_preprocessor.json`). `lerobot-eval` replaces the stored map with `--rename_map` (default empty, `lerobot_eval.py:810`). Fix: the runner reads the map from the checkpoint. **G0-A and the first G0-C runs (`g0a_*`, `g0c_20261017-151400`, `g0c_20261017-151517`) are invalid** (wrist/right swapped).
- **Bug found: scene split.** For a single task, LeRobot's RoboCasa env uses `split="all"` (`envs/robocasa.py:230`); `smolvla_robocasa` trained on the `target` split. Runner now passes `--env.split=target` (default) and saves `run_config.json`. Run `g0c_20261017-152512` (cameras fixed, split=all; CloseFridge 0/5) is also confounded.
- Image orientation verified: training-data frames and our frames are both upright (`dsprobe/orientation_check.png`).
- **Bug found: instruction text.** Trained on task names (`"CloseFridge"`, dataset `tasks.parquet`); `lerobot-eval` sends `ep_meta["lang"]` (`"Close the fridge door(s)."`). Fix: `executors/robocasa/rollout.py` (mirrors lerobot-eval; instruction from the checkpoint's training vocabulary). Contract: `docs/paper/appendix-engineering.md (§A)`.
- **A/B, CloseFridge, same seeds 1000–1004, all fixes applied:** task name **1/5** (seed 1002, 245 steps) vs sentence **0/5**. Only seed 1002 differs. Runs: `results/runs/rollout_CloseFridge_{taskname,lang}_*`.
- Reading: the instruction matters, but even correctly fed, the checkpoint is weak on CloseFridge (1/5, 95% CI ≈ 4–62%).
- **SmolVLA, corrected setup** (task-name instruction, checkpoint cameras, target split, seeds 1000–1004):

  | Task | Success | Objects | Run |
  |---|---|---|---|
  | CloseFridge | 1/5 | lightwheel | `rollout_CloseFridge_taskname_20261017-154211` |
  | OpenDrawer | 1/5 | lightwheel | `rollout_OpenDrawer_taskname_20261017-155446` |
  | TurnOnSinkFaucet | 1/5 | lightwheel | `rollout_TurnOnSinkFaucet_taskname_20261017-155842` |
  | TurnOnMicrowave | 1/5 (lightwheel-only crashed at reset: no valid object category) | objaverse+lightwheel | `rollout_TurnOnMicrowave_taskname_20261017-160410` |

  Reading: ~20% per task (each 95% CI ≈ 4–62%); too unreliable as the executor for supervisor experiments.

## Next
- G0-B: MJev (after MLX port) on RoboCasa frames. G0-C: capability table, 10 CI tasks.
