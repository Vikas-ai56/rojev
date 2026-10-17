# Appendix: engineering facts needed for reproducibility

Keys → `references.bib`. Paths are relative to the repo root.

## A. SmolVLA executor contract and evaluation pitfalls [shukor2025smolvla, smolvla_robocasa]
Sources: LeRobot `docs/source/smolvla.mdx`, `policies/smolvla/{modeling,processor}_smolvla.py`, the checkpoint's `config.json` / `train_config.json` / `policy_{pre,post}processor.json`, dataset `lerobot/robocasa_target_human_unified` `meta/info.json` + `meta/tasks.parquet`. Paper: arXiv 2506.01844.

#### Model
- 450M params: SmolVLM2-500M backbone cut to 16 layers + flow-matching action expert (width ×0.75, cross-attention, self-attention every 2nd layer).
- One forward per chunk: 10 denoising steps → **50 actions** (`chunk_size = n_action_steps = 50`, 2.5 s at 20 fps), executed open-loop.
- Trained 20k steps × batch 32 on `robocasa_target_human_unified` (50 tasks, 25,307 episodes, 15M frames, 20 fps): ≈ 4% of one epoch. No success rates published.

#### Inputs (per step, batch of 1)
| Key the policy sees | From env key | Format | Processing |
|---|---|---|---|
| `observation.images.camera1` | `robot0_agentview_left` | 256×256 RGB uint8 | → float [0,1] → resize-with-pad 512×512 → [-1,1] (SigLIP); no mean/std |
| `observation.images.camera2` | **`robot0_agentview_right`** | same | same |
| `observation.images.camera3` | **`robot0_eye_in_hand`** | same | same |
| `observation.state` | `agent_pos` (16-D: base pos, base quat, relative EE pos, relative EE quat, gripper qpos) | float32 [16] | mean/std normalised, zero-padded to 32 |
| `task` | instruction text | string | `"\n"` appended, tokenised (SmolVLM2), **max 48 tokens**, right-padded |

#### Output
- `action`: 12-D in [-1,1] after un-normalising (mean/std): base motion 4 · control mode 1 · EE position 3 · EE rotation 3 · gripper 1.
- Control-mode entry → an all-zero action is **not** a safe hold.

#### Mismatches found in LeRobot's documented eval path (all fixed in `executors/robocasa/`)
| ### | What | Trained with | `lerobot-eval` / docs send | Fix |
|---|---|---|---|---|
| 1 | Camera mapping | right→camera2, wrist→camera3 | wrist→camera2, right→camera3 | read from checkpoint (`checkpoint.rename_map`) |
| 2 | Scene split | `target` | `all` for a single task | `--env.split=target` |
| 3 | **Instruction** | task names (`"CloseFridge"`) | natural language (`"Close the fridge doors."`) | `rollout.py` sends the task name, checked against the training vocabulary |
| — | Image orientation | upright | upright | none needed (verified) |

#### Read these yourself
- Paper: https://arxiv.org/abs/2506.01844
- Model code: `vendor/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py` (`prepare_images` L373, `prepare_state` L448)
- Processor pipeline: `vendor/lerobot/src/lerobot/policies/smolvla/processor_smolvla.py`
- Checkpoint files: https://huggingface.co/lerobot/smolvla_robocasa/tree/main (`config.json`, `train_config.json`, `policy_preprocessor.json`)
- RoboCasa in LeRobot: `vendor/lerobot/docs/source/robocasa.mdx`; env: `vendor/lerobot/src/lerobot/envs/robocasa.py`

## B. π0.5² executor [pi052_robocasa]
- 4.14B parameters (2.82B bf16 + 1.32B fp32), ≈ 11 GB; doesn't fit the Mac's ~9 GB budget → Kaggle 2×T4.
- bf16 on T4 (no native bf16): matmul probe runs, rel. error 0.29% vs fp32 (2026-10-17); full-policy check pending. Fallback = fp32 over 2×T4.
- Steering point: LeRobot runtime subtask slot; low-level prompt `"{subtask}, State: …"` (non-joint mode).
- Hub dependencies at inference: PaliGemma tokenizer (gated, Gemma licence) and `lerobot/fast-action-tokenizer` (FAST step is built from the saved preprocessor but is a no-op without actions).
- LeRobot language runtime with `ctrl_hz>0` drops queued actions under inference lag (real-robot catch-up) → use one action per env step in sim.
