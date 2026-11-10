# GR00T N1.7 on the SO-101: bring-up sheet

S1 candidate for the real SO-101 (decision 2026-11-07). Served by `executors/so101-groot` (`groot_io.py`, `serve.py`,
`gpu/hook.sh`). Contract read from Isaac-GR00T source @ `9fbe86a` (2026-11-06); re-read it on any pin change.

## Contract
| Item | Value | Source (Isaac-GR00T) |
|---|---|---|
| Model | ~3B: Cosmos-Reason2-2B VLM (Qwen3-VL) + flow-matching DiT action head; embodiment `NEW_EMBODIMENT`, fine-tune required (no zero-shot SO-101) | `getting_started/policy.md` |
| Cameras | `front`, `wrist`: uint8 RGB `(1, 1, H, W, 3)`, current frame only; any size (256 shortest edge, 0.95 crop inside) | `examples/SO100/so100_config.py`, `processing_gr00t_n1d7.py:225` |
| State | `single_arm` (5) + `gripper` (1), float32 `(1, 1, D)`, order `shoulder_pan … wrist_roll, gripper` | `gr00t/eval/real_robot/SO100/eval_so100.py:111-139` |
| Units | the fine-tuning data's: ours are LeRobot's (body degrees, gripper 0–100); `rojev.robot.so101.LeRobotUnits` converts | decision 2026-11-07 |
| Language | `{<key>: [[text]]}`, `<key>` from the checkpoint: `annotation.human.task_description` (whole-task SO100 recipe) or `sub_task` (step-trained: labels from `episodes.jsonl` `sub_tasks`); we send the step text, else the task | `eval_so100.py:142`, `lerobot_episode_loader.py:594` |
| Actions | `single_arm (1, 16, 5)` + `gripper (1, 16, 1)`, absolute (arm decoded from relative with the input state) | `gr00t/policy/gr00t_policy.py:381-428` |
| Step labels (training) | `episodes.jsonl` `sub_tasks: [{start, end, text}]` | `gr00t/data/dataset/lerobot_episode_loader.py:543` |
| Load | `Gr00tPolicy(model_path, NEW_EMBODIMENT, device)`: model cast to bf16; backbone and processor built from the gated repo id `nvidia/Cosmos-Reason2-2B` | `gr00t_policy.py:106-124`, `qwen3_backbone.py:183` |
| Env | Python 3.12, torch 2.9.0, transformers 4.57.3, flash-attn 2.8.3 (SDPA fallback if missing); Ampere+ GPU, 16 GB+ for inference, 40 GB+ to fine-tune | `pyproject.toml`, `qwen3_backbone.py:168-178`, README |

## Our adapter (thin binding over GR00T's own API)
- **From the model:** keys and horizons come from `Gr00tPolicy.get_modality_config()`. Validation is GR00T's own `strict` mode (`--strict`, on by default; policy.md says keep it on while developing).
- **From us:** `executors/so101-groot/modality.json`, in GR00T's dataset-layout format, which our datasets will carry. It gives each state/action key's joint slice and each camera key's `observation.images.<robot camera>`.
- **Load check:** `Binding.of` fails at load if the two disagree, if there is history, if there is more than one language key, or if the joints aren't covered.
- **Ours on top:** radians ↔ LeRobot units; a non-finite action fails the request, since GR00T's check covers keys, shapes and dtypes only (#791).
- **Serving:** `GrootPolicy` behind `PolicyService` + `Endpoint` on 127.0.0.1:8005 with a bearer token, not GR00T's ZeroMQ `PolicyServer`:
  - one transport and replay cache for every S1;
  - a timed-out request is simply abandoned, whereas a ZeroMQ REQ socket must receive before it sends again.
- **No RTC:** `accepts_continuation` is False, since GR00T's policy path has none.
- **Environment = NVIDIA's tested versions.** `pyproject.toml` `constraint-dependencies` come from Isaac-GR00T's own `uv.lock` at the pin (`python -m gpu.constraints <checkout>`, then `uv lock`).
  - Resolving its `pyproject` fresh drifted on 72 of 139 packages. Among them were pyarrow 26 (needs NumPy 2, so every parquet read failed against GR00T's NumPy 1.26.4) and TensorRT 11 for 10.
  - Now 0 of 140 differ; only rojev's own extras (mujoco, aioquic) sit on top.
  - Video decoding is torchcodec on the system FFmpeg (4–7), which `gpu_box.sh` installs.
- GPU hook (`gpu/hook.sh`):
  - the gated backbone is **downloaded on the box with your Hugging Face login**. The Mac's token file (checked to be mode 600) is copied as a file over SSH to the box's `~/.cache/huggingface/token`, kept at 600, and never printed or put on a command line. `scripts/hf_gated.py snapshot` downloads the pinned revision and writes its cache ref. GR00T runs online (transformers 4.57.3 queries the Hub while loading the tokenizer even with `HF_HUB_OFFLINE=1`); `scripts/hf_gated.py snapshot --check` stops a run if the Hub's main no longer equals the pin. Verified on Linux that the token file alone authenticates. Uploading 4.9 GB from the Mac was the main setup cost;
  - the official base `nvidia/GR00T-N1.7-3B` is pre-downloaded (`config/groot_base_weights.txt`);
  - Isaac-GR00T is a vendor checkout at `config/isaac-groot.commit`, for its own tools. Vendors clone LFS files as pointers; the hook installs LFS for that repo and pulls only `demo_data/droid_sample` and `demo_data/cube_to_bowl_5`. A plain `lfs pull` silently skips writing files when LFS isn't installed for the repo, which a real clone showed;
  - readiness imports the flash-attn kernels and loads torchcodec's decoders;
  - every preparation step works without a fine-tune. `serve` needs `$GROOT_CHECKPOINT` (a box directory) or a pin in `config/groot_weights.txt`; until then use `gpu_setup.sh --skip-serve`.
- **GPU session** (`gpu/session.sh`, run by `scripts/gpu_session.sh` from `config/gpu_session.txt`). Session 1 uses the official base, zero-shot, on NVIDIA's DROID sample; no rojev code is in the measured path.
  - `droid_reference`: `standalone_inference_script.py` exactly as `examples/DROID/README.md` documents it (episodes 1 2, horizon 8; episode 0 has an empty instruction). The result is checked against NVIDIA's published numbers: MSE ≈ 0.0149, MAE ≈ 0.0753, ~262 ms per step on an H100.
  - `droid_latency`: `benchmark_inference.py`, eager and `torch.compile`.
  - `droid_probe`: our `probe.py`, input and output variety in one load:
    - 8 time points per episode;
    - the episode's instruction plus `gpu/droid_variants.json` (paraphrase, other target, unrelated, empty);
    - 5 seeds;
    - 4 and 8 denoising steps.

    It reports action error, spread across seeds, how far each instruction moves the actions, and latency. A failing variant is recorded, not fatal. Every chunk goes to `actions.npz`.
  - **SO-101 steps** (`so101_finetune`, `so101_openloop`, `so101_parity`: our binding and service against GR00T called directly, plus a swapped-instruction probe) need a fine-tune. They wait for our own demos and are not in the session 1 plan.

## Measured (2026-11-07, A100-SXM4-40GB, base N1.7-3B zero-shot on the DROID sample, episodes 1–2)
Results: `results/gpu-session/20261107-183203/`.

| Check | Result |
|---|---|
| Reference (`standalone_inference_script.py`, horizon 8) | MAE 0.0765 (NVIDIA 0.0753), MSE 0.0201 (NVIDIA 0.0149; episode 2 alone 0.037) |
| Latency (`benchmark_inference.py`, 4 denoising steps, 40-action chunk) | eager 157 ms, `torch.compile` 89 ms (backbone 56, action head 23) |
| Denoising 4 vs 8 (probe, eager) | same error, p50 171 vs 261 ms: keep 4 |

Instruction probe (`probe.py`: 8 points × 2 episodes × 5 seeds × 2 denoising settings; distances in action units):

| Instruction | Error vs recorded | Moved from original | Seed spread |
|---|---|---|---|
| original | 0.164 | — | 0.053 |
| paraphrase | 0.167 | 0.039 | 0.055 |
| other target | 0.185 | 0.081 | 0.056 |
| unrelated | 0.192 | 0.113 | 0.063 |
| empty | 0.183 | 0.094 | 0.053 |

- Paraphrases move actions less than seed noise; wrong or empty text moves them 1.5–2× noise and raises error ~15%.
- The scene dominates: a contradicting instruction nudges, it does not redirect. The base model is not step-steerable zero-shot; step following needs a fine-tune on step-labelled demos, then this probe again.

## Not validated (needs the arm or a fine-tune)
- **Parity:** our binding and service against GR00T called directly (`so101_parity`, needs a fine-tune).
- **Step following after fine-tuning:** the probe above on SO-101 step-labelled data. #793 (overlapping step labels) and #585 (ignored prompts) still apply.
- **Units:** the real arm's sign, offset and gripper span, taken from its calibration.
