# v2 executor: π0.5² (`lerobot/pi052_robocasa`)

## Set up (2026-10-17)
- Code: LeRobot branch `feat/smolvla-on-steerable` @9d00293 (`config/lerobot-steerable.commit`) → `vendor/lerobot-steerable`.
- Env: `executors/robocasa-pi052/` (own uv project; `lerobot[pi,dataset]`, robocasa, robosuite; same robocasa pin override). `PI052Policy` imports; torch 2.11, transformers 5.5.4, mujoco 3.3.1.
- Access: PaliGemma tokenizer (gated) works with the user's HF login.
- Model: 4.14B params (2.82B bf16 + 1.32B fp32), Apache-2.0. Not downloaded (doesn't fit the Mac; runs on Kaggle).

## How it is steered
- Low-level actions are conditioned on `"User: {subtask}, State: …"` only.
- Its own planner = `_generate_low_level_subtask` (regenerated every chunk from images + task only; memory off). Arm A = unchanged.
- OneJev arm = runtime subtask slot `set_context("subtask", …)` with subtask generation off (direct-subtask mode); no subclass.

## Subtask vocabulary (OneJev's candidate pool)
- Extracted by `executors/robocasa-pi052/extract_subtask_vocab.py` → `data/vocab/pi052_robocasa_subtasks.json` (9.7 MB): **21,326 distinct subtasks over 32,043 episodes**.
- Long tail: 212 in ≥100 episodes, 1,133 in ≥20, 7,415 in 2–19, 12,778 only once.
- Paraphrases coexist ("open the refrigerator door" 2,980 / "open the fridge door" 596); 1,237 fridge-related variants.
- Top: open the refrigerator door (2,980) · open the cabinet door (2,726) · pick up the bowl (2,359) · close the refrigerator door (1,430) · open/close the microwave door · turn on the faucet.
- Dataset files used for extraction were deleted from the HF cache after extraction (3.11 GB freed); `extract_subtask_vocab.py` re-downloads them if needed.

## Driver (`executors/robocasa-pi052/drive.py`)
- Built from LeRobot's own language runtime (`lerobot.runtime`, behind `lerobot-rollout --sim`): same env factory, policy/processor loading, PI052 adapter.
- Added: per-step env success (`RecordingBackend`), subtask-change log, `--planner own|direct`. OneJev later plugs into the `direct` hook: `runtime.state.set_context("subtask", …)`.
- `ctrl_hz=0`: one action per env step. With `ctrl_hz>0` the runtime pops `round(lag·hz)` queued actions and runs only the last (real-robot catch-up) → skips actions in sim.
- Synchronous: language head runs only between action chunks (timing T1).
- Output: `run_config.json`, `episode.json` (success, success_step, subtasks, setup/load/run times, steps/s, GPU peak), annotated mp4.

## Kaggle (2×T4) runner (`executors/robocasa-pi052/kaggle/`)
- `push.py push`: tars `pyproject.toml`, `uv.lock`, `drive.py` + `provenance.json` (commit pins, sha256, repo git state) into `kaggle_run.py`; pushes `vikask2006/rojev-pi052` (private, GPU, internet, 4 h limit). Refuses if `vendor/*` ≠ `config/*.commit`.
- `kaggle_run.py` stages (timed → `outputs/setup.json`, output streamed live): tokenizer → gpu → bundle → vendor_clone (pinned SHAs) → `uv sync --frozen` (torch 2.11+cu128) → assets (tex, generative tex, lightwheel fixtures, objaverse + lightwheel objects) → render probe (EGL, else OSMesa) → bf16 probe → model download → `drive.py` runs.
- `push.py status` / `push.py pull` → `results/kaggle/pi052_<time>/`.
- Gated PaliGemma files (tokenizer.json, tokenizer_config.json, config.json, special_tokens_map.json, added_tokens.json @35e4f46) come from the private dataset `vikask2006/rojev-paligemma-tokenizer` and are passed as a local dir: `drive.py --tokenizer-dir` → preprocessor overrides (`pi052_text_tokenizer.tokenizer_name`, `action_tokenizer_processor.paligemma_tokenizer_name`) + `config.tokenizer_name` (PI052 runtime). Public repos download normally. Reason: CLI-pushed runs cannot read Kaggle secrets (HTTP 400, v1–v2).
- Global `HF_HUB_OFFLINE=1` does not work: the FAST tokenizer loads `bpe_tokenizer/config.json`, which does not exist on the Hub; online the 404 is tolerated, offline it is fatal (v3–v4).
- Precision: checkpoint dtype bf16 (config allows bf16/fp32 only). T4 has no native bf16 → bf16 probe measures whether it runs; fallback = fp32 across both GPUs.

## Next
1. First Kaggle run = setup check only (one 50-step chunk via the `direct` hook). No standalone π0.5² evaluation; it is measured in the supervisor workflow (decision 2026-10-17).

- FAST action tokenizer (`lerobot/fast-action-tokenizer`): training-only auxiliary loss; at inference the preprocessor step is built (needs the files) but skips (`action is None`). Actions come from the flow-matching action expert.
