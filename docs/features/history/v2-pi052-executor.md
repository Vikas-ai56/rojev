# v2 executor: π0.5² set-up and Kaggle runner (historical)

Historical (v2 design, superseded 2026-10-21 by the v3 runtime with π0.5² as a remote service, then 2026-10-22 by XR-1 as the RoboCasa S1; see [`../remote-models.md`](../remote-models.md), `scripts/gpu_setup.sh`).

## What it was (2026-10-17)
- π0.5² = `lerobot/pi052_robocasa` (LeRobot branch `feat/smolvla-on-steerable`, `config/lerobot-steerable.commit`), RoboCasa PandaOmron, sim only, Apache-2.0; 4.14B params (2.82B bf16 + 1.32B fp32); model card reports no success rates.
- Steering point: low-level actions conditioned only on `"{subtask}, State: …"`; its own planner (`_generate_low_level_subtask`) regenerates the subtask every chunk; an empty slot = own planner (arm A), a set slot = direct mode (arm D). The same slot is how `serve.py` steers it today (subtask given, or its planner run once).
- Subtask vocabulary (E8): `extract_subtask_vocab.py` → `data/vocab/pi052_robocasa_subtasks.json`, 21,326 distinct subtasks over 32,043 episodes; long tail (12,778 seen once) and coexisting paraphrases ("refrigerator" 2,980 vs "fridge" 596). Fed the v2 candidate filter.
- `drive.py`: π0.5² on LeRobot's own language runtime (`--planner own|direct`), still in the executor.
- Kaggle 2×T4 runner (`executors/robocasa-pi052/kaggle/`): bundled code + pins, ran setup stages and episodes; setup ≈ 11.6 min per session (v5, `results/kaggle/pi052_20261017-192153/`).

## What replaced it
- Synchronous per-chunk driving → `serve.py` behind `rojev.transport`, driven by the v3 runtime (`realtime.py`).
- Kaggle sessions → one A100 box set up by `scripts/gpu_setup.sh`.
- π0.5² as RoboCasa S1 → XR-1 (decision 2026-10-22; experiments E23).

## Lessons that still apply
- LeRobot's runtime with `ctrl_hz > 0` pops `round(lag·hz)` queued actions and runs only the last (real-robot catch-up): skipped actions in sim; use one action per env step.
- Gated files (PaliGemma tokenizer): download with the user's own login and ship only the files; never send the token to a remote machine (kept in `gpu_setup.sh`).
- Global `HF_HUB_OFFLINE=1` breaks the FAST tokenizer: it probes `bpe_tokenizer/config.json`, absent on the Hub; online the 404 is tolerated, offline it is fatal.
- T4 has no native bf16: probe the checkpoint dtype before trusting it (bf16 probe rel. err 0.29%).
- Read remote datasets once, not by column: π0.5²'s parquet files (939 row groups) re-downloaded per read.
