# v2 engineering notes: code reuse, MJev on Apple silicon, platform (historical)

Historical (v2 design, superseded 2026-10-21 by the v3 runtime and v1 = Route A; see [`../v3-runtime.md`](../v3-runtime.md), [`../../model-bringup/checklist.md`](../../model-bringup/checklist.md)).

## What it was (2026-10-17)
- Reuse plan for the v2 arms (decision 2026-10-17): import π0.5² unchanged (actions, its planner as arm A, its subtask vocabulary as OneJev's candidates); port LeRobot's external-VLM planner (arm C) and RoboCombine's switching and metrics; MemER and Goal2Skill as ideas only (no usable code or licence); SkillNet, EmbodiedJev, VLASH not used.
- MJev-Spatial MLX port for the 16 GB M5 (4-bit language + 8-bit vision ≈ 2.7 GB weights, estimated): not built; MJev ran on the GPU box instead and left the v1 loop on 2026-10-22.

## What replaced it
- The arms went with v2: arm C and the RoboCombine ports never landed (their ideas sit in `playground/timing/README.md`).
- The one port in use today is XR-1's input code (`executors/robocasa-xr1/xr1_inputs.py`, Apache-2.0, header names its source).
- Licence checks for new models: [`../../model-bringup/checklist.md`](../../model-bringup/checklist.md) §0.

## Lessons that still apply
- A port keeps its original licence header and names its source file and commit; code without a licence is reimplemented from the paper, never copied.
- macOS: LeRobot's multi-process RoboCasa envs crash at reset (GL context in worker processes); run envs in one process.
- RoboCasa's target split needs the objaverse object pack; lightwheel-only crashes some tasks at reset (E7).
- A quantized port is validated against the full-precision reference (hidden-state cosine, top-1 agreement) on a machine that fits the reference.
