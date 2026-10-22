# Engineering notes (moved from docs/paper/appendix-engineering.md, 2026-10-17)

**Status:** v2-era build notes (2026-10-17), kept for history: the licence and reuse rules still apply to code still in use; the arm and selector rows (OneJev hook, arm C, G3 memory) are v2 and superseded by v3 ([`v3-runtime.md`](v3-runtime.md)).

Not paper content; kept for the build. Paper-relevant pitfalls stay in `docs/paper/appendix-engineering.md`.

## 1. Code reuse decisions and licences
How: **import** = use vendored code unchanged · **port** = copy with licence header + attribution, adapt the interface · **ideas** = reimplement from the paper (no licence / no code) · **no** = not used.

| Source | Licence | What we take | How | Where it lands |
|---|---|---|---|---|
| π0.5² (`pi052`, LeRobot `feat/smolvla-on-steerable` @9d00293) | Apache-2.0 | Low-level action path + subtask prompt `"{subtask}, State: …"` | import | executor env |
| π0.5² | Apache-2.0 | Its own high-level planner (`_generate_low_level_subtask`) = **Arm A baseline** | import | executor env |
| π0.5² | Apache-2.0 | OneJev hook: subclass overrides `_generate_low_level_subtask` | import (subclass, no fork) | executor adapter |
| π0.5² training data (`language_persistent`, style `subtask`) | dataset licence | **Subtask vocabulary = OneJev's candidate set** (the strings the low level was trained on) | import (read) | candidate builder |
| π0.5² recipe `subtask_mem.yaml` | Apache-2.0 | Memory stream (`Previous memory` / `Completed subtask`) as the memory baseline for G3 | import | G3 |
| LeRobot `steerable_external_vlm_main` `rollout/planner.py` (319 lines) | Apache-2.0 | External generative-VLM planner (Steerable Policies prompt, closed vocabulary, JSON reply) = **Arm C**, run with Qwen3.5-0.8B (OneJev's parent) | port | `rojev/` supervisor arms |
| RoboCombine `handoff_eval/{hse,real_switch_runtime}.py` | MIT | Latch, pending switch, cache clear, debounce (3 frames, dwell 5); HSE modes → arms (full-inst / predicted / oracle); ordered + transition metrics | port | `playground/timing/`, evaluator |
| LeRobot `policies/rtc/`, `async_inference/` | Apache-2.0 | Async chunk transitions for timing T3 | import | playground T3 |
| MJev-Spatial | Apache-2.0 | Model runtime → MLX 4-bit port | port | MJev service |
| MemER | **none found** | Keyframe memory (1D single-linkage), boundary-accuracy metric | ideas | G3, evaluator |
| Goal2Skill | no public code | completed/ongoing/failed verifier states, per-subtask step budgets, retry limits | ideas | monitor |
| SkillNet | — | Not used: it changes the executor (skill-MoE inside the VLA), a separate axis | no | — |
| EmbodiedJev | MIT | Not used: text-only Jev workbench; related work | no | — |
| VLASH, RTC-kinetix | Apache / MIT | Not now (async refinements beyond LeRobot RTC) | no | — |

Rules: every port keeps its original licence header and is recorded here with its source commit; ported files record the source commit.

## 2. MJev-Spatial on Apple M5 (16 GB, ~9 GB usable)
- Checkpoint = language side only (4.11B params fp32, embed + layers 0–30 + the Q/K readout); vision comes from `Qwen/Qwen3.5-4B` @851bf6e (9.32 GB download). Total ≈ 25.8 GB.
- bf16 ≈ 8.9 GB: does not fit. 8-bit ≈ 5 GB. **4-bit (group 64) language + 8-bit vision ≈ 2.7 GB weights, ~3–3.5 GB peak** (estimates).
- MLX ≥ 0.30 uses the M5 GPU Neural Accelerators (macOS ≥ 26.2). mlx-vlm already has Qwen3.5 (Gated DeltaNet), the vision tower and hidden-state capture. Port estimate 1.5–2 weeks: converter, Q/K readout in fp32, request builder, 3-stage pointing loop.
- Neural Engine (Core ML): not viable (fp16-only, static shapes, no Gated DeltaNet support).
- Latency estimates (unverified, from community M5 prefill numbers): choice ~0.5–1 s; 3-stage point ~3–6 s; the fanless Air may throttle.
- A quantized port must be validated against the fp32 reference (hidden-state cosine ≥ 0.999, top-1 agreement ≥ 98–99%). The fp32 reference cannot fit in Mac RAM, so it needs a GPU machine or slow layer-streamed runs.

## 3. Simulator/platform notes
- macOS: LeRobot async (multi-process) RoboCasa envs crash at reset (GL context in worker processes) → run envs sequentially on the Mac.
- macOS: MuJoCo depth lacks `ARB_clip_control` (no reverse-Z); measured error still ≤ 3 mm at ~0.8 m.
- RoboCasa target split needs the objaverse object pack; lightwheel-only crashes some tasks at reset (E7).
- Remote parquet column reads of the π0.5² dataset (939 row groups per file) re-download files repeatedly; download once instead (100 s for 3.11 GB in parallel).
- LeRobot language runtime (`lerobot-rollout --sim`, `ctrl_hz>0`): under inference lag it pops several queued actions and executes only the last (real-robot catch-up) → skipped actions in sim. Our driver uses `ctrl_hz=0` (one action per env step).
