# Research notes: decision timing in hierarchical VLA systems, and MJev on the Mac (2026-10-17)

## How existing systems time the high level against the low level
| System | High level | Low level | Sync / async | Switch rule | Staleness handling |
|---|---|---|---|---|---|
| MemER (2510.20328, read in full) | Qwen2.5-VL-7B ~1 Hz, own server | π0.5 ~2 Hz, 15-action chunk, 8 executed | **async** ("improves responsiveness and stability") | low level always uses the latest subtask | none; latency limit < 1 s (API VLMs at 10–15 s → "complete failures") |
| RoboCombine (2610.15425, read in full; code MIT) | learned progress predictor (Florence + GRU + threshold) | VLA chunks | cue monitored during the chunk, **latched** | instruction changes **only before the next chunk query**; discard-cache or pending-switch | debounce: 3 consecutive frames + min dwell 5 steps |
| Goal2Skill (2604.13942, read; repo 404) | API VLM verifier every k chunks | VLA chunks | checkpointed | completed / ongoing / failed → advance / keep / reflect | per-subtask step budgets, retry limits |
| Figure Helix | 7B VLM 7–9 Hz | 80M policy 200 Hz | async, shared-memory latent | latest latent each tick | latency offset in training |
| GR00T N1 / N1.6 | VLM 10 Hz | DiT 120 Hz | coupled / async + RTC | per chunk | RTC |
| π0.5 + Real-Time Chunking (2506.07339) | below action rate | 50 Hz, H=50 | async | freeze the first d actions, inpaint the rest | delay-aware; naive switching → out-of-distribution jerks |
| SmolVLA async (2506.01844) | — | 30 fps, n=50 | async | timestep-merged queues | drop past timesteps; sync 78.3% vs async 73.3% success |
| LeRobot main (unreleased) | language query | RTC | **hold during language decisions** | task_version invalidation | version checks |
| Gemini Robotics ER | step-level | VLA as a tool | **blocking** | waits for the step to finish | cancel on barge-in |
| OneTwoVLA | reasoning | — | **sync** (2–3 s pauses) | — | — |
| HiRT (2410.05273) | cached VLM latent | 9.8 Hz | async | latest latent | refreshing every 6 steps: 94.2 → 63.4 success (staleness cost) |

No paper found that directly compares synchronous vs asynchronous subgoal replanning. That is an open gap.

## Reusable code (licences checked)
- RoboCombine `handoff_eval/hse.py`, `real_switch_runtime.py` (MIT): latch, pending switch, cache clearing, debounced threshold.
- LeRobot (Apache-2.0): `async_inference/`, `policies/rtc/`; on main, `inference/rtc.py` (task_version, hold).
- MemER: **no licence**, read for ideas only. Goal2Skill: no public code.

## MJev-Spatial on the M5 (16 GB, ~9 GB usable)
- Checkpoint = language side only (4.11B params fp32, embed + layers 0–30 + the Q/K readout); vision comes from `Qwen/Qwen3.5-4B` @851bf6e (9.32 GB download). Total ≈ 25.8 GB.
- bf16 ≈ 8.9 GB: does not fit. 8-bit ≈ 5 GB. **4-bit (group 64) language + 8-bit vision ≈ 2.7 GB weights, ~3–3.5 GB peak** (estimates).
- MLX ≥ 0.30 uses the M5 GPU Neural Accelerators (macOS ≥ 26.2). mlx-vlm already has Qwen3.5 (Gated DeltaNet), the vision tower and hidden-state capture. Port estimate 1.5–2 weeks: converter, Q/K readout in fp32, request builder, 3-stage pointing loop.
- Neural Engine (Core ML): not viable (fp16-only, static shapes, no Gated DeltaNet support).
- Latency estimates (unverified, from community M5 prefill numbers): choice ~0.5–1 s; 3-stage point ~3–6 s; the fanless Air may throttle.
- A quantized port must be validated against the fp32 reference (hidden-state cosine ≥ 0.999, top-1 agreement ≥ 98–99%). The fp32 reference cannot fit in Mac RAM, so it needs a GPU machine or slow layer-streamed runs.
