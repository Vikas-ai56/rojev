# Search directions when a model underperforms

Lists cannot cover everything; these are the places and methods that find the next unknown failure.

## Sources, in order of trust
1. The model's own evaluation client and server code: diff its input building, history, crop, state, prompt and
   action decoding line by line against ours (XR-1 `eval_robocasa365/entry.py`).
2. Checkpoint files: `config.json`, processor / preprocessor configs, normalization stats, `trust_remote_code`
   modules (input shapes, dtypes, required kwargs, attention backends).
3. Training recipe and dataset docs: definitions of state, action, frames, rate; what the instruction text was.
4. Benchmark: env source at the pinned commit, leaderboard PRs (exact protocol and counts), task-set registries.
5. Upstream issues / discussions (GitHub, Hugging Face): "reproduce", "low success", "eval mismatch", version pins.
6. Papers for known failure modes: train / inference shift, causal confusion from history, async chunking (RTC),
   resolution / crop shift, memory (MEM).

## Instruments (cheap, run early)
- Look: a frame strip and the video of the first episode.
- Integrity: neighbour-pixel difference per camera per frame; blank / noise detection.
- Parity dumps: model inputs (tokens, pixel tensors, state) from our path vs the reference path on identical frames.
- Distributions: actions per dimension, clamped-action counts, gripper / mode switch timing.
- Runtime: requests per tick, timeouts, starved ticks, latency p50 / p95, committed steps per chunk.
- Per-episode comparison with published outcomes (same seed): success and time to success.

## Bisection
- Plain script with the same env and actions vs the harness; add harness pieces one at a time (loop, threads, GC,
  tick hooks, policy, network) until the symptom appears (this found the render-context bug).
- Replay recorded actions in a clean process: separates physics / actions from process state.
- Swap one component for a known-good one: stand-in policy, their client, lockstep vs real time, blend 0.
- Change one variable per run; keep seeds fixed; check the stand-in first.

## Platform specifics to suspect
- GL backend and context ownership (CGL / GLFW on macOS, EGL / OSMesa on Linux; render on the owning thread).
- Garbage collection and object lifetimes in third-party wrappers (`__del__` freeing shared resources).
- Script names shadowing stdlib modules (`profile.py`, `bisect.py` broke imports).
- GPU architecture and kernels (flash-attn needs Ampere+), memory sharing between services, wall vs robot clocks.
