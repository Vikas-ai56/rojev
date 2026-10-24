# Bring-up checklist (in order)

Each gate: what to check · how · what broke when it was skipped (2026-10-22 unless noted). Record each gate's result
in the run folder or the bring-up note; a later failure withdraws every result that depended on the gate.

## 0. Claims and licence (before any download)
- Published numbers from primary sources (paper table, model card, leaderboard PR), with protocol: task set, split,
  trials per task, seeds, horizon, env version. Aggregator / news numbers are unverified.
  · Hit: an aggregator table mixed protocols; Axiom-0 had no primary source; XR-1's 57.28% is target50 only.
- Licence of code and weights separately (e.g. Awomo-0.5 weights CC BY-NC 4.0, code Apache-2.0).
- What is actually released: weights, inference, post-training, pre-training, data (XR-1: post-training only).
- Does the benchmark cover our tasks? (Our two RoboCasa tasks were not in target50: no published number applied.)

## 1. Serving the model
- Pin the weights revision (`config/<name>_weights.txt`) and read any `trust_remote_code` files before running them.
- Never expose the vendor's server if it unpickles network bytes (XR-1 `deploy/server.py`): serve behind our wire +
  `Endpoint` (token, body limit, replay cache).
- Own uv env per model; check pins against the core (transformers 4.57.1 vs `qev`'s ≥ 5 → S1-side-only
  `dependency-metadata` override).
- Prebuilt wheels: file-name version = metadata version? (flash-attn 2.8.3: no → `uv sync --frozen --check` never
  passes). Attention backends the model accepts (XR-1: flash-attn only, no SDPA → Ampere+ GPU).
- GPU memory: measured peak, co-residency with other services, growth over a long session (SAM 2.1 sessions kept
  frames on GPU: OOM after 177 frames), errors answered (500), never a dropped connection.
- Latency on the target GPU: preprocess / infer / postprocess, p50 and p95, at the real input size.

## 2. Input contract (bit-level parity with the model's own client)
- Cameras: names, order, resolution, crop, flip, colour order, dtype.
- History: frames per query, interval, clamping at episode start; sent by the robot side (the server sees one frame
  per chunk). Only what the model was trained on: naive extra history hurts (MEM, causal confusion).
- State: layout and order per field, units, quaternion convention, normalization (raw vs normalized), padding.
  · Near miss: `agent_pos` read base-first instead of end-effector-first; caught only by reading the env source.
- Language: exact prompt template, instruction source (env sentence vs subtask), special tokens (`/no_cot`),
  robot-type tag.
- Check: feed recorded frames + states through our path and the reference client path; inputs must match exactly.

## 3. Output / action contract
- Dimension order and meaning, delta vs absolute, units, frame, gripper / switch thresholds, normalization stats.
- Chunk length, steps executed per query (theirs: 16; ours with lockstep refill 0.05: ~15), control rate.
- Mapping into the robot's action space lives on the robot side (`embodiment.to_action_space`).
- Check: action histograms per dimension; `violations` per tick (clamped actions) near 0.

## 4. Environment parity
- Same wrapper, split, object registries, constructor and episode seeds (seed formula), horizon, env version /
  commit, render size.
- Check: same seed → same layout / style ids and pixel-identical first frame vs their env construction.

## 5. Observation integrity in the live process (every run)
- Frames the model receives are real images in the actual process, not only in a standalone script.
  · Hit: robosuite freed stale render contexts in the live GL context on `gc.collect()`; every frame was noise
    (neighbour-pixel difference 70–84 vs ~5) and π0.5² / XR-1 results were invalid until `render_context.py`.
- Check: automatic in `realtime.py` (`harness/frame_check.py`, summary `frame_check`); a run whose frames fail is
  not scored. Render only on the thread that owns the GL context.

## 6. Runtime interaction
- Lockstep first (time stops while the model thinks), then real time.
- Request deadline: robot-clock concept; none in lockstep (hit: wall-clock deadline abandoned queued requests →
  217/292 starved ticks).
- Lead time / refill, commitment policy, blending at splices, starvation, timeouts, errors: all in `summary.json`.
- Real-time feasibility: chunk duration ≥ 2 × end-to-end latency (16 actions = 0.8 s vs ~0.35–0.4 s latency is
  borderline).

## 7. Reproduce, then measure
- Their test cases (tasks, seeds, horizons) in our harness, per-episode comparison against their outcomes
  (XR-1: 18/20 vs 20/20, success times within ~1 s on most episodes).
- Enough episodes for the claim (binomial CI); same protocol when comparing models.
- Controls: a stand-in or scripted policy to validate the harness; an oracle planner to separate planning from
  execution; S1 alone vs S1 + S2 on the T8 scenarios.

## 8. Our own trained models (additional)
- Normalization stats from the training set saved with the checkpoint; the same pre/post-processing at inference.
- Dataset conventions (state, action, frames, rate) equal the env's / robot's; held-out seeds and scenes.
- Train / eval leakage (scenes, objects, seeds); report seen vs unseen separately.
- Ablations that prove a new input is used (shuffled / zeroed latent or history must change behaviour).
- Sim-to-real: inputs available on the real robot only (no sim-only depth or truth), real latency budget.
