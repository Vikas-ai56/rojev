# Appendix: engineering for reliability and reproducibility

What makes the loop correct in real time, honest to evaluate, portable to another robot and reproducible on a fresh GPU
box. Each item: **problem** → what we do · *evidence* · `source`. Paths are relative to the repo root; dated entries
are in `docs/decisions.md`. Code as of 2026-10-24.

## A. S1 executor contracts

### A.1 Xiaomi-Robotics-1 (RoboCasa365 checkpoint), as our adapter sends it
Pinned `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 @ 3a6d029…` (`config/xr1_weights.txt`). Inputs are ported from the
model's own evaluation client so a query matches its published evaluation.

| Part | Exactly what is sent | Where |
|---|---|---|
| Cameras | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`, labelled "Left / Right / Wrist camera:", each as one video | `executors/robocasa-xr1/xr1_inputs.py:17,58` |
| History | 4 frames per camera, 2 control ticks apart (0.1 s at 20 Hz), ending at the queried observation, clamped at the episode start (first query = one frame 4×); kept on the robot side, sent as images `<cam>@k` (k = 0 oldest) | `executors/robocasa-pi052/history.py:31,47` |
| Crop | centre crop 0.95, bilinear resize back; processor `do_resize=False` | `xr1_inputs.py:47`, `serve.py:46` |
| State | per frame, 14-D end effector first: ee position, ee axis-angle, gripper (2), base position, base axis-angle (xyzw quaternions → axis-angle, w ≥ 0), zero-padded to 60 → (1, 4, 60) | `xr1_inputs.py:38,69` |
| Prompt | `…Generate robot actions for the task:\n{instruction} /no_cot`, assistant turn `<cot></cot>`; instruction = subtask if given, else the task; `robot_type` / `task_id` = `robocasa365` | `xr1_inputs.py:58`, `serve.py:46-51` |
| Model | `AutoModel` with the checkpoint's own code, `flash_attention_2`, bf16. No SDPA path → flash-attn and an Ampere+ GPU are required | `executors/robocasa-xr1/serve.py:35` |
| Output | `decode_action` → 16 steps × first 12 values (0.8 s chunk), returned raw. The robot side applies `to_action_space`: gripper and base mode snapped at 0.5 to ±1, as RoboCasa's gym wrapper does; continuous dimensions untouched | `serve.py:56`, `executors/robocasa-pi052/embodiment.py:45`, `realtime.py:305` |

- Stateless server (history comes with the request): a retried request answers the same. *Evidence:* `executors/robocasa-xr1/tests/test_xr1_inputs.py` (state reorder, frame order + padding),
  `executors/robocasa-pi052/tests/test_history.py`; A100: 10.2 GB, ~40 ms preprocess + 275–290 ms inference; their
  target50 cases (4 tasks × 5 seeds, lockstep) 18/20 vs their 20/20 (`docs/features/system1-map.md` §12).
- Lesson: `agent_pos` has two layouts across LeRobot branches (`vendor/lerobot/.../envs/robocasa.py:36` base first;
  `vendor/lerobot-steerable/.../envs/robocasa.py:36` end effector first). Read the env source of the branch you run.

### A.2 π0.5² (`lerobot/pi052_robocasa`) [pi052_robocasa]
- Steering point: the subtask slot, low-level prompt `"{subtask}, State: …"`; empty subtask → its own planner, whose
  choice comes back in `info` (planner memory off by default). `executors/robocasa-pi052/serve.py`.
- K samples per observation in one forward batch (`Draws`); switch dims mapped by `to_action_space` server-side
  (before it, 486 of 600 ticks reported violations from outputs up to 1.018).
- RTC is opt-in (`serve.py --rtc H`): the request's continuation → checkpoint normalizer → LeRobot RTC guidance
  [black2025realtimechunking], under `no_grad`. Off by default; earlier delay / RTC A/B withdrawn (corrupted frames).

### A.3 SmolVLA eval pitfalls (historical, 2026-10-17) [shukor2025smolvla]
| Mismatch | Trained with | LeRobot eval / docs sent | Fix (`executors/robocasa/`) |
|---|---|---|---|
| Camera mapping | right → camera2, wrist → camera3 | swapped | read from the checkpoint's rename map |
| Scene split | `target` | `all` | `--env.split=target` |
| Instruction | task names (`CloseFridge`) | sentences | send the task name (A/B 1/5 vs 0/5, same seeds) |

## B. Real-time correctness (`rojev/control`, `rojev/system1`, `rojev/owner`)
- **Stale motion after slow inference.** Actions are placed at observation time (`t_capture + k·dt`); actions already
  due when the chunk lands are skipped, never replayed. *Evidence:* `tests/test_control_execution.py::
  test_elapsed_actions_are_skipped_never_replayed`, `tests/test_system1_runner.py::
  test_actions_due_during_inference_are_skipped_and_latency_is_measured`. `rojev/control/execution.py:220,299`.
- **Gaps between chunks.** Latency L = max of the last 20 observation ages; request when less than
  `max(refill_s, L + dt)` is queued; commit at least ⌈2L/dt⌉ + 1 actions (whole chunk before the first measurement);
  `floor_over_chunk` and `starved_ticks` counted, not inferred. Deadline = one chunk duration (an older answer is all
  in the past); none in lockstep. *Evidence:* `test_after_the_first_request_the_lead_time_covers_the_latency`; XR-1
  real time: ~0.5 s wall per request vs a 0.8 s chunk → floor exceeds the chunk, supply-bound. `rojev/system1/runner.py:129-158`.
- **Jumps at a chunk splice.** Each new segment keeps 1 − s(k/ticks) of the jump in continuous dims (quintic
  smoothstep, `blend_s` 0.2 s); output projected onto bounds; the runtime's own blend never counts as a violation.
  *Evidence:* `test_new_segment_is_blended_in_continuous_dims_only`, `test_blending_stays_inside_the_bounds_and_is_not_a_violation`.
- **Who is active.** A command activates only when its first action is applied (`activated` ACK); a lease covers its
  whole life; `review` keeps it moving with switches frozen and the lease shortened to the interaction deadline;
  `revoke` drops its queue at once; stale epoch / revoked / expired / wrong-`dt` chunks are refused with an ACK.
  *Evidence:* `tests/test_control_execution.py` (lease, review, revoke, `test_chunk_at_another_rate_is_rejected`).
- **Irreversible actions.** Switches marked irreversible in the `ActionSpec` (gripper) change only while running an
  active, unreviewed chunk; otherwise they keep their last value. *Evidence:*
  `test_irreversible_switch_only_changes_while_running_a_chunk`; run invariant `switches_held`.
- **Dead or wrong upper layers.** `Guard` in the control loop: no valid action for 0.5 s → hold, command kept
  ("starved"); owner heartbeat older than 2 s → hold, command dropped; runtime exception → hold, dropped.
  *Evidence:* three `test_guard_*` tests + `test_an_owner_lost_hold_drops_the_running_command_and_a_new_one_moves_again`.
  `rojev/control/guard.py`.
- **Conflicting intent.** The state owner is the single writer of intent: one thread, one event queue, never waits
  for a model; what is active comes only from runtime ACKs (newest `seq`). The interaction lane outranks planning; a
  change nobody answered holds the robot and never reaches planning. `rojev/owner/state_owner.py` docstring;
  `test_an_unanswered_change_holds_the_robot_and_never_reaches_planning`.
- **Late interaction answers.** OneJev takes 1.1–1.5 s against a 1.0 s deadline: the owner holds (`UNANSWERED`), then
  applies the late answer to the situation it was decided on (CONTINUE resumes what was in force), never re-decides.
  *Evidence:* `test_a_late_answer_is_applied_to_the_situation_it_was_decided_on`; not yet exercised live (campaign
  F13 answered in 0.85–0.90 s). Decisions 2026-10-24.
- **Policies that can use the prefix.** `Continuation(queued_from(cmd, t_capture), ⌈L/dt⌉)` goes only to a policy
  declaring `accepts_continuation` (wire v3); the policy decides how to use it (8.7 µs on the runner thread).
  XR-1 does not accept one. *Evidence:* `test_a_continuing_policy_gets_the_queued_actions_from_its_observation_and_the_latency_in_steps`.
- **GC pauses.** `gc.collect(); gc.freeze()` before the threads start (a full pass over the setup heap measured
  45–96 ms vs a 50 ms period). `executors/robocasa-pi052/realtime.py:561`.

## C. Evaluation integrity (`executors/robocasa-pi052`)
- **Model time vs network time.** `--lockstep`: the loop does not tick while S1 or either S2 lane is busy, so every
  model's computation takes zero simulated time. Used for model comparisons in sim (published benchmarks are scored
  this way; our S1 is remote, 0.35–0.55 s vs a 0.8 s chunk); real time is reported alongside, and hardware needs
  real-time passes with 0 starved ticks. *Evidence:* pausing for S2 too turned late answers (OneJev 1.1–1.5 s →
  1.75–1.85 s simulated) into wait 3.05 s, resume 6.05 s, success 10.5 s. Decisions 2026-10-22 / -23; `docs/hardware-port.md`.
- **Silently corrupted frames.** robosuite freed a replaced simulator's render context in the *current* GL context,
  deleting the live one's framebuffer after any `gc.collect()`: every frame was noise (neighbour-pixel difference
  70–84 vs ~5). `render_context.install()` frees each context in its own GL context. Every earlier live result was
  withdrawn. *Evidence:* `executors/robocasa-pi052/tests/test_render_context.py`. GLFW/CGL only. `render_context.py`.
- **Frame check on every run.** A thread samples the newest capture every 1 s: std < 3 → blank, mean horizontal
  neighbour difference > 25 → noise (963 real frames: ≤ 7.4 and std ≥ 26.8). A run with any bad frame is not scored.
  Frozen frames are not detected. `harness/frame_check.py`, `tests/test_frame_check.py`.
- **Run invariants** over `ticks.jsonl` / `owner.jsonl` / `summary.json`: `no_stale_action`, `switches_held`,
  `active_monotonic`, `changes_answered`, `no_violations`, `frames_real`. `harness/invariants.py`,
  `tests/test_harness.py::test_invariants_catch_what_they_promise`.
- **Judging interaction by state.** Per user line, the robot's state is read from owner events logged *before* the
  line (moving · held · stopped · idle · done); the first decider answer is checked against `allowed(intent, state)`;
  a wrong answer that moves the robot is `unsafe`, one that holds, asks or stops is a `safe_miss`.
  `harness/judge.py`, `tests/test_campaign.py` (5). Campaign 2026-10-24: 82 lines, 55 pass, 2 unsafe, 24 safe misses.
- **Stochastic S1.** XR-1's sampling diverges between runs of the same task and seed from the first chunk (actions
  differ at t = 0.1 s), so success per case needs repeats; verdicts per line are unaffected.
  `results/campaign/20261024-134629/analysis.md`.
- Bring-up gates for any new model (input parity, env parity, frame integrity, lockstep then real time):
  `docs/model-bringup/checklist.md`.

## D. Transport and serving (`rojev/transport`, `rojev/system1/{wire,service}.py`)
- **Unsafe deserialisation.** Our wire is compressed `.npz` with `allow_pickle=False`, versioned (v3). XR-1's own
  server unpickles network bytes and is never exposed. Frames WebP-90 by default (~26 KB vs ~150 KB raw; actions
  within π0.5²'s sampling noise, 0.078 vs 0.070).
- **Retries that run twice.** Each request carries a 128-bit `X-Request-Id`; the client retries once with the same
  id; `Endpoint`'s `ReplayCache` (last 256) runs each id at most once, waits on an in-flight copy, 409 on reuse for a
  different body. Token (`hmac.compare_digest`) and declared length (32 MB) checked before any body byte is buffered.
  *Evidence:* `tests/test_transport.py` (lossy TCP proxy, overlap, admission). `rojev/transport/endpoint.py`.
- **Connection setup over a WAN.** One kept connection per client. π0.5², India ↔ us-east: 1,012 → 572 ms round
  trip. OneJev (`KeptClient`, `rojev/system2/models/onejev.py:31`): median 1269 → 604 ms per decision, identical
  probabilities.

| OneJev decision, Mac → A100 over the tunnel (3 cameras, JPEG q95) | ms |
|---|---|
| server: render + one forward pass | 75–140 |
| one round trip on a kept connection | ~270–345 |
| image upload (3 images vs none) | ~215 |
| extra round trip of a new connection per request (removed) | ~650 |

- **Exposure.** Every service binds 127.0.0.1 on the box and is reached only through SSH tunnels (qev has no auth,
  so `--host 127.0.0.1` is mandatory). Tokens: made on the Mac, mode 600 on both sides, sent only inside the
  SSH-streamed script, compared by hash, never printed or on a command line. No Hugging Face token leaves the Mac.
  `docs/features/remote-models.md`, `docs/features/gpu-setup.md`.

## E. GPU setup reproducibility (`scripts/gpu_setup.sh`, `scripts/gpu_box.sh`)
- One command turns a fresh Ubuntu GPU box into running services + tunnels + a service JSON; idempotent, `--dry-run`
  changes nothing.
- Components plug in through `<kind>/<name>/gpu/hook.sh` (key, auth, GPU memory, files, vendors, port, prepare, serve):
  `executors/robocasa-{pi052,xr1}`, `services/{onejev,mjev,sam2}`.
- Pins: weights `config/*_weights.txt` (`repo @ revision`), vendor clones `config/*.commit`, one `uv.lock` per
  component (`uv sync --frozen`, uv-managed Python).
- flash-attn 2.8.3 for XR-1 from its sdist with the CUDA build skipped, so its setup fetches the matching prebuilt
  wheel under its own version (`uv sync --frozen --check` then passes); readiness imports `flash_attn_2_cuda` to
  catch a kernel-less build. `executors/robocasa-xr1/pyproject.toml`, `gpu/hook.sh`.
- Env, assets and remote preparation run for all components at once (2026-10-24); services start one by one, each
  refused if free GPU memory < its need, a foreign process on its port is never touched, and each waits for
  `GET /health` (again through its tunnel).
- *Evidence:* first fresh-box run 2026-10-22 found a hook missing `embodiment.py` (fixed, re-run idempotent);
  pitfalls with cause → fix in `docs/features/gpu-setup.md`.

## F. Embodiment plug-in (`rojev/robot`, `rojev/control`)
- `ActionSpec`: per dimension name, bounds, hold value (number or `KEEP` = repeat the last), switch `threshold`,
  `irreversible` (must be a switch); `control_hz`. `limit` clamps and reports every clamp. `rojev/robot/action_spec.py`.
- `EffectModel` (execution monitor): `tracked(state)`, `command(state, action)`, `resolution`; the monitor fits the
  command → motion map, so a robot states only which numbers correspond. `rojev/robot/effect.py`.
- Backends: RoboCasa PandaOmron (`executors/robocasa-pi052/embodiment.py`: 12 dims, gripper irreversible at 0.5,
  base mode a switch, 20 Hz) and any `rojev.robot.Robot` via `RobotBackend` + `joint_target_spec` (MuJoCo SO-101
  now; `SafetyFilter` clamps reported). *Evidence:* `test_robocasa_panda_omron_plugs_in`,
  `test_any_rojev_robot_plugs_in_so101_lockstep`, `test_so101_realtime_keeps_its_clock_and_lag_budget`,
  16 contract tests in `tests/test_robot_contract.py`.
- A new robot supplies: a backend (`observe` / `step`, or a `Robot` with clock and limits), its `ActionSpec`, an
  `EffectModel` if monitored, and the map from each policy's outputs into its action space (never truncate another
  robot's action to fit). Model needs (history, crop, prompt) stay in that policy's executor. On hardware: camera
  staleness, a model-independent stop path, one clock for cameras and joints (`docs/hardware-port.md`).

## G. Tests re-run 2026-10-24 (Mac, no GPU)
| Suite | Result |
|---|---|
| core `uv run pytest -q` | 329 passed, 1 skipped, 2 timing deselected |
| `executors/robocasa-pi052` | 36 passed |
| `executors/robocasa-xr1` | 2 passed |
