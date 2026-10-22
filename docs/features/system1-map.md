# System 1 map: how actions reach the robot (code as of 2026-10-22)

How System 1 keeps the robot moving: control loop → guard → execution runtime ← S1 runner ← policy (π0.5² over the
network). System 2 never sends actions; its only input to this side is a command from the owner (see
[`system2-map.md`](system2-map.md)). Details per feature: `s1-supply-and-commitment.md`, `execution-monitor.md`,
`remote-models.md`, `v3-runtime.md`.

## 0. Call map
```
realtime.main → executors/robocasa-pi052/realtime.py          builds and wires every part, one thread each
 ├ ControlLoop      → rojev/control/loop.py                    one tick per control period (20 Hz RoboCasa)
 │   └ Guard        → rojev/control/guard.py                   watchdog, owner heartbeat, stop
 │       └ ExecutionRuntime → rojev/control/execution.py       what is physically active; one action per tick
 ├ ExecutionMonitor → rojev/system1/monitor.py                 per tick (in the loop thread): expected vs measured motion
 └ S1Runner         → rojev/system1/runner.py                  keeps the active command supplied with chunks
     ├ CommitPolicy → rojev/system1/commit.py                  how much of a chunk to commit
     └ RemotePolicy → rojev/system1/remote.py → wire.py         π0.5² request / response (v2, K samples)
           └ transport → rojev/transport/{http,h3,endpoint}.py  kept connection, replay-safe retry
                 └ serve.py (GPU box) → Pi052Policy             π0.5² inference
```
Dependencies point down: protocol, robot ← control ← system1 (+ transport). Nothing in System 1 imports System 2 or the
owner. Robot specifics enter only as an `ActionSpec` and an `EffectModel` (`executors/robocasa-pi052/embodiment.py`).

## 1. Entrypoint (`realtime.py`)
- `RoboCasaBackend` = the `Backend` (`observe`, `step`); the loop steps it at k·dt, paced to the wall clock.
- `ExecutionRuntime(PANDA_OMRON, epoch=1, last=START, blend_s, stop_s)`, `Guard(runtime)`.
- `ExecutionMonitor(PandaOmronEffect(), …, on_deviation)` runs inside the loop's `on_tick`.
- `S1Runner(runtime, RecordingPolicy(make_policy(a)), loop.latest, loop.now, refill_s, commit)` (`realtime.py:355`).
- Flags: `--policy remote|skills|wiggle`, `--commit-policy fixed|phase`, `--speculative-predict --samples K`,
  `--execution-monitor M.json`, `--refill`, `--blend`, `--stop-s`, `--lease`.
- Outputs: `ticks.jsonl` (one line per tick), `s1_requests.jsonl`, `commits.jsonl`, summary S1 stats.
- `gc.collect(); gc.freeze()` before the threads start (a full GC pass measured 45–96 ms).

## 2. Control loop (`control/loop.py`)
- `_run` (`:85`): with a robot clock → `Rate`; else tick k due at `t0 + k·dt` on the wall clock; lateness recorded
  (`lag_s`, `max_lag_s`), never skipped.
- `_tick` (`:107`): `guard.step(t)` → `backend.step(action)` → new `Capture(obs_id+1, t, obs)` into the latest-value slot →
  `on_tick(TickRecord(result, before, after, step_ms, lag_s))`.
- `latest()` (`:62`): the newest capture. S1 and S2 read only this, never a backlog.

## 3. Guard (`control/guard.py`)
- `step` (`:29`):
  - owner heartbeat older than `owner_timeout_s` (2 s) → `hold("owner lost", drop_commands=True)`;
  - RUNNING without a valid action for `watchdog_s` (0.5 s) → `hold("starved", drop_commands=False)`;
  - runtime exception → `hold("runtime error…", drop_commands=True)`.
- Dropping a command sends a `dropped` ACK, so the owner re-decides; a starved command resumes with its next chunk.

## 4. Execution runtime (`control/execution.py`); the authority on what is active
- `accept(goal)` (`:123`): refuses stale epoch / revoked / not newer / expired. A non-moving operation (ABORT, WAIT,
  ASK) becomes active at once as a hold. A moving one becomes `pending`; `pause_hold` (or not running) holds until its
  first chunk.
- `renew` (`:150`), `review` (`:160`), `revoke` (`:174`): lease extension; review (keeps moving, switches frozen, lease
  shortened); revoke (drops its queued actions at once).
- `submit(chunk)` (`:220`): refuses wrong epoch, unknown, revoked or expired command, wrong `dt`, wrong shape, a
  non-moving command, paused, held. Actions already due are skipped. A chunk for the pending command splices in at its
  first due time; one for the active command replaces its own future queue.
- `target()` (`:267`): the command that needs chunks: pending, else the active one while moving (or starved).
- `tick(now)` (`:287`), in order:
  1. lease expiry;
  2. skip past-due actions;
  3. the due action: the first one of the pending command activates it (`activated` ACK);
  4. bounds (clamps reported as `violations`);
  5. inertialized blend of a new segment;
  6. otherwise the hold, with a decelerating ramp;
  7. irreversible switches frozen outside a running, unreviewed chunk;
  8. projection onto the bounds.
- `last_action()` (`:277`): copy of the last applied action (the commitment policy's switch state in effect).
- ACKs are an outbox the owner drains (`drain_acks`, `:382`); `seq` strictly increases.

## 5. S1 runner (`system1/runner.py`); supply
- `step` (`:123`), on its own thread every `poll_s`:
  1. `goal = runtime.target()`; none → nothing to do (holding, or no command).
  2. Request only if a refill is pending for this command, or less than `lead_s()` is queued (`queued_until`).
  3. `capture = latest()`; observation task = `goal.instruction`; subtask = `goal.operation.subtask` ("" = π0.5²
     plans its own).
  4. `policy.act` (or `act_samples(K)` for consensus): one request, one upload.
  5. `floor(horizon)` (`:104`) = ⌈2L/dt⌉ + 1 (whole chunk before the first measurement); `floor_over_chunk` counted.
  6. `commit.plan(candidates, spec, floor, runtime.last_action())` → the committed prefix.
  7. `TimedChunk(epoch, cmd_id, obs_id, t_start = capture.t, dt, actions)` → `runtime.submit`.
  8. L = now − capture.t into the latency window (the max of the last 20 is the estimate).
- `lead_s` (`:99`) = max(`refill_s`, L + dt): request before the queue runs out.
- `refill_now(cmd_id)` (`:111`): a fresh chunk at once, taken only for the command being supplied (else counted in
  `refill_ignored`).

## 6. Commitment (`system1/commit.py`)
- `FixedCommit` (default): the whole chunk.
- `PhaseCommit`: through the first low-speed point after the floor (PACE).
- `ConsensusCommit` (`--speculative-predict`, opt-in):
  - switches decided per step by strict majority (`majority_switches`, `:84`); a tie keeps the value in effect
    (previous step's, or the robot's from `last` at step 0);
  - the chosen candidate is the one closest to the mean of those following the majority;
  - the agreeing prefix ends at the first step after the floor without a strict majority;
  - a disputed switch at or after the floor stops the commitment before it.
- Never below `min(floor, horizon)`.

## 7. Remote policy and transport
- `RemotePolicy.act_samples(obs, subtask, k)` (`remote.py:30`): `wire.encode_request` (v2, `samples` = K, WebP-90
  frames by default) → `transport.request("POST", "/act")` → `decode_response` → K `ActionChunk`s with timings
  (`network`, `round_trip`) and `info` (π0.5²'s chosen subtask in planner mode).
- `HttpTransport` / `H3Transport`: one kept connection; one retry reusing the same `X-Request-Id`.
- Server side, `Endpoint` (`transport/endpoint.py:101`):
  - `admit` checks the token and declared length before reading the body;
  - `ReplayCache` runs each request id at most once, so a retry never runs π0.5² twice.
- `serve.py`, `Pi052Policy.act_samples` (`:90`):
  1. preprocess once;
  2. in planner mode, generate the subtask once (`plan_state` kept across requests until the task text changes);
  3. `Draws` repeats the batch K times → K noise draws in one forward pass;
  4. `to_action_space` snaps switch dimensions to the robot's levels;
  5. timings: preprocess, plan, infer, postprocess.

## 8. Execution monitor (`system1/monitor.py`; opt-in via `--execution-monitor`)
- `observe(TickRecord)` (`:170`), in the loop thread:
  1. commanded change = `EffectModel.command(before, action)`; measured = `tracked(after)`;
  2. over a window: measured change vs a linear map of this and the previous window's commands;
  3. directions never exercised in calibration are not scored;
  4. score > conformal threshold → one `Deviation(epoch, cmd_id, t, score, threshold)` per excursion.
- `realtime.on_deviation` (`:330`): `owner.post(d)` (S2 decides) and `runner.refill_now(d.cmd_id)` (S1 replans at once).
- Contact-blind: deliberate contact and a blocked arm look alike (`blocked` scenario fails).

## 9. Messages on this side
| Message | From → to | Carries |
|---|---|---|
| `AcceptedGoal` / renew / review / `Revoke` | owner → runtime | cmd_id, instruction, operation (subtask), transition, lease |
| `TimedChunk` | runner → runtime | cmd_id, obs_id, t_start (observation time), dt, committed actions |
| `ExecutionAck` | runtime → owner (drained) | activated · revoked · rejected · chunk_rejected · expired · dropped · stopped |
| `Capture` | loop → runner, owner, watcher | obs_id, t, observation (images, state, task) |
| `TickRecord` | loop → monitor, logs | guard result, before / after capture, step_ms, lag_s |
| `Deviation` | monitor → owner (+ runner refill) | cmd_id, t, score, threshold |
| wire v2 request / response | runner ↔ GPU service | observation, subtask, K / K chunks, timings, info |

## 10. Timeline: a new command until steady supply (fixed commitment, π0.5² remote; latencies from the 2026-10-21 live run)
| t | What happens |
|---|---|
| 0.00 | Owner `accept`s cmd 1 `SWITCH_TO("pick up the steak")`; it is pending, the robot holds |
| 0.00 | Runner: `target()` = cmd 1, nothing queued → request on capture t = 0.00 |
| ~0.65 | Chunk (50 actions, 2.5 s) back; actions due before 0.65 skipped; `submit` accepted |
| next tick | First due action applied → `activated` ACK → the owner records cmd 1 active |
| … | L ≈ 0.65 s → floor = ⌈1.3/0.05⌉ + 1 = 27; lead = max(1.0, 0.70) = 1.0 s |
| ~1.5 | Less than 1.0 s queued → next request; its chunk replaces the queue from its first due time (blended) |
The cold first request took 3.96 s in the live run; requests after it were p50 650 ms / p90 845 ms.

## 11. Evidence (2026-10-22)
| Suite | Result |
|---|---|
| core `uv run pytest -q` (runtime, runner, commitment, monitor, transport, owner, S2) | 265 passed, 1 skipped, 2 timing deselected |
| `-m timing` (pacing, SO-101 real-time budget) | 2 passed (one earlier failure under machine load) |
| executor tests (harness, serve) | 9 passed |
| scenario matrix, scripted S1 | 17/19 (`blocked` contact-blind; `step-events` needs an S1 that completes steps) |
| π0.5² live, 30 s (2026-10-21) | real-time factor 0.971, 0 starved, 9 continuous overshoot ticks |

## 12. Proper vs declared, open
- Proper: floor and lead time come from measured latency; the runtime alone decides what is active; replay safety on
  the server; switch majority with the value in effect.
- Declared tuned constants: `refill_s` 1 s, `tolerance` 0.5, `smooth` 3, `slow_fraction` 0.3 (T8a measures them on
  π0.5²), monitor α.
- Open:
  - K-sample latency, draw diversity and memory on the GPU;
  - whether the real adapter accepts a K-row batch;
  - the contact-aware monitor;
  - phase-boundary and π0.5² subtask-switch step events (S1 info is logged but not yet posted to the owner);
  - a native control core is needed only above ~100–200 Hz (research brief 2026-10-22).
