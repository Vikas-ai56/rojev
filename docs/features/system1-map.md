# System 1 map: how actions reach the robot (code as of 2026-11-10)

How System 1 keeps the robot moving: control loop → guard → execution runtime ← S1 runner ← policy (XR-1 or π0.5²
over the network). System 2 never sends actions; its only input to this side is a command from the owner (see
[`system2-map.md`](system2-map.md)). Whole system: [`pipeline.md`](pipeline.md). Details per feature:
`s1-supply-and-commitment.md`, `execution-monitor.md`, `remote-models.md`, `live-harness.md`, `v3-runtime.md`,
`app-session.md`, `robot-interface.md`.

## 0. Call map
```
evaluate.main → executors/libero/evaluate.py                  one Session per LIBERO task episode
 └ Session     → rojev/app/session.py                         the shared assembly: same loop / guard / runtime / runner
     ├ LiberoBackend → executors/libero/backend.py             LeRobot's LiberoEnv + its observation processing
     └ RemotePolicy  → GPU box: services/lerobot-policy/serve.py → LeRobotPolicy (adapter.py)   port 8007
realtime.main → executors/robocasa-pi052/realtime.py          legacy RoboCasa assembly, one thread each
 ├ RoboCasaBackend  → realtime.py + render_context.py          the Backend; GL teardown fixed before the env is built
 ├ ControlLoop      → rojev/control/loop.py                    one tick per control period (20 Hz), or run_ticks in lockstep
 │   └ Guard        → rojev/control/guard.py                   watchdog, owner heartbeat, stop
 │       └ ExecutionRuntime → rojev/control/execution.py       what is physically active; one action per tick
 ├ on_tick          → FrameHistory (history.py), ExecutionMonitor (rojev/system1/monitor.py), SwitchCues, ticks.jsonl, video
 ├ FrameCheck       → harness/frame_check.py                   are the camera images real? (run validity)
 └ S1Runner         → rojev/system1/runner.py                  keeps the active command supplied with chunks
     ├ CommitPolicy → rojev/system1/commit.py                  how much of a chunk to commit
     └ make_policy  → --s1 pi052 | xr1
         ├ pi052: RemotePolicy → rojev/system1/remote.py → wire.py (v3, K samples, continuation)
         └ xr1:   HistoryPolicy (history.py) → RemotePolicy        4 frames 2 ticks apart, actions → to_action_space
               └ transport → rojev/transport/{http,h3,endpoint}.py  kept connection, replay-safe retry
                     └ GPU box: PolicyService (rojev/system1/service.py)
                         ├ executors/robocasa-xr1/serve.py → XR1Policy (+ xr1_inputs.py)    port 8004
                         ├ executors/robocasa-pi052/serve.py → Pi052Policy (opt-in RTC)    port 8000
                         └ executors/so101-groot/serve.py → GrootPolicy (Isaac format)     port 8005
```
- Dependencies point down: protocol, robot ← control ← system1 (+ transport). Nothing in System 1 imports System 2 or
  the owner.
- Robot specifics enter only as an `ActionSpec` and an `EffectModel` (`embodiment.py`); model specifics only inside an
  executor's `Policy` (inputs, outputs, history).

## 1. Entrypoints
Shared assembly (`rojev/app/session.py`): the same runtime, guard, loop and S1 runner as the RoboCasa app, but
whole-chunk commitment only, no motion monitor, no frame check. In lockstep S1 requests carry no deadline. LIBERO
(`executors/libero/evaluate.py`) runs it once per episode with a fresh environment and the whole task sent to S1.

RoboCasa app (`realtime.py:503`):
- `RoboCasaBackend` (`:97`) = the `Backend` (`observe`, `step`); `render_context.install()` (`:104`) before the env.
- `ExecutionRuntime(PANDA_OMRON, epoch=1, last=START, blend_s, stop_s)`, `Guard(runtime)`.
- `FrameHistory((4 − 1) · 2 + 1)` (`:529`), fed every tick by `on_tick` (`:531`) and once with the initial capture.
- `ControlLoop(..., realtime=not a.lockstep, on_tick)` (`:548`).
- `S1Runner(runtime, RecordingPolicy(make_policy(...)), loop.latest, loop.now, commit, on_step, on_plan,
  deadline=not a.lockstep)` (`:553`).
- `make_policy` (`:322`): `--policy remote` + `--s1 pi052` → `RemotePolicy`; `--s1 xr1` → `HistoryPolicy(remote,
  history, *XR1_HISTORY, to_action_space)`; `skills` / `wiggle` are stand-ins. `--s1` defaults to `xr1` (2026-10-25).
- `FrameCheck(loop.latest, CAMS, every_s)` (`:594`) is one of the run's threads.
- `gc.collect(); gc.freeze()` (`:617`) before the threads start (a full GC pass measured 45–96 ms).
- Run loop (`:622`):
  - real time: the loop and runner run on their own threads; the main thread only posts user speech;
  - **lockstep**: no loop / runner / owner threads; per iteration `owner.step()`, then if the planning or interaction
    lane is busy or the owner still has an event → wait (no tick); else `runner.step()` then `loop.run_ticks(1)`. Every model's computation takes zero simulated time.
- Flags (S1 side): `--s1`, `--policy remote|skills|wiggle`, `--service`, `--lockstep`, `--commit-policy
  fixed|phase`, `--speculative-predict --samples K`, `--execution-monitor M.json`, `--blend`, `--stop-s`,
  `--lease`, `--frame-check-every S`, `--record-frames S`, `--video`, `--viewer`.
- Outputs: `config.json`, `ticks.jsonl`, `s1_requests.jsonl` (wall ms, continuation, info), `commits.jsonl`,
  `summary.json` (S1 stats, `frame_check`), `view.mp4`, `frames/`.

## 2. Control loop (`control/loop.py`)
- `_run` (`:85`): real time → tick k due at `t0 + k·dt` on the wall clock; lateness recorded (`lag_s`, `max_lag_s`),
  never skipped. Not started in lockstep.
- `run_ticks(n)` (`:80`): n ticks at once on the simulated clock (lockstep driver).
- `_tick` (`:107`): `guard.step(t)` → `backend.step(action)` → new `Capture(obs_id+1, t, obs)` into the latest-value slot →
  `on_tick(TickRecord(result, before, after, step_ms, lag_s))`.
- `latest()` (`:62`): the newest capture. S1, S2 and the checks read only this, never a backlog.

## 3. Guard (`control/guard.py:29`)
- Owner heartbeat older than `owner_timeout_s` (2 s) → `hold("owner lost", drop_commands=True)`.
- RUNNING without a valid action for `watchdog_s` (0.5 s) → `hold("starved", drop_commands=False)`.
- Runtime exception → `hold("runtime error…", drop_commands=True)`.
- Dropping a command sends a `dropped` ACK, so the owner re-decides; a starved command resumes with its next chunk.

## 4. Execution runtime (`control/execution.py`); the authority on what is active
- `accept(goal)` (`:123`): refuses stale epoch / stopped / revoked / not newer / expired. A non-moving operation
  (ABORT, WAIT, ASK, COMPLETE) becomes active at once as a hold. A moving one becomes `pending`; `pause_hold` (or not
  running) holds until its first chunk.
- `renew` (`:150`), `review` (`:160`), `revoke` (`:174`): lease extension; review (keeps moving, switches frozen, lease
  shortened); revoke (drops its queued actions at once).
- `submit(chunk)` (`:220`): refuses wrong epoch, unknown, revoked or expired command, wrong `dt`, wrong shape, a
  non-moving command, paused, held. Actions already due are skipped. A chunk for the pending command splices in at its
  first due time; one for the active command replaces its own future queue.
- `target()` (`:267`): the command that needs chunks: pending, else the active one while moving (or starved).
- `queued_from(cmd_id, t)` (`:287`): the actions queued for the command from time t (the continuation's source);
  `queued_end` (`:294`): when its queue runs out (the last action's due time plus a tick).
- `tick(now)` (`:299`), in order:
  1. lease expiry;
  2. skip past-due actions;
  3. the due action: the first one of the pending command activates it (`activated` ACK);
  4. bounds (clamps reported as `violations`);
  5. inertialized blend of a new segment;
  6. otherwise the hold, with a decelerating ramp;
  7. irreversible switches frozen outside a running, unreviewed chunk;
  8. projection onto the bounds.
- `active_goal()` (`:277`): the active command under the lock (the step watcher's read); `last_action()` (`:282`):
  the last applied action (switch state in effect).
- ACKs are an outbox the owner drains (`drain_acks`, `:394`); `seq` strictly increases.

## 5. S1 runner (`system1/runner.py`); supply
- `step` (`:142`), on its own thread every `poll_s` (lockstep: once per tick from the run loop):
  1. `goal = runtime.target()`; none → nothing to do (holding, or no command).
  2. Request only if a refill is pending for this command, or `chunk_due`: nothing is queued, or the current chunk's
     replacement point (its `execute_steps`, else `queued_end`) is within `lead_s()` (half a tick counts as reached).
  3. `capture = latest()`; observation task = `goal.instruction`; subtask = `goal.operation.subtask` ("" = the policy
     works from the task alone).
  4. `_continuation` (`:200`): for a policy that `accepts_continuation`, after the first latency measurement,
     `Continuation(queued_from(cmd, capture.t), ⌈L/dt⌉)` = what runs while it thinks (`stats.continuations`).
  5. `policy.act` (or `act_samples(K)` for consensus) with `deadline_s()`: one request, one upload.
  6. `floor(horizon)` (`:124`) = ⌈2L/dt⌉ + 1 (whole chunk before the first measurement); `floor_over_chunk` counted.
  7. `commit.plan(candidates, spec, floor, runtime.last_action())` → the committed prefix.
  8. `TimedChunk(epoch, cmd_id, obs_id, t_start = capture.t, dt, actions)` → `runtime.submit`.
  9. L = now − capture.t into the latency window (the max of the last 20 is the estimate).
  10. No subtask on the command and the chunk reports one (π0.5² planner mode): `_planner_moved_on` (`:209`) posts a
      `PlannedSubtask` (`on_plan`) and, on a switch, an unchecked `StepEvent(source="s1-planner")` (`on_step`: to the
      step watcher as a claim when one runs, else to the owner). XR-1 reports no subtask, so neither fires.
- `lead_s` = L + dt when the clock runs while S1 thinks (real time, model delay); 0 in lockstep, so the robot runs
  exactly the model's `execute_steps` (GR00T on LIBERO: 8 of 16) or the whole chunk. The rest of a chunk stays
  queued behind its replacement point: a late answer leaves the robot moving instead of holding.
- `deadline_s` (`:130`): one chunk duration (an older answer is entirely in the past); `TimeoutError` → `timeouts` +
  an immediate new request. None with `clock_runs_while_thinking=False` (lockstep: simulated time does not move while S1 thinks, so a
  wall-clock deadline only abandons good answers; first lockstep runs: 217 of 292 starved before this; that run is
  not in `results/`, and whether the unit was requests or ticks is unresolved).
- `refill_now(cmd_id)` (`:133`): a fresh chunk at once, taken only for the command being supplied (else counted in
  `refill_ignored`).

## 6. Commitment (`system1/commit.py`)
- `FixedCommit` (`:116`, default): the whole chunk.
- `PhaseCommit` (`:131`): through the first low-speed point after the floor (PACE).
- `ConsensusCommit` (`:158`, `--speculative-predict`, opt-in):
  - switches decided per step by strict majority (`majority_switches`, `:84`); a tie keeps the value in effect;
  - the chosen candidate is the one closest to the mean of those following the majority;
  - the agreeing prefix ends at the first step after the floor without a strict majority;
  - a disputed switch at or after the floor stops the commitment before it.
- Never below `min(floor, horizon)`.

## 7. Policies, wire and transport
### Robot side
- `RemotePolicy.act_samples(obs, subtask, k, timeout_s, continuation)` (`remote.py:34`, `accepts_continuation`):
  `wire.encode_request` (v3: K, WebP-90 frames, optional prefix + delay steps) → `POST /act` → K `ActionChunk`s with
  timings (`network`, `round_trip`) and `info` (π0.5²'s subtask in planner mode).
- `HistoryPolicy.act` (`history.py:47`; does not accept a continuation, XR-1 is synchronous):
  1. `FrameHistory.window(obs, 4, 2)` (`:31`): the capture located by identity in the history (else appended), then
     4 observations 2 ticks apart, clamped at the oldest;
  2. images as `<cam>@k`, states stacked, on the unchanged wire;
  3. returned actions → `to_action_space` (`embodiment.py:45`): switch dimensions snapped to RoboCasa's levels.
- `HttpTransport` / `H3Transport`: one kept connection; one retry reusing the same `X-Request-Id`; the transport bounds
  the whole call and closes the connection past it.

### GPU side
- `Endpoint` (`transport/endpoint.py:101`): `admit` (`:109`) checks the token and declared length before the body;
  `ReplayCache` (`:62`) runs each request id at most once.
- `PolicyService.handle` (`system1/service.py:29`): `/health`, `/reset`, `/act` → `Policy.act_samples`, one lock; a
  continuation reaches the policy only if it `accepts_continuation`. `serve_policy` (`:52`): HTTP on 127.0.0.1 (+ H3),
  stops after `idle_min`.
- `XR1Policy.act` (`executors/robocasa-xr1/serve.py:43`):
  1. `unpack` (`xr1_inputs.py:69`): frames per camera + states back from `<cam>@k`;
  2. `messages` (`:58`): the checkpoint's prompt (`/no_cot`), instruction = subtask else task, `center_crop` 0.95
     (`:47`);
  3. `state14` (`:38`): end-effector-first 14-dim state (padded to 60 by the processor);
  4. processor → model (flash-attn 2, bf16) → `decode_action` → 16 steps of `ACTION_DIM`.
  Stateless: the robot sends the history, so a retried request answers the same.
- LeRobot policy service: any LeRobot checkpoint chosen by a config file, prepared as LeRobot's own evaluation does
  (text = the step, else the task); a non-number in the answer fails the request ([`remote-models.md`](remote-models.md)).
- GR00T for the SO-101 (`executors/so101-groot`): GR00T's own API bound to our SO-101 layout.
- `Pi052Policy.act_samples` (`executors/robocasa-pi052/serve.py:111`): preprocess once; planner mode generates the
  subtask once; `Draws` = K noise draws in one pass; `guidance` (`:100`) turns a continuation into LeRobot RTC
  arguments only with `--rtc H`; `to_action_space`; timings.

## 8. Run validity (executor)
- `render_context.install` (`render_context.py:28`): robosuite's `MjRenderContext.__del__` freed a replaced
  simulator's GL resources in the *current* context, i.e. the live one, after `gc.collect()`; `_free_in_own_context`
  (`:16`) makes the dying context current, frees, restores. Every live result before 2026-10-22's fix is withdrawn.
- `FrameCheck` (`harness/frame_check.py:36`): every `every_s` the newest capture, per camera `frame_problem` (`:24`):
  std < 3 → blank, mean horizontal neighbour difference > 25 → noise. `report()` → `summary["frame_check"]`; an
  invalid run prints a warning, fails `frames_real` (`harness/invariants.py`) and is skipped by `score_matrix`.
- Frozen frames are not detected (a still scene is legitimately identical).

## 9. Execution monitor (`system1/monitor.py`; opt-in via `--execution-monitor`)
- `observe(TickRecord)` (`:160`), in the loop thread:
  1. commanded change = `EffectModel.command(before, action)`; measured = `tracked(after)`;
  2. over a window: measured change vs a linear map of this and the previous window's commands;
  3. directions never exercised in calibration are not scored;
  4. score > conformal threshold → one `Deviation(epoch, cmd_id, t, score, threshold)` per excursion.
- `on_deviation` (`realtime.py:521`): `owner.post(d)` (S2 decides) and `runner.refill_now(d.cmd_id)` (S1 replans).
- Contact-blind: deliberate contact and a blocked arm look alike (`blocked` scenario fails).

## 10. Messages on this side
| Message | From → to | Carries |
|---|---|---|
| `AcceptedGoal` / renew / review / `Revoke` | owner → runtime | cmd_id, instruction, operation (subtask), transition, lease |
| `TimedChunk` | runner → runtime | cmd_id, obs_id, t_start (observation time), dt, committed actions |
| `ExecutionAck` | runtime → owner (drained) | activated · revoked · rejected · chunk_rejected · expired · dropped · stopped |
| `Capture` | loop → runner, owner, watcher, frame check, history | obs_id, t, observation (images, state, task) |
| `TickRecord` | loop → history, monitor, logs | guard result, before / after capture, step_ms, lag_s |
| `Deviation` | monitor → owner (+ runner refill) | cmd_id, t, score, threshold |
| `PlannedSubtask` / `StepEvent` | runner → owner / watcher (π0.5² planner only) | cmd_id, subtask; step done, source `s1-planner` |
| wire v4 request / response | runner ↔ GPU service | observation (history-encoded for XR-1), subtask, K, continuation / K chunks, timings, info, `execute_steps` |

## 11. Timeline: a new command until steady supply (XR-1 parity setup: lockstep, fixed commitment; as run on 2026-10-22, before `execute_steps`)
| sim t | What happens |
|---|---|
| 0.00 | Owner `accept`s cmd 1 `SWITCH_TO(None)` with the task text; pending, the robot holds |
| 0.00 | Run loop: lanes idle → `runner.step()`: `target()` = cmd 1, nothing queued → request on capture t = 0.00 (history = that frame repeated 4×) |
| 0.00 | Simulated time waits for the answer (wall ≈ 0.5 s); 16 actions (0.8 s) submitted (first request: floor = whole chunk); L = 0 in simulated time → floor 1 from now on |
| next tick | First action applied → `activated` ACK → the owner records cmd 1 active |
| ~0.70 | One tick of actions left (< `lead_s` = 0.05 s) → next request on the newest capture; history now 4 real frames 2 ticks apart |
| … | One request per ~15 ticks (450 ticks / 31 requests per CloseToasterOvenDoor episode); no deadline, no starvation. Now (no `refill_s`, `queued_end` to the chunk's end, XR-1 sends no `execute_steps`): one request per 16 ticks, XR-1's own recipe |
In real time a 0.5 s latency already needs a floor of ⌈2·0.5/0.05⌉ + 1 = 21 > 16 actions, so supply is the bottleneck
(F13 measured L ≈ 0.9 s, floor 38, 651 of 985 ticks starved); why lockstep is the sim evaluation mode (`decisions.md`
2026-10-23).

## 12. Evidence (tests re-run 2026-11-10)
| Suite / run | Result |
|---|---|
| core `uv run pytest -q` (incl. exact actions per chunk in lockstep, replacement point in real time) | 361 passed, 1 skipped, 3 timing deselected; `-m timing` on a loaded Mac: 1 passed, 2 pacing tests over their lag budget (they do not reach the S1 runner) |
| executor `executors/robocasa-pi052` `uv run pytest -q` (harness, serve, history, render context, frame check, campaign) | 37 passed |
| `executors/libero` (fake LIBERO under LeRobot's real env and processors; zero policy through the real `Session`) | 22 passed |
| `services/lerobot-policy` (stand-in policy) | 34 passed |
| XR-1 parity, their target50 cases, 4 tasks × 5 seeds, lockstep (`results/xr1-parity/20261022-fixed`) | 18/20 (CloseToasterOvenDoor 4/5, TurnOnElectricKettle 5/5, PickPlaceCounterToCabinet 5/5, RinseSinkBasin 4/5) vs their 20/20 |
| XR-1 request wall time, same runs (1000 requests: 1020 minus each run's cold first, Mac ↔ GPU box) | p50 511 ms, p90 731 ms |
| Interactive run 5 (XR-1 + Jev, lockstep, `results/v1-interactive/run5`) | task success at 36.75 s; frames valid |
| XR-1 real time, F13 (`results/campaign/20261024-134629/F13`) | real-time factor 0.806; 64 requests, 36 accepted, 13 timeouts; 651 of 985 ticks starved; no success |
| π0.5², earlier delay / RTC A/B | withdrawn (corrupted frames); not re-run |

## 13. Proper vs declared, open
- Proper: floor and lead time from measured latency; the runtime alone decides what is active; replay safety on the
  server; history and action mapping in the executor, not the core; lockstep removes model time instead of tuning
  deadlines; frame validity is checked on every run.
- Declared tuned constants: `tolerance` 0.5, `smooth` 3, `slow_fraction` 0.3, monitor α, frame-check
  thresholds 25 / 3 (measured margins).
- Spoofs / sim-only: `render_context` (GLFW/CGL only); lockstep itself (no robot can pause the world).
- Open:
  - XR-1 in real time (needs S1 next to the robot, or a policy that accepts a continuation: RTC / tail reuse);
  - XR-1 input parity test against its own client (bit-level), and a fair task sample beyond 4 tasks;
  - no completion signal: S1 keeps executing a finished task (run 5);
  - the contact-aware monitor; K-sample consensus on XR-1;
  - a native control core is needed only above ~100–200 Hz.
