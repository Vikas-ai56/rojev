# v3 runtime core: messages, execution runtime, guard, control loop, state owner, S2 worker (v1 steps 1–4)

Code: `rojev/protocol/`, `rojev/robot/action_spec.py`, `rojev/control/` (execution, guard, loop, robot_backend), `rojev/owner/state_owner.py`, `rojev/system2/` (decider, worker), `rojev/eval/deciders.py`. Spec: `docs/plan/v3-v1-spec.md` §3–6, §13–14.
Elsewhere: S1 supply, latency, commitment → [`s1-supply-and-commitment.md`](s1-supply-and-commitment.md) · execution monitor → [`execution-monitor.md`](execution-monitor.md) · transport, codecs, model services, WAN numbers → [`remote-models.md`](remote-models.md) · live entrypoint, scenario matrix, spoofs → [`live-harness.md`](live-harness.md).

## Threads
```
ControlLoop (20 Hz)  Guard.step → ExecutionRuntime.tick → backend.step → latest Capture
S1Runner             runtime.target() → policy → TimedChunk → runtime.submit
DecisionWorker ×1–2  ObservationSnapshot → decider → DecisionProposal | DecisionFailure → owner
StateOwner           events (instruction, proposal, failure, deviation) + 50 ms tick → accept / renew / review / revoke
ACKs                 runtime outbox → owner (drained every owner tick); nothing waits for a model
```

## Messages (`rojev/protocol/`; every message carries `epoch`, times on the robot clock)
- `Observation{images, state, task}` · `Capture{obs_id, t, observation}` (`obs_id` + 1 per tick).
- `Operation{kind, subtask, target, dest, template, objects}`; kinds CONTINUE · SWITCH_TO · RETRY · ABORT · GUARDED_WAIT · ASK · OBSERVE · COMPLETE; moving = SWITCH_TO / RETRY / OBSERVE; `subtask=None` = S1 plans its own subtasks.
- `AcceptedGoal{cmd_id, instruction_rev, operation, transition: immediate | pause_hold, expires_t (lease), instruction}`; S1 is conditioned on `instruction`, never on stale task text.
- `Revoke{cmd_id, reason}` · `TimedChunk{cmd_id, obs_id, t_start, dt, actions H×D, timings_ms}`: action k due at `t_start + k·dt`.
- `ExecutionAck{cmd_id, seq, status, t, applied_index, detail}`; status `activated` · `revoked` · `rejected` (a command message) · `chunk_rejected` (one chunk; the command stands) · `expired` · `stopped`; `seq` strictly increasing.
- `UserInstruction{text}` · `Deviation{cmd_id, t, score, threshold, source}` · `DecisionFailure{request_id, error}`.
- `Keyframe{cmd_id, subtask, instruction_rev, capture, evidence}`: a step judged done, with the observation the judgement used.
- `ObservationSnapshot{request_id, now, capture, instruction_rev, instruction, active_cmd, active_operation, active_since, keyframes, trigger, review, deviation}`.
- `DecisionProposal{request_id, obs_id, t_capture, instruction_rev, base_active_cmd, operation, scores, preconditions, expires_t, evidence}`; `Precondition`, `Advice` defined, not used yet.

## Execution runtime (`control/execution.py`, `ExecutionRuntime`)
- One action per tick; modes IDLE · RUNNING · TRANSITION · HOLD · STOPPED; one lock (owner, S1 runner and loop threads call it).
- `accept`: refused (`rejected`) on another epoch, STOPPED, revoked id, id not newer, lease already over. A newer goal supersedes a pending one (`revoked`). Non-moving operations (wait / abort / ask / complete) hold at once and are `activated` at once.
- Transitions: `immediate` keeps the old command until the new chunk is due (blended); `pause_hold` (or not RUNNING) drops the queue and holds.
- Active only when its first action is applied (`activated`); a pending goal whose lease ends first `expired`.
- `renew` extends the lease of the active or pending command and ends its review; `revoke` drops its queued actions at once (a queued gripper release never happens).
- `review(cmd, until_t)`: keeps moving, freezes irreversible switches (`TickResult.review`), lease capped at `until_t`; renew or a new command ends it.
- Lease ends a command in any mode (running, held, starved) → `expired` + hold.
- `submit` → `chunk_rejected` on: epoch, STOPPED, revoked, neither active nor pending, lease over, `dt` ≠ control period (resample in the adapter), wrong shape, non-moving command, active paused for a transition, holding for any reason but "starved", entirely in the past.
- Elapsed actions skipped (`TickResult.skipped`), never replayed; a new chunk replaces the queue from its first due time; an old command's chunk cannot delay a scheduled takeover.
- Tick order: lease expiry → skip elapsed → activate pending → incoming action checked and clamped (`violations`, upstream only) → blend → decelerate → starvation count → irreversible gating → projection onto the bounds (not reported).
- Blend: each new segment keeps 1 − s(k/n) of the jump in continuous dims over `blend_s` (0.2 s), s = quintic smoothstep; switches never blended.
- Decelerating hold: `stop_s` > 0 ramps continuous dims with a numeric hold value to it when motion turns into a hold (library default 0 = instant; `realtime.py` 0.3 s); also on `stop`.
- Irreversible switches change only on a chunk action of a RUNNING command outside review; held otherwise (`held_switches`).
- `starved_ticks`: ticks where an active moving command (running or starved-hold) had no due action.
- `ActionSpec.hold` refuses to guess KEEP dims: seed `last` with the robot's current command.

## Guard (`control/guard.py`)
- No owner heartbeat for `owner_timeout_s` (2 s) → hold "owner lost", active and pending command dropped (`dropped` ACK; a new command from the owner moves again); RUNNING without a valid chunk action for `watchdog_s` (0.5 s) → hold "starved" (resumes when chunks return); runtime exception → hold, commands dropped, counted in `errors`.
- `stop()` needs nothing but the loop; STOPPED is final for the epoch.

## Control loop (`control/loop.py`, `ControlLoop`)
- Backend with its own `clock` (any `rojev.robot.Robot`) → `Rate` on that clock; without one (RoboCasa) tick k due at k·dt, wall-paced (`realtime=True`) or back to back; `run_ticks(n)` for lockstep.
- Lag reported (`lag_s`, `max_lag_s`), never skipped; newest observation in a latest-value slot (`latest()`).
- `TickRecord{tick, before, after, step_ms, lag_s}`: the observation the action was applied to and the one it produced (no tick lost to the monitor).

## Robot plug-ins (the runtime never assumes a robot)
- `ActionSpec` per dim: bounds, hold (number or `KEEP`), switch threshold, irreversible (must be a switch).
- Any `rojev.robot.Robot`: `RobotBackend` + `joint_target_spec(robot, switches, irreversible)`; actions = joint targets, robot safety filter still applies.
- RoboCasa PandaOmron: `executors/robocasa-pi052/embodiment.py` (12-D, 20 Hz; arm / torso deltas and base velocity hold 0; gripper and base mode KEEP switches at 0.5; gripper irreversible).

## State owner (`owner/state_owner.py`, `StateOwner`)
- Single writer of instruction + revision, desired command, active record, keyframes, hold, dialogue. One thread: event queue + tick every `period_s` (heartbeat, ACK drain, interaction deadline, hold upkeep, heartbeat decision every `decide_every_s` = 1 s unless holding).
- Instruction: revision + 1, `complete` cleared, user `Turn` added, `ACK` utterance at once (`say`), then a decision (`instruction`). Moving command: `review` (default) → `runtime.review`, lease shrinks to `review_s` (1 s); `revoke` (or a refused review) → revoked at once (stop-and-go baseline).
- Proposal refused (logged with the reason) unless: same epoch, same revision, newer than the last applied request, a request this owner made, not a planning proposal while an interaction request pends, not a superseded interaction request, about the current desired command, not a routine planning proposal while a hold stands, not expired, no preconditions (unchecked until step 6).
- CONTINUE or the same operation renews the lease (`lease_s` 3 s) and ends review; if the desired command predates the current revision and moves, it is **reaffirmed**: same operation, new command id, new instruction, always `immediate`, no pause. The `continue` event records `renewed`. COMPLETE sets `complete` (no more heartbeat decisions) and holds; anything else = new command.
- Transition: configured one between two moving commands; `pause_hold` into or out of a hold.
- `active_since` kept across a reaffirmation (same operation activated again).
- ACKs: newest `seq` only; `activated` sets active (clears an old command's deviation, step event and review); `rejected` / `expired` / `dropped` of the desired command → desired falls back to the still-running command, new decision (`cmd N rejected|expired|dropped`); `chunk_rejected` logged only.
- Deviation of the active command → `snapshot.deviation`, trigger `deviation`; `StepEvent` about it → `snapshot.step_event`, trigger `step` (planning); other command or epoch → logged as stale.
- Templated operation accepted (ASK, report) → `Utterance` + robot `Turn`; snapshots carry the last `dialogue_turns` (8) turns.

### Holds (2026-10-22; fixes the stop-then-restart bug)
- ABORT / GUARDED_WAIT / ASK accepted in answer to a change (`instruction`, `deviation`, `hold`) = standing `Hold{operation, since, interrupted}` (`snapshot.hold`).
- While it stands: no heartbeat; routine planning requests not sent (`planning held`) and their proposals refused; the owner renews the hold command's lease itself; every `hold_recheck_s` (if set) the interaction lane is asked again (trigger `hold`).
- Released by a moving operation or COMPLETE answering a change (`hold released`); `interrupted` lets the decider resume the step.

### Two lanes (step 4.4)
- `interaction=None` → one lane. Given: triggers `instruction`, `deviation`, `hold` → interaction worker; heartbeat, `step` and failed / expired / dropped commands → planning.
- Priority: while an interaction request pends, planning requests are deferred (the latest is sent once the interaction decision is applied; an escalation replaces it) and planning proposals refused.
- Release by its proposal, its `DecisionFailure`, or `interaction_timeout_s` (1 s); on failure, timeout or refusal the same trigger escalates to planning, so every change is answered.

### Progress keyframes (always on)
- Recorded when an accepted SWITCH_TO to another subtask, or COMPLETE, moves past the step that is actually running; with the observation the decision used.
- Not for ABORT, waits, asks, retries, reaffirms or never-started steps; kept across deviations and revisions; `snapshot.keyframes`.
- Evidence = the step event about the running command (`"<source>: <detail>"`) if one came, else the decider's judgement.

## S2 worker (`system2/worker.py`, `DecisionWorker`)
- Contract `system2/decider.py`: `Decider.decide(snapshot) → Verdict{operation, reason, scores, preconditions, evidence}`.
- Own thread; newest request wins (older unstarted one replaced, `replaced` counted); `busy` gates heartbeat requests.
- Proposal stamped with the snapshot's epoch, request id, revision, base command, observation; expiry `proposal_ttl_s` (2 s); evidence adds reason, trigger, decide_ms, decider.
- Decider error → `DecisionFailure` posted at once (releases the interaction lane); the robot keeps doing what it was doing.

## Baseline deciders (`eval/deciders.py`, evaluation only)
- `ScriptedSequence` (`scripted`): each authored subtask for its budget after activation (`realtime.py`: `chunks_per_subtask` × 2.5 s), COMPLETE after the last; ignores instruction changes.
- `TruthSequence` (`oracle`): step after the latest one whose simulator truth holds now (`next_after_latest_done`); may go back.
- `NativePlanner` (`native`): one SWITCH_TO without subtask; π0.5²'s own planner chooses subtasks.
- `AlwaysContinue` (`continue`): interaction-lane baseline; every change answered with CONTINUE (= reaffirm).

## v1 role split (decided 2026-10-21) and what is planned
- v1 = plan Route A: π0.5² picks the next step and moves; Jev answers only changes (continue / revise / wait / ask / abort) and ambiguous checks; MJev checks end states and grounds objects. Helix-type route B (small S1 on a learned Jev latent, trained) = v2.
- **Built 2026-10-22:** step-completion events (`StepEvent` → planning lane; end-state checks in `features/jev-interaction.md`); Jev v1 interaction decider (same doc).
- **Planned:** π0.5²'s planner switching subtask (`ActionChunk.info`) and phase boundaries as step-event sources; heartbeat slowed to just under the lease once events cover steps; precondition checks.

## Tests (no models; `uv run pytest`, re-run 2026-10-22: 74 passed, 2 `timing` deselected)
| File | Tests | Pins |
|---|---|---|
| `tests/test_control_execution.py` | 33 | intake / rejection reasons, activation, skip, revoke of a queued release, irreversible gating, leases (also starved), transitions, supersede, blend in bounds, violations upstream only, review, decelerating hold, starvation counting, guard paths (owner-lost drop + recovery), PandaOmron spec |
| `tests/test_owner.py` | 39 | proposal refusal reasons, revoke / review / reaffirm (always immediate), ACK fallback incl. `dropped`, COMPLETE, deviations, two lanes (routing, deferral replayed, supersede, escalation on failure / timeout / refusal), keyframes (step-event evidence), `DecisionFailure`, holds (stop stays through heartbeats, resume, recheck, one-lane race), dialogue + utterances, step events, continue `renewed`; live multi-thread arm runs |
| `tests/test_system1_runner.py` | 10 + 2 `timing` | runner timing on simulated / lockstep time; SO-101 plugs in; `@pytest.mark.timing` real-time budgets run alone with `uv run pytest -m timing` (2026-10-22: 5 of 6 runs passed; one SO-101 lag-budget failure while other processes loaded the Mac) |
| `tests/test_architecture.py` | 3 | layering: imports point down the stack, nothing depends on `legacy_v2` |

## Measured live (RoboCasa on the Mac, 20 Hz, `executors/robocasa-pi052/realtime.py`)
Stand-in S1 (`wiggle`, a sleep + sine chunk; 2026-10-20, step 2):
| Latency, refill, run | Real-time factor | Max lag | Env step p50 / p95 | Chunk ticks | First chunk |
|---|---|---|---|---|---|
| 0.3 s, 1.0 s, 15 s | 1.00 | 3.5 ms | 41 / 45 ms | 295 / 302 | 0.35 s |
| 0.6 s, 0.8 s, 12 s | 1.00 | 3.3 ms | 38 / 42 ms | 236 / 249 | 0.65 s |

- Arms with a change of mind (MicrowaveThawingFridge, `wiggle` 0.3 s, revoke mode, 2026-10-20): scripted (14 s) and oracle (8 s) pick → revoke → pick, native (8 s) own planner → revoke → own planner; real-time factor 1.00, max lag ≤ 3.9 ms, 0 errors, 0 violations; scripted also refused one late chunk.
- Review vs revoke (native arm, `wiggle`, change at 4 s, 2026-10-20): review → 30/30 ticks in motion after the change, 1 reaffirm, 0 revokes; revoke → 6 hold ticks (0.3 s stop) before motion resumes; real-time factor 1.00, 0 errors, 0 violations.
- Two lanes (MicrowaveThawingFridge, scripted arm, `wiggle` 0.3 s, `--interaction continue`, change at 8 s, 2026-10-21): reaffirmed by the interaction lane at 8.0 s, 40/40 chunk ticks in the next 2 s, keyframe "pick up the steak" (rev 1, scripted evidence), 0 starved ticks, 0 violations.

π0.5² as S1 (A100, us-east, over the WAN; MicrowaveThawingFridge seed 1000, `revisable` arm, `--interaction rules`, "wait" at 12 s, "go on" at 15 s, review mode, fixed commitment, 30 s; 2026-10-21; `results/runs/pi052-live-*`; invariants re-checked 2026-10-22):
| Run | Real-time factor | Starved | S1 request p50 / p90 (after the first) | Floor | Violation ticks | Other 4 invariants |
|---|---|---|---|---|---|---|
| `pi052-live-1404` (before the adapter fix) | 0.97 | 0 | 732 / 841 ms | 34 | 486 (gripper, base mode) | 0 findings |
| `pi052-live-v2` (after) | 0.971 | 0 | 650 / 845 ms (cold first request 3.96 s: chunk in the past, first motion 4.65 s) | 37 | 9 (`base_y` ≤ 1.0077, last 0.4 s) | 0 findings |
- Both: "wait" → GUARDED_WAIT from the interaction lane at 12.0 s (moving command under review, its late chunk `chunk_rejected`), "go on" → the interrupted pick resumed at 15.0 s; no refused proposals; steak not picked in 30 s; keyframe for "pick up the steak" is the planner's time budget, not truth.
- Switch violations were the adapter's (`serve.py` `to_action_space`, decisions 2026-10-21); the 9 continuous overshoots are reported upstream as designed.
- Not run with π0.5²: a destination change, native planner mode, `revoke` mode, consensus / phase commitment.

## Open / not validated
- Reaffirm grasp delay (gripper frozen up to `review_s`) not measured (roadmap T8).
- All stand-in numbers are mechanism-level; performance claims need π0.5² runs (T8a / T8).
