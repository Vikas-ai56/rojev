# v3 runtime core: messages, execution runtime, guard, control loop, state owner, S2 worker

## Threads
```
ControlLoop (20 Hz)  Guard.step → ExecutionRuntime.tick → backend.step → newest Capture
S1Runner             runtime.target() → policy → TimedChunk → runtime.submit
DecisionWorker ×1–2  ObservationSnapshot → decider → DecisionProposal | DecisionFailure → owner
StateOwner           events + 50 ms tick → accept / renew / review / revoke; drains ACKs; guard heartbeat
```
- Nothing waits for a model; every consumer reads the newest observation, never a backlog.
- Lockstep (`realtime.py --lockstep`, the simulator evaluation mode, decisions 2026-10-23): no loop or runner thread;
  the main thread runs `runner.step()` then one tick, and skips the tick while a decision lane is busy, so S1 and S2
  computation take zero simulated time. Real time is reported alongside.

## Messages (`rojev/protocol/`; every message carries `epoch`; times on the robot clock)
- `Observation{images, state, task}` · `Capture{obs_id, t, observation}` (`obs_id` + 1 per tick).
- `Operation{kind, subtask, target, dest, template, objects}`. Moving kinds: SWITCH_TO, RETRY, OBSERVE; the runtime
  holds for ABORT, GUARDED_WAIT, ASK, COMPLETE; CONTINUE never becomes a command (the owner renews the lease).
  `subtask=None` = S1 plans its own steps.
- `AcceptedGoal{cmd_id, instruction_rev, operation, transition, expires_t, instruction}`: `expires_t` is the lease;
  S1 is conditioned on `instruction` (the task text in force), never on the environment's text.
- `TimedChunk{cmd_id, obs_id, t_start, dt, actions}`: action k due at `t_start + k·dt`. `Revoke{cmd_id, reason}`.
- `ExecutionAck{cmd_id, seq, status, t, applied_index, detail}`, `seq` strictly increasing. Status: `activated` ·
  `revoked` · `rejected` (a command message) · `chunk_rejected` (one chunk; the command stands) · `expired` ·
  `dropped` (a safety hold ended it) · `stopped`.
- Into the owner: `UserInstruction{text}` · `Deviation{cmd_id, t, score, threshold, source}` ·
  `StepEvent{cmd_id, t, source, step, detail, checked}` · `PlannedSubtask{cmd_id, t, subtask}` · `DecisionFailure`.
- Owner state: `Keyframe{cmd_id, subtask, capture, evidence, checked}` · `Hold{operation, since, interrupted}` ·
  `Turn` (dialogue line) · out: `Utterance{template, objects}` (`ACK`, `UNANSWERED`, or an operation's template).
- `ObservationSnapshot`: capture, revision, `instruction`, desired command and operation, `active_since`, keyframes,
  `trigger`, `review`, `deviation`, `step_event`, `planned`, `hold`, `dialogue`; `latest_utterance` = newest user line.
- `DecisionProposal`: request id, revision, `base_active_cmd`, operation, scores, preconditions, `expires_t`,
  evidence, `task` (new task text or None). `Precondition`, `Advice` are defined, not used.

## Execution runtime (`control/execution.py`, `ExecutionRuntime`): the authority on what is active
- Modes IDLE · RUNNING · TRANSITION · HOLD · STOPPED; one lock (owner, runner and loop threads call it).
- `accept`: refused (`rejected`) on another epoch, STOPPED, a revoked id, an id not newer, a lease already over. A newer
  goal supersedes a pending one (`revoked`). A holding operation is active at once (`activated`, hold).
- A moving command is pending until its first action is applied (`activated`); `immediate` keeps the old motion until
  then (blended), `pause_hold` (or not RUNNING) holds. A pending goal whose lease ends first is `expired`.
- `renew` extends the lease and ends review; `review(cmd, until_t)` keeps motion, freezes irreversible switches, caps
  the lease; `revoke` drops the command's queued actions at once (a queued gripper release never happens).
- A lease ends a command in any mode (running, held, starved) → `expired` + hold.
- `submit` → `chunk_rejected` on: epoch, STOPPED, revoked, neither active nor pending, lease over, `dt` ≠ control
  period, wrong shape, holding command, active paused for a transition, a hold other than "starved", entirely past.
- Elapsed actions are skipped, never replayed; a chunk replaces its command's queue from its first due time; an old
  command's chunk cannot delay a scheduled takeover.
- Tick order: lease expiry → skip elapsed → activate pending → clamp the incoming action (`violations`, upstream only)
  → blend → decelerate → count starvation → freeze irreversible switches → project onto the bounds (not reported).
- Blend: a new segment keeps 1 − s(k/n) of the jump in continuous dims over `blend_s` (0.2 s), s = quintic smoothstep.
- Decelerating hold: `stop_s` ramps continuous dims to their hold value (library default 0; `realtime.py` 0.3 s).
- `starved_ticks`: ticks where an active moving command had no due action. `queued_from` / `queued_until` feed the
  runner; `last_action()` is the switch state in effect.

## Guard (`control/guard.py`)
- No owner heartbeat for `owner_timeout_s` (2 s) → hold "owner lost", active and pending dropped (`dropped`).
- RUNNING without a valid chunk action for `watchdog_s` (0.5 s) → hold "starved"; the next chunk resumes it.
- Runtime exception → hold, commands dropped, counted in `errors`. `stop()` is final for the epoch.

## Control loop (`control/loop.py`, `ControlLoop`)
- Backend with its own `clock` (any `rojev.robot.Robot`) → `Rate` on that clock; without one (RoboCasa) tick k is due
  at k·dt, wall-paced (`realtime=True`) or back to back; `run_ticks(n)` drives lockstep.
- Lag reported (`lag_s`, `max_lag_s`), never skipped. `TickRecord{tick, before, after, step_ms, lag_s}` to `on_tick`.
- Robot plug-ins: `ActionSpec` per dim (bounds, hold value or `KEEP`, switch threshold, irreversible);
  `RobotBackend` + `joint_target_spec` for any `Robot`; PandaOmron in `executors/robocasa-pi052/embodiment.py`
  ([`robot-interface.md`](robot-interface.md)).

## State owner (`owner/state_owner.py`, `StateOwner`): the single writer of intent
- One thread: event queue + tick every `period_s` (heartbeat, ACK drain, interaction deadline, hold upkeep, planning
  heartbeat every `decide_every_s` unless holding, complete or a lane is busy).
- Instruction: revision + 1, `ACK` said at once, user `Turn` added, decision asked (trigger `instruction`). The task
  text changes only when an accepted proposal carries `task`. A moving command goes under review until the answer is
  due (`review_window_s` = `interaction_timeout_s` with an interaction lane, else `review_s`); `--on-change revoke`
  revokes instead (stop-and-go baseline).
- Proposal refused (logged with why) unless: same epoch and revision, newer than the last applied, made by this owner,
  not planning while an interaction request pends, not a superseded interaction request, about the desired command,
  not a routine planning proposal while a hold stands, not expired, no preconditions.
- Apply: CONTINUE or the same operation renews the lease (`renewed` logged); with new task text a moving command is
  reaffirmed (same operation, new id, always `immediate`). COMPLETE sets `complete` and holds. Anything else = new
  command; `pause_hold` into or out of a hold, the configured transition between moving commands.
- Lanes: triggers `instruction`, `deviation`, `hold` → interaction worker; heartbeat, `step`, `planned` and ended
  commands → planning. While an interaction request pends, planning requests are deferred (the latest is replayed
  after the answer) and planning proposals refused. `interaction=None` = one lane.
- Unanswered change (failure, refusal, deadline): GUARDED_WAIT `UNANSWERED` as a standing hold, said to the user;
  planning never answers a change. After a failure or refusal the lane is asked again every `hold_recheck_s`.
- Late answer (decisions 2026-10-24): after the deadline the answer is still awaited; when it arrives it is applied to
  the situation it was decided on (the placeholder hold is not its base; CONTINUE restores the operation then in force).
- Holds: ABORT / GUARDED_WAIT / ASK answering a change = `Hold` (no heartbeat, routine planning refused, owner keeps
  its lease); released by a moving operation or COMPLETE answering a change; `interrupted` lets a decider resume.
- Keyframes: when an accepted SWITCH_TO another subtask or COMPLETE moves past the step actually running, with the
  decision's observation; evidence = the step event about it, else the decider. For a subtask-less command only a
  checked `StepEvent` is a keyframe. Never for aborts, waits, asks, retries, reaffirms.
- ACKs: newest `seq` only; `activated` sets the active command (clears older deviation, step event, plan, review);
  `rejected` / `expired` / `dropped` of the desired command → fall back to the running one and decide again.

## S2 worker and planning deciders
- `Decider.decide(snapshot) → Verdict{operation, reason, scores, preconditions, evidence, task}` (`system2/decider.py`).
- `DecisionWorker` (`system2/worker.py`): own thread; newest request wins (`replaced` counted); `busy` gates the
  heartbeat and lockstep; proposal expiry `proposal_ttl_s` (2 s); decider error → `DecisionFailure` at once.
- Baselines (`eval/deciders.py`, evaluation only): `ScriptedSequence` (authored steps on a time budget),
  `TruthSequence` (oracle on simulator truth, may go back), `NativePlanner` (one SWITCH_TO without subtask: S1 plans;
  the v1 planning lane with XR-1), `AlwaysContinue` (interaction baseline: the utterance becomes the task).
- `CommitVerifyPlanner` (`system2/planning.py`, `--arm commit`): pins S1's planned subtask until a checked step event,
  else releases after `max_commit_s`. With MJev out (decisions 2026-10-22) only the bound releases it.

## Measured live (RoboCasa)
| Run | Result | Status |
|---|---|---|
| XR-1 + Jev, lockstep with S2 paused, "wait" / "go on" (2026-10-23) | wait at 3.05 s, resume 6.05 s, success 10.5 s; with S2 not paused both answers were late | valid (decisions 2026-10-23) |
| XR-1 real time, F13 | real-time factor 0.806, 651 of 985 ticks starved, no success | valid (E24) |
| Live rerun of late answers, 0.3 s deadline | both late answers applied, not re-decided; stop / go on ×2: 4 / 4 | valid (E30) |
| Stand-in `wiggle` S1, review vs revoke (2026-10-20, decisions "Step 4.1 built") | review: no pause after a change of mind; revoke: a 0.3 s stop | before the render fix; the stand-in ignores images, so the mechanism result stands (rule of E15) |
| π0.5² over the WAN | — | withdrawn: corrupted frames. The bug it exposed (switch dims regressed as numbers, 486 violation ticks) stands, fixed by `to_action_space` (decisions 2026-10-21) |

## Open / not validated
- No completion signal: XR-1's native planner never yields COMPLETE, so a finished task keeps running (run 5, E25).
- Preconditions: proposals carrying any stay refused until perception exists.
- Reaffirm grasp delay (gripper frozen up to the review window) not measured.
- Lockstep resumes ticking when the worker finishes, not when the owner has applied the answer (up to one owner
  period, 50 ms wall, in between).
- Real-time behaviour with XR-1 is supply-bound (F13); hardware needs real-time passes first.
