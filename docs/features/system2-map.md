# System 2 map: how decisions are made and reach System 1 (code as of 2026-10-22)

System 2 decides *what* to do (which step, how to answer a change, whether a step is done); System 1 does it (see
[`system1-map.md`](system1-map.md)). The state owner is the only bridge: S2 proposes an `Operation`; the owner turns it
into a command for the runtime; S1 is conditioned on that command. Details per feature: `jev-interaction.md`,
`v3-runtime.md`.

## 0. Call map
```
realtime.main → executors/robocasa-pi052/realtime.py          builds and wires every part, one thread each
 ├ StateOwner        → rojev/owner/state_owner.py              gatekeeper: lanes, holds, dialogue, commands
 ├ DecisionWorker ×2 → rojev/system2/worker.py                 one decider per lane; stamps proposals
 │  ├ planning       → rojev/eval/deciders.py | harness         next step (NativePlanner = π0.5² plans itself)
 │  └ interaction    → rojev/system2/jev/interaction.py         answers changes
 │       ├ acts.py → chooser.py → models/onejev.py → qev        legal acts, one OneJev Choice
 │       └ calibration.py                                       temperature + margin
 └ StepWatcher       → rojev/system2/verify/watcher.py          is the running step done?
      └ checks.py → models/mjev.py → rojev/transport             MJev, one kept connection
```
Dependencies: protocol, transport, observability ← system2; owner → control + system2; the executor on top. S2 never
imports S1 or control: it reaches the robot only through the owner.

## 1. Entrypoint (`realtime.py`)
- Planning worker: `DecisionWorker(RecordingDecider(make_planner(a)))` (`realtime.py:358`); arms `fixed`, `scripted`,
  `oracle`, `revisable(-truth)`, `native`.
- Interaction worker (`:361`), if `--interaction` is given: `continue` (baseline), `rules` (keyword spoof) or `jev`
  (`make_jev`: `--chooser onejev|keywords`, `--reviser s1|rewrite`, `--max-asks`, `--jev-calibration`).
- `StateOwner(runtime, guard, planning, …, interaction, interaction_timeout_s, say=Speaker.say, hold_recheck_s)`
  (`:364`).
- `StepWatcher(make_verifier(a), lambda: runtime.active, loop.latest, owner.post, …)` (`:368`) with `--verify mjev|truth`.
- User speech: `--say T:TEXT` → `owner.post(UserInstruction)` (`:399`). Robot speech: `Speaker.say` prints and logs
  `utterances.jsonl`.

## 2. Messages
```
user ─UserInstruction─▶ StateOwner ─say(Utterance)─▶ executor (print / TTS)
                          │ ObservationSnapshot              ▲ DecisionProposal | DecisionFailure
                          ├──────────▶ worker[interaction] → JevInteraction ─┤
                          ├──────────▶ worker[planning]    → planner ────────┘
                          │ AcceptedGoal · renew · review · Revoke
                          ▼
                   ExecutionRuntime ◀─TimedChunk─ S1Runner ◀─ActionChunk─ π0.5²
                      │ ExecutionAck (owner drains every tick)   ▲ target(): obs.task = goal.instruction,
                      ▼                                           │ subtask = goal.operation.subtask
                   Guard → ControlLoop → robot ─Capture (latest)─┴──▶ snapshots, watcher
ExecutionMonitor ─Deviation─▶ owner (+ runner.refill_now)
StepWatcher (runtime.active + latest capture) ─StepEvent─▶ owner
```
| Message | From → to | Carries |
|---|---|---|
| `UserInstruction` | user → owner | text |
| `ObservationSnapshot` | owner → worker | capture, instruction + rev, desired cmd + operation, active_since, keyframes, trigger, review, deviation, step_event, hold, dialogue |
| `Verdict` | decider → worker | operation, reason, scores, preconditions, evidence |
| `DecisionProposal` / `DecisionFailure` | worker → owner | the verdict stamped with epoch, request id, rev, base cmd, obs, expiry / the error |
| `Utterance` | owner → executor | `ACK` or a template + objects, cmd_id |
| `StepEvent` | watcher → owner | cmd_id, t, source, detail (P(done), obs_id) |
| `Deviation` | monitor → owner | cmd_id, score, threshold |
| `AcceptedGoal` / renew / review / `Revoke` | owner → runtime | the command S1 then supplies |
| `ExecutionAck` | runtime → owner | what is actually active |

**S2 → S1 handoff:**
1. S2 outputs an `Operation`, never actions.
2. The owner wraps it as `AcceptedGoal{cmd_id, instruction, operation}`; the runtime holds it pending.
3. `S1Runner.step` sees it through `runtime.target()` and calls π0.5² with that instruction and subtask.
4. The `TimedChunk` carries `cmd_id`, so the runtime accepts it only for that command.
5. The first applied action → `activated` → the owner records the command as active.

ABORT, WAIT and ASK activate at once as holds; `target()` is then None, so S1 is no longer asked.

## 3. State owner (`owner/state_owner.py`)
- `handle` (`:152`): routes instruction, deviation, step event, failure or proposal; one thread, one queue, never
  waits for a model.
- `tick` (`:164`), every 50 ms:
  - heartbeat to the guard;
  - drain ACKs;
  - an interaction request past its deadline escalates to planning;
  - a standing hold → `_keep_hold`, and no heartbeat;
  - else a planning heartbeat every `decide_every_s`.
- `request_decision` (`:181`):
  - lane from the trigger: `instruction` / `deviation` / `hold` → interaction; heartbeat / `step` / `cmd N
    rejected|expired|dropped` → planning;
  - planning deferred while interaction is pending (the latest trigger is replayed once that decision is applied);
  - routine planning blocked while a hold stands;
  - freezes the snapshot.
- `_on_instruction` (`:217`): rev + 1; user `Turn`; `ACK` utterance at once; the moving command under review (keeps
  moving, switches frozen, lease 1 s); asks the interaction lane.
- `_on_deviation` (`:236`), `_on_step` (`:246`): only about the active command, else logged stale; ask at once
  (deviation → interaction, step → planning).
- `_on_proposal` (`:266`), refusal chain (first match):
  1. epoch;
  2. revision;
  3. not newer than the last applied request;
  4. not a request this owner made;
  5. interaction pending (refuses planning);
  6. superseded interaction request;
  7. wrong base command;
  8. a hold stands and the proposal isn't answering a change;
  9. expired;
  10. preconditions.

  A refused interaction answer escalates its trigger to planning.
- `_apply` (`:302`):
  - same operation under a newer instruction → **reaffirm**: a new command, always `immediate`;
  - CONTINUE → renew the lease (`renewed` logged);
  - else keyframe → hold update → `_issue`.
- `_update_hold` (`:334`) / `_keep_hold` (`:347`):
  - ABORT, WAIT or ASK answering a change = a standing `Hold{operation, since, interrupted}`;
  - the owner keeps its command leased;
  - every `hold_recheck_s` it asks the interaction lane again (trigger `hold`);
  - released only by a moving operation or COMPLETE answering a change.
- `_record_progress` (`:359`): a keyframe when a running step is moved past; evidence = the step event about it, else
  the decider's reason.
- `_issue` (`:376`): new `AcceptedGoal`; transition = configured between moving commands, `pause_hold` into or out of a
  hold; a templated operation → `Utterance` + robot `Turn`.
- `_on_ack` (`:398`): the only source of "active"; `rejected` / `expired` / `dropped` → fall back to the command still
  running and ask again.

## 4. Decision worker (`system2/worker.py`)
- `request` (`:45`): the newest snapshot replaces an unstarted one (`replaced`).
- `decide` (`:65`): `Verdict` → `DecisionProposal` (epoch, request id, rev, base cmd, obs, expiry 2 s, evidence:
  reason, trigger, decide_ms, decider); an exception → `DecisionFailure`.

## 5. Planning lane (route A, v1)
- With π0.5²: `NativePlanner`: one `SWITCH_TO(None)`, then CONTINUE. π0.5²'s language head picks each subtask on the
  server; the chosen subtask returns in `ActionChunk.info`.
- Baselines (`eval/deciders.py`): `ScriptedSequence` (time budget), `TruthSequence` (oracle), `AlwaysContinue`
  (interaction baseline). Harness stand-in: `RevisableSequence`.

## 6. Jev v1 (`system2/jev/`; stateless, everything comes from the snapshot)
`JevInteraction.decide` (`interaction.py:29`):
1. `hold` trigger: a stop keeps holding; else report `waiting_for_user` once, then keep holding (D7). No model.
2. Dwell: a deviation < `min_dwell_s` (1 s) after the step started → CONTINUE. No model.
3. `candidates` (`acts.py:64`): legal acts only.

   | Situation | Acts |
   |---|---|
   | instruction, moving | continue · revise · wait · ask · abort |
   | instruction, idle | continue · revise · ask · abort |
   | instruction, held | resume · continue (keep holding) · revise · ask · abort (not if already stopped) |
   | deviation | continue · retry · revise · ask |

   `ask` only while `may_ask` (≤ `max_asks` questions, never two in a row).
4. `Chooser.choose` → one OneJev Choice (`chooser.py:50`):
   - state: instruction, trigger, current step, steps done, conversation, hold, motion check;
   - image: cameras side by side.
5. `Calibration.apply` (temperature); top-two margin < `margin` → ask (if allowed) else wait; waiting while held →
   CONTINUE.
6. `operation` (`acts.py:89`):
   - continue → CONTINUE;
   - resume → the interrupted step;
   - retry → RETRY of the running step;
   - revise → `Reviser` (default `S1Replans` = `SWITCH_TO(None)`: π0.5² re-plans under the new instruction);
   - wait → GUARDED_WAIT;
   - ask → ASK(`clarify`);
   - abort → ABORT.

**Calibration:** `fit_temperature` (`calibration.py:36`) is an offline NLL fit on labelled cases; at runtime only
`apply` runs. Today T = 1 and margin 0.2 are declared constants (no labelled set yet).

## 7. Verifier (`system2/verify/`)
- `StepWatcher.check_once` (`watcher.py:54`), every 1 s:
  - takes the runtime's active command (only if it moves and has a subtask);
  - MJev answers "Robot step: '…'. Which is true right now?" (done / possible / blocked; `checks.py`);
  - after 2 consecutive P(done) ≥ 0.7 → one `StepEvent` per command;
  - failures are counted, never posted.
- The owner asks planning (trigger `step`); the event becomes the keyframe's evidence.
- Native mode (subtask None) is not checked.

## 8. Timeline: "wait", then "go on" (designed sequence, unit-tested; not yet run live with OneJev)
| t | What happens |
|---|---|
| 0 | Planning heartbeat → `SWITCH_TO("pick up the steak")` → cmd 1 pending; the runner requests a chunk |
| ~0.65 s | Chunk in, first action applied, `activated`; the owner records cmd 1 active |
| 5.0 | "wait": rev 1, user turn, `ACK` said, cmd 1 under review (moving, gripper frozen, lease 1 s); interaction request |
| 5.0 + d | Jev: {continue, revise, wait, ask, abort} → wait → GUARDED_WAIT cmd 2 (`pause_hold`), active at once; `Hold(interrupted = pick)`; `target()` = None |
| 6–9 | No heartbeats; the owner renews cmd 2's lease; the robot holds |
| 9.0 | "go on": snapshot carries `hold.interrupted = pick` → resume → cmd 3 `SWITCH_TO(pick)`; hold released; S1 supplies cmd 3 |
d = OneJev latency (unmeasured). d > 1 s → the change escalates to planning instead (risk D6).

## 9. Evidence (2026-10-22)
| Suite | Result |
|---|---|
| `test_owner` 39, `test_system2_jev` 15, `test_system2_verify` 5, `test_control_execution` 33, architecture | 94 passed |
| core suite overall | 265 passed, 1 skipped |
| executor `test_harness` (Jev with keyword scores = the rules) | passed (9 executor tests) |
| matrix: `stop-stays`, `jev-wait-resume`, `jev-ask-answer`, `jev-wait-budget` | pass (17/19 overall) |
| live OneJev / MJev in the loop | not run (GPU session next) |

## 10. Proper vs declared, open
- Proper:
  - the owner owns holds, dialogue, utterances, step events and the dropped-command path;
  - acts are generated in code;
  - the decider is stateless;
  - one generic end-state question;
  - S2 reaches S1 only through owner commands.
- Declared tuned constants: T = 1, margin 0.2, P(done) ≥ 0.7 × 2 checks, dwell 1 s, 2 questions, 8 dialogue turns,
  hold recheck.
- Open:
  - **Planning can override a stop:** an escalated "stop" (interaction lane failed) answered by planning can move the
    robot.
  - **Native mode has no step events:** π0.5²'s subtask switch is not yet a `StepEvent`.
  - **Unlocked read:** the watcher reads `runtime.active` without the lock.
  - **`say` runs on the owner's thread** and must stay fast.
  - **ASK has one generic template** until perception supplies object ids.
  - **OneJev latency vs the 1 s deadline is unmeasured.**
