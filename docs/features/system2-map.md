# System 2 map: how decisions are made and reach System 1 (code as of 2026-10-24)

System 2 decides *what* to do (how to answer a change, whether a step is done, which step next where S1 cannot plan);
System 1 does it (see [`system1-map.md`](system1-map.md)). The state owner is the only bridge: S2 proposes an
`Operation`; the owner turns it into a command for the runtime; S1 is conditioned on that command. Details per feature:
`jev-interaction.md`, `live-harness.md`, `v3-runtime.md`.

## 0. Call map
```
realtime.main → executors/robocasa-pi052/realtime.py          builds and wires every part, one thread each
 ├ Console           → realtime.py:385                         --interactive: typed lines → UserInstruction
 ├ StateOwner        → rojev/owner/state_owner.py              gatekeeper: lanes, holds, dialogue, commands, speech
 ├ DecisionWorker ×2 → rojev/system2/worker.py                 one decider per lane; stamps proposals
 │  ├ planning       → rojev/eval/deciders.py | system2/planning.py   NativePlanner (S1 plans) | commit-verify | baselines
 │  └ interaction    → rojev/system2/jev/interaction.py         answers changes
 │       ├ acts.py → chooser.py → models/onejev.py → qev        legal acts, one OneJev Choice (all cameras, one request)
 │       ├ task_choice.py → chooser.py                           revise: which trained task sentence (or none → ask)
 │       └ calibration.py                                       temperature + abstain margin, one per question
 └ StepWatcher       → rojev/system2/verify/watcher.py          is the running step done? (opt-in --verify)
      └ checks.py → OneJevVerifier | MJevVerifier | TruthVerifier
models/sam2.py (RemoteSam2)                                     measurement only (sam_check.py), not in the loop
```
- Dependencies: protocol, transport, observability ← system2; owner → control + system2; the executor on top. S2 never
  imports S1 or control: it reaches the robot only through the owner.
- v1 default stack: XR-1 (S1, plans its own steps from the task text) + Jev over OneJev (interaction) + `NativePlanner`
  (planning). MJev is not loaded (failed the done-check gate, `decisions.md` 2026-10-22).

## 1. Entrypoint (`realtime.py:462`)
- Planning worker: `DecisionWorker(RecordingDecider(make_planner(a)))` (`:514`); `--arm fixed | scripted | oracle |
  revisable(-truth) | native | commit`.
- Interaction worker (`:517`), if `--interaction` is given: `continue` (baseline), `rules` (keyword spoof) or `jev`
  (`make_jev`, `:346`: `--chooser onejev|keywords`, `--reviser canonical|rewrite`, `--skills FILE` (trained task
  sentences; the run's instruction is always one), `--max-asks`, `--jev-calibration` `{act, task}` JSON).
- `StateOwner(runtime, guard, planning, latest, now, instruction, lease_s, decide_every_s, on_change, review_s,
  interaction, interaction_timeout_s, say=Speaker.say, hold_recheck_s)` (`:520`); `instruction` = `--instruction`,
  else the environment's sentence.
- `StepWatcher(make_verifier(a), runtime.active_goal, loop.latest, owner.post, …)` (`:524`) with `--verify
  onejev|mjev|truth`; `make_verifier` (`:369`) wraps it in `RecordingVerifier` (every check logged with the truth).
- User speech: `--say T:TEXT` (scripted) or `--interactive` (`Console`, `typed.jsonl`), both → `owner.post(UserInstruction)`.
  Robot speech: `Speaker.say` prints and logs `utterances.jsonl`.
- **Lockstep** (`--lockstep`, run loop `:588`; the simulator evaluation mode, decisions 2026-10-23 / 25): the owner
  runs on the main thread (`owner.step()`), and the world does not tick while either lane is `busy` (`worker.py:44`;
  cleared only after the answer is posted) or, read after the lanes, `owner.has_events`, so a decision takes zero
  simulated time and is applied before the next tick and the owner's deadlines
  stay meaningful (smoke test: OneJev's 1.1–1.5 s wall had become 1.75–1.85 s simulated against a 1.0 s deadline).
- `--record-interaction DIR`: exact model inputs per question asked (act, task) + `labels.jsonl` to fill (calibration).

## 2. Messages
```
user ─UserInstruction─▶ StateOwner ─say(Utterance)─▶ executor (print / TTS)
                          │ ObservationSnapshot              ▲ DecisionProposal | DecisionFailure
                          ├──────────▶ worker[interaction] → JevInteraction ─┤
                          ├──────────▶ worker[planning]    → planner ────────┘
                          │ AcceptedGoal · renew · review · Revoke
                          ▼
                   ExecutionRuntime ◀─TimedChunk─ S1Runner ◀─ActionChunk─ XR-1 / π0.5²
                      │ ExecutionAck (owner drains every tick)   ▲ target(): obs.task = goal.instruction,
                      ▼                                           │ subtask = goal.operation.subtask
                   Guard → ControlLoop → robot ─Capture (latest)─┴──▶ snapshots, watcher
ExecutionMonitor ─Deviation─▶ owner (+ runner.refill_now)
StepWatcher (runtime.active + latest capture) ─StepEvent─▶ owner
S1Runner (π0.5² planner mode) ─PlannedSubtask / StepEvent claim─▶ owner / watcher
```
| Message | From → to | Carries |
|---|---|---|
| `UserInstruction` | user → owner | text |
| `ObservationSnapshot` | owner → worker | capture, instruction + rev, desired cmd + operation, active_since, keyframes, trigger, review, deviation, step_event, planned, hold, dialogue |
| `Verdict` | decider → worker | operation, reason, scores, preconditions, evidence, task (REVISE only) |
| `DecisionProposal` / `DecisionFailure` | worker → owner | the verdict stamped with epoch, request id, rev, base cmd, obs, expiry / the error |
| `Utterance` | owner → executor | `ACK` or a template + objects, cmd_id |
| `StepEvent` | watcher / runner → owner | cmd_id, t, source, detail (P(done), obs_id) |
| `Deviation` | monitor → owner | cmd_id, score, threshold |
| `AcceptedGoal` / renew / review / `Revoke` | owner → runtime | the command S1 then supplies |
| `ExecutionAck` | runtime → owner | what is actually active |

**S2 → S1 handoff:**
1. S2 outputs an `Operation`, never actions; the task text changes only through an accepted `Verdict.task`.
2. The owner wraps it as `AcceptedGoal{cmd_id, instruction, operation}`; the runtime holds it pending.
3. `S1Runner.step` sees it through `runtime.target()` and calls S1 with that instruction and subtask (XR-1: subtask
   None → the task text is its whole prompt).
4. The `TimedChunk` carries `cmd_id`, so the runtime accepts it only for that command.
5. The first applied action → `activated` → the owner records the command as active.

ABORT, WAIT and ASK activate at once as holds; `target()` is then None, so S1 is no longer asked.

## 3. State owner (`owner/state_owner.py`)
- `handle` (`:175`): routes instruction, deviation, step event, planned subtask, failure or proposal; one thread, one
  queue, never waits for a model.
- `tick` (`:189`), every 50 ms:
  - heartbeat to the guard; drain ACKs;
  - an interaction request past its deadline → `_unanswered(awaiting=True)` (hold, said; never planning; the answer
    is still awaited);
  - a standing hold → `_keep_hold`, and no heartbeat;
  - else a planning heartbeat every `decide_every_s`.
- `request_decision` (`:206`):
  - lane from the trigger: `instruction` / `deviation` / `hold` → interaction; heartbeat / `step` / `planned` /
    `cmd N rejected|expired|dropped` → planning;
  - planning deferred while interaction is pending (the latest trigger is replayed once that decision is applied);
  - routine planning blocked while a hold stands;
  - freezes the snapshot; a new interaction request ends any awaited answer.
- `_on_instruction` (`:250`): rev + 1; user `Turn`; `ACK` utterance at once; the task text is NOT changed; the moving
  command under review until the interaction deadline (`review_window_s`, `:244`); asks the interaction lane.
- `_on_deviation` (`:270`), `_on_step` (`:280`), `_on_planned` (`:292`): only about the active command, else logged
  stale.
- `_on_proposal` (`:316`), refusal chain (first match): epoch → revision → not newer than the last applied request →
  not a request this owner made → interaction pending (refuses planning) → superseded interaction request → wrong base
  command (the owner's own placeholder hold does not count for an awaited answer) → a hold stands and the proposal
  isn't answering a change → expired → preconditions. A refused interaction answer → `_unanswered` (`:401`):
  GUARDED_WAIT `UNANSWERED` as a standing hold, said, decided again on the hold recheck.
- Late answer (fix 2026-10-24): an awaited answer that passes the chain → `_as_decided` (`:392`): applied to the
  situation it was decided on; CONTINUE = re-issue the operation in force then (a plain hold if none), so "keep going"
  resumes the motion the placeholder stopped. Live 2026-10-23 cause: OneJev 1.1–1.5 s vs the 1.0 s deadline; the answer
  was refused (its base was the placeholder) and the change decided again.
- `_apply` (`:360`):
  - `p.task` set and different → it becomes the task text; the same moving operation under new task text →
    **reaffirm**: a new command carrying it, always `immediate`;
  - same operation or CONTINUE → renew the lease; if the operation carries a template and answers an instruction, the
    robot turn is appended and **said again** (`:382`): a repeated question or "waiting" is never silent;
  - else COMPLETE flag → keyframe → hold update → `_issue`.
- `_update_hold` (`:420`) / `_keep_hold` (`:433`): ABORT, WAIT or ASK answering a change = a standing
  `Hold{operation, since, interrupted}`; leased by the owner; every `hold_recheck_s` the interaction lane is asked
  again (trigger `hold`) unless an answer is awaited; released only by a moving operation or COMPLETE answering a
  change.
- `_record_progress` (`:446`): a keyframe when a running step is moved past; evidence = the step event, else the reason.
- `_issue` (`:468`): new `AcceptedGoal`; transition configured between moving commands, `pause_hold` into or out of a
  hold; a templated operation → `_utter` (`:485`) + robot `Turn`.
- `_on_ack` (`:490`): the only source of "active"; `rejected` / `expired` / `dropped` → fall back to the command still
  running and ask again.

## 4. Decision worker (`system2/worker.py`)
- `request` (`:45`): the newest snapshot replaces an unstarted one (`replaced`); `busy` (`:41`) feeds lockstep.
- `decide` (`:65`): `Verdict` → `DecisionProposal` (epoch, request id, rev, base cmd, obs, expiry 2 s, evidence:
  reason, trigger, decide_ms, decider); an exception → `DecisionFailure`.

## 5. Planning lane
- **XR-1 (v1 default): `NativePlanner`** (`eval/deciders.py:84`): one `SWITCH_TO(None)` under the task text, then
  CONTINUE. XR-1 sequences its own steps inside the model and reports none: no `PlannedSubtask`, no step events, no
  completion. The planning lane never says COMPLETE, so S1 keeps acting after success (run 5).
- **π0.5² commit-and-verify** (`--arm commit`, `system2/planning.py:23` `CommitVerifyPlanner`): π0.5²'s planned subtask
  (`snapshot.planned`) is committed as `SWITCH_TO(text)`; a checked step event releases it; `max_commit_s` (25 s)
  releases an unverifiable one. Live: switches 19 → 2, but nothing verified and a wrong first pick pinned.
- Baselines: `ScriptedSequence` (time budget), `TruthSequence` (oracle), `AlwaysContinue`; harness stand-in
  `RevisableSequence`.

## 6. Jev v1 (`system2/jev/`; stateless, everything comes from the snapshot)
`JevInteraction.decide` (`interaction.py:39`):
1. `hold` trigger (not `UNANSWERED`) → `_wait_budget_spent` (`:82`): a stop keeps holding; else report
   `waiting_for_user` once, then keep holding. No model.
2. Dwell: a deviation < `min_dwell_s` (1 s) after the step started → CONTINUE. No model.
3. `candidates` (`acts.py:107`): legal acts from the situation table (`situation_of`, `:88`), one per distinct effect,
   each worded as what the user wants; acts that keep the task name it (`snapshot.instruction`).

   | Situation | Acts |
   |---|---|
   | moving | continue (a comment or a speed / care request keeps the task) · revise · wait · ask · abort |
   | idle | continue · revise · ask · abort |
   | held | resume (names the interrupted task) · continue (keep holding) · revise · ask · abort (not if already stopped) |
   | deviation | continue (motion fine) · wait (motion wrong: hold until the user decides) |

   - no retry (removed 2026-10-24): XR-1 takes only a task sentence, so a retry is the same sentence = continue;
   - revise = `NEW_TASK` (`:41`): only a *complete* new task (what and with which objects);
   - ask = unclear, or a change named without the full new task; offered only while `may_ask` (`:68`: ≤ `max_asks`,
     never two in a row).
4. Question one: `OneJevChooser.choose(snapshot, options, ACT_QUESTION)` (`chooser.py:55`) → one OneJev Choice over
   `onejev.KeptClient`: `situation` (`:26`) = task, what the user just said, trigger, current step, steps done,
   conversation, hold, motion check; every camera its own `<image:k>` in one request.
5. `Calibration.apply` (temperature); probabilities summed per effect (`acts.effect`: move · hold · switch); top-two
   effect margin < `margin` → ask (if allowed) else wait; else the most probable act with the winning effect.
6. Question two, only for revise: `CanonicalTask.revise` (`task_choice.py:42`): one Choice over the trained task
   sentences (`--skills` + the run's instruction) and `none`, own calibration; the chosen sentence verbatim becomes the
   task text. `none` or a low margin → ask (if allowed) else wait. The user's words are evidence only, never S1's
   prompt (`S1Replans`, which sent them raw, removed 2026-10-24).
7. Wait while already held: on an instruction → GUARDED_WAIT `waiting_for_user` (the owner says it); otherwise
   CONTINUE.
8. `operation` (`acts.py:132`): continue → CONTINUE; resume → the interrupted operation; wait → GUARDED_WAIT;
   ask → ASK(`clarify`); abort → ABORT; revise → the `Revision`.

**Calibration** (offline; at runtime only `apply`; one per question, `act` and `task`): `fit_temperature`
(`calibration.py:50`, NLL), `fit_margin` (`:68`, smallest margin whose decided cases reach the target accuracy),
`fit_calibration` (`:83`). `calibrate_jev.py --question act|task --fit DIR… --held-out DIR…` writes a calibration only
if its decided accuracy on the held-out runs reaches the target (the act margin judged over effects, as Jev decides).
The 2026-10-23 fit (T 0.441, margin 0.103, 30 cases, one scene, no held-out set, old act set) is withdrawn; today
T = 1, margin 0.2 (declared).

On a `hold` recheck of an `UNANSWERED` hold, Jev skips the wait-budget rule and decides the original change.

## 7. Verifier (`system2/verify/`; opt-in, off in the v1 default)
- `StepWatcher.check_once` (`watcher.py:96`), every `--verify-every` (1 s): the runtime's active command (moving, with a
  subtask); "Robot step: '…'. Which is true right now?" (done / possible / blocked); 2 consecutive P(done) ≥ 0.7 → one
  `StepEvent` per command; failures counted, never posted.
- `claim` / `confirm` (`:62`, `:66`): an S1-planner step event is checked once before it counts.
- `OneJevVerifier` (`checks.py:64`): all steps in one request; `MJevVerifier` (`:41`) kept for A/B.
- Done-check A/B (222 simulator-labelled pairs): balanced accuracy MJev 0.50 / 0.45, OneJev 0.52 / 0.52: neither is a
  completion signal.
- SAM 2.1 (`models/sam2.py:43`, `services/sam2`): masks IoU 0.64 from correct points, a bad first mask never recovers;
  measurement only.

## 8. Timeline: "wait", then "go on" (XR-1 + Jev, lockstep with S2 paused; smoke test 2026-10-23)
| sim t | What happens |
|---|---|
| 0 | Planning heartbeat → `NativePlanner` → cmd 1 `SWITCH_TO(None)` under the task text; XR-1 supplies |
| < 3.05 | "wait": rev 1, user turn, `ACK` said, cmd 1 under review; interaction request; the world stops while Jev decides |
| 3.05 | Jev: {continue, revise, wait, ask, abort} → wait (p 0.88) → GUARDED_WAIT cmd 2 (`pause_hold`), active at once; `Hold(interrupted = cmd 1's operation)`; `target()` = None |
| … | No heartbeats; the owner renews cmd 2's lease; the robot holds |
| 6.05 | "go on": snapshot carries `hold.interrupted` → resume → cmd 3 = cmd 1's operation; hold released; XR-1 supplies cmd 3 |
| 10.5 | Task success (simulator check, logged only; nothing in S2 knows) |
Real time: if OneJev's answer misses `interaction_timeout_s` (1.0 s) the robot holds (`UNANSWERED`) at the deadline and
the answer is applied when it arrives (since 2026-10-24; confirmed live in E30 with a 0.3 s deadline).

## 9. Evidence (tests re-run 2026-10-24; runs 1–5 from 2026-10-23, before the Jev fixes; E27–E30 after)
| Suite / run | Result |
|---|---|
| core `uv run pytest -q` (owner, Jev, verify, runner, …) | 329 passed, 1 skipped, 2 timing deselected |
| executor tests (harness incl. both questions as cases, held-out calibration, campaign judge, frame check, serve) | 36 passed |
| smoke test (above) | wait 3.05 s, resume 6.05 s, success 10.5 s |
| interactive run 1 (+ replay) | a remark became the task; a repeated question went unsaid (fixed: owner re-says, WAITING reply) |
| run 2 | goals outside XR-1's repertoire → erratic motion |
| runs 3–4 | hesitation between equivalent acts; "close the cabinet door" (a trained task) not taken as a task; typos sent raw; "make it fast pls" still revised |
| run 5 (2026-10-23 calibration, now withdrawn) | success at 36.75 s; then abort / resume of the finished task → the base circles (no completion signal) |
| E27 campaign (`results/campaign/20261024-134629/analysis.md`) | 82 lines: 55 pass, 2 unsafe, 24 safe misses, 1 unanswered by design ([`campaign.md`](campaign.md)) |
| E28 replay / E30 live rerun (`results/campaign/20261024-rerun/`) | effect hesitation: stop / go on ×2 4/4, "ruko" asks, one stop → wait; late answers applied, not re-decided |

## 10. Proper vs declared, open
- Proper:
  - the owner owns holds, dialogue, utterances, step events and the dropped-command path;
  - acts generated in code; the decider is stateless; S2 reaches S1 only through owner commands;
  - REVISE chooses a trained task sentence; the user's words never become S1's prompt;
  - a late answer is applied as decided, never re-decided;
  - a calibration ships only if it holds on held-out runs;
  - lockstep pauses S2 as well as S1 (no latency hidden in one lane only).
- Declared / heuristic: option wording (lives in `acts.py`; belongs in the OneJev chooser adapter), P(done) ≥ 0.7 × 2,
  dwell 1 s, 2 questions, 8 dialogue turns, hold recheck, `max_commit_s` 25 s, T = 1 / margin 0.2 for both questions.
- Open (next build, `decisions.md` 2026-10-23 / 24):
  - **Capability signal**: REVISE now chooses among trained sentences, but the list is given per run (`--skills`); which
    of them can run *in this state* needs the critic's feasibility head;
  - **Completion signal**: nothing in S2 knows a task is done (trained progress / done head, Helix-style);
  - **Outcome-level abstention**: calibration cannot fix semantic misreads (stop → wait); a speed remark read as
    revise now ends in `none` → ask, not in a new task;
  - **Task splits**: a composite task given as a sequence of trained sentences has no sequencer or per-step done
    signal yet;
  - status questions ("are you done?") have no act;
  - preconditions stay refused until perception checks them; ASK has one generic template;
  - OneJev in the loop ~930 ms (E30) vs the 1 s deadline: the server drops idle connections after ~5 s (keep-alive
    on the server, open); late answers are applied meanwhile.
