# v3 v1 implementation spec (no training)

**v1 = the complete System-1 / System-2 control loop built only from available open-source components, frozen,
pipelined as the Jev × Helix plan prescribes.** In plan terms: WP0 (freeze contracts) + WP1 (S1 baseline, simulator
adapter, event log, fault-injection harness) + WP2 (direct visual Jev, one state owner, text-conditioning baseline),
route A, S2-A (pinned, frozen Jev). Everything from WP3 on needs training and is not v1.
Tracker: `docs/plan/v3-roadmap.md`. Research behind §4–§9: `docs/research/2026-10-20-jev-role-and-novelty.md`.
Package layout: `docs/CODEBASE.md` §2 (enforced by `tests/test_architecture.py`).

**Revision 2026-10-20:** step 4 added (each step-4 feature is opt-in, §14) (review mode, commitment policy, execution monitor, progress keyframes —
borrowed from CheckVLA, CA³C, PACE); Jev v1 = interaction-act layer + verifier (not a next-subtask selector); later
steps renumbered to build on step 4.

## 0. Done when
- In RoboCasa, real-time loop: π0.5² acts continuously; a mid-episode instruction change ("actually, put it in …")
  never freezes the robot: the owner enters review, Jev chooses the interaction act (continue / revise / guarded-wait /
  ask / abort), π0.5² follows the new instruction, irreversible actions wait for that decision; every event traced.
- Execution deviations (blocked arm, slip) are detected by the execution monitor and answered the same way.
- Same tasks run with: native π0.5² · scripted · oracle · Jev; revision benchmark (§11, T8) scored for each.
- Fault-injection suite (roadmap §6 + §10 here) passes on the same loop.

## 1. Components (all open source, frozen)
| Role | Component | Status |
|---|---|---|
| S1 | **π0.5²** `lerobot/pi052_robocasa`, text-conditioned (route A); empty subtask = its own planner | `executors/robocasa-pi052/serve.py` (planner mode untested until GPU) |
| S1 smoothness | RTC (LeRobot `policies/rtc`; π0.5² lacks the hook → port, inference-time only) | step 7 |
| S1 speed | FlashRT FP8 MLP option (`policies/pi052/flashrt_fp8.py`) | step 7 |
| S1 commitment | Commitment policy: fixed → PACE-style phase boundaries → K-sample action consensus (CA³C-lite) | step 4 |
| S1 execution check | Kinematic expected-vs-observed monitor + conformal threshold (CheckVLA-lite) | step 4 |
| S2 planner (open-ended) | π0.5²'s own planner (proposes subtask text); a generative VLM later if needed | step 6 |
| S2 Jev | **OneJev 1.6 4B**: interaction acts + verifier checks (typed, calibrated) | step 6 |
| S2 grounding | **MJev-Spatial-4B** points → tracker; fact checks (holding, inside, open) | step 6 |
| Object identity | **SAM 2** tracker; compared MJev-point → SAM 2 vs Grounded SAM 2 (detector → SAM 2) | step 6 |
| Simulator | RoboCasa (20 Hz, 3 cameras, PandaOmron) | `executors/robocasa-pi052/realtime.py` |
| Baselines | scripted / oracle / native deciders | `rojev/eval/deciders.py` |
| Runtime (ours) | protocol, control, S1 runner, S2 worker, owner | steps 1–3 built |
Not v1: SO-101 (needs S1 fine-tuning); backend + realtime clock ready.

## 2. Files
Built (steps 1–3): `rojev/protocol/` · `rojev/robot/action_spec.py` · `rojev/control/` (execution, guard, loop,
robot_backend) · `rojev/system1/` (policy, wire, remote, runner) · `rojev/system2/` (decider, worker, models/) ·
`rojev/owner/state_owner.py` · `rojev/eval/deciders.py` · `executors/robocasa-pi052/{realtime,embodiment}.py`.
Step 4: `rojev/control/execution.py` (review, decelerating hold) · `rojev/system1/commit.py` · `rojev/system1/monitor.py`
· `rojev/protocol/decisions.py` (`Deviation`, `Postcondition`) · `rojev/owner/state_owner.py` (review, two lanes,
keyframes) · `rojev/robot/effect.py` (`EffectModel`) · conformal fit in `system1/monitor.py` · robot plug-ins add an
effect model
(`executors/robocasa-pi052/embodiment.py`).
Step 5: `rojev/eval/scenarios.py`, `tests/test_faults.py`.
Step 6: `rojev/system2/jev/` (interaction acts, verifier checks, skill templates → supported subtasks),
`rojev/system2/perception/` (tracker, grounding).
Step 7: `executors/robocasa-pi052/serve.py` (RTC prefix, FP8, K samples batched).
Step 8: `rojev/eval/scorecard.py`, revision benchmark runs.
`rojev/legacy_v2/` deleted after T8.

## 3. Messages (`rojev/protocol/`; every message carries `epoch`, times on the robot clock)
Built:
- `Operation{kind: CONTINUE|SWITCH_TO|RETRY|ABORT|GUARDED_WAIT|ASK|OBSERVE|COMPLETE, subtask (None = S1's own planner),
  target, dest, template, objects}`
- `AcceptedGoal{epoch, cmd_id, instruction_rev, operation, transition: pause_hold|immediate, expires_t, instruction}`
- `Revoke{epoch, cmd_id, reason}` · `TimedChunk{epoch, cmd_id, obs_id, t_start, dt, actions, timings_ms}`
- `ExecutionAck{epoch, cmd_id, seq, status: activated|revoked|rejected|chunk_rejected|expired|dropped|stopped, t,
  applied_index, detail}` (`dropped`: a safety hold ended the command, 2026-10-22)
- `UserInstruction{text}` · `ObservationSnapshot{epoch, request_id, now, capture, instruction_rev, instruction,
  active_cmd, active_operation, active_since, keyframes, trigger, review, deviation}`
- `DecisionProposal{epoch, request_id, obs_id, t_capture, instruction_rev, base_active_cmd, candidates_rev, operation,
  scores, preconditions, expires_t, evidence}` · `Precondition{kind, object_id, tol}` · `Advice{…}`
Step 4 adds:
- ✓ `Deviation{epoch, cmd_id, t, score, threshold, source: kinematic}` — execution monitor → owner (4.3).
- `Postcondition{kind, object_id, tol}` on proposals — what must be true after the step (planner-stated, B-direction).
- ✓ Snapshot gains `review: bool`, `deviation`, `keyframes` (replacing `history`); `trigger` ∈ {heartbeat,
  instruction, deviation, cmd … expired / rejected}.
- ✓ `Keyframe{cmd_id, subtask, instruction_rev, capture, evidence}` · `DecisionFailure{epoch, request_id, error}` (4.4).
Step 6 adds (✓ 2026-10-22):
- `StepEvent{epoch, cmd_id, t, source, detail}` — step-completion evidence → owner (trigger `step`).
- `Turn{t, speaker: user|robot, instruction_rev, text, template, objects}` · `Utterance{epoch, t, template, objects,
  cmd_id}` (owner → executor; `ACK` on every instruction) · `Hold{operation, since, interrupted}`.
- Snapshot gains `step_event`, `hold`, `dialogue` (last `dialogue_turns` turns).

## 4. State owner (`rojev/owner/state_owner.py`)
Only writer of instruction revision, desired command, active record (from ACKs), progress; never blocks on inference.
- **Instruction change (step 4, replaces revoke):** `instruction_rev += 1`; the utterance joins the dialogue (the task
  text changes only when an accepted decision adopts new text, e.g. REVISE); the moving command enters **review**
  (lease until the interaction deadline, irreversible switches frozen); ask the **interaction lane**.
  - CONTINUE under the new revision → **reaffirm**: same operation as a new command carrying the new instruction,
    `immediate` (S1 re-conditioned, no pause).
  - SWITCH_TO → new command, `immediate` (blended handover).
  - GUARDED_WAIT / ASK / ABORT → hold now.
  - No decision within `review_s` → lease expires → decelerating hold → decision requested.
- **Deviation (step 4):** to the interaction lane (continue / retry / revise / ask); S1 refills at once.
- **Proposal:** accept iff same epoch and revision, newer request, about the current desired command, not expired,
  not a routine planning proposal while a hold stands, preconditions hold (checkable from step 6; refused before);
  else logged.
- **Holds (✓ 2026-10-22):** ABORT / GUARDED_WAIT / ASK answering a change = standing `Hold` until a later change
  resolves it (moving op or COMPLETE): no heartbeat, routine planning refused, hold command kept leased,
  `hold_recheck_s` → interaction lane again (trigger `hold`, Jev's wait budget).
- **Dialogue + utterances (✓ 2026-10-22):** instruction → `ACK` utterance at once + user turn; templated operation →
  utterance + robot turn; snapshots carry the dialogue.
- **Two lanes (✓ 4.4):** interaction lane (instruction / deviation triggers) and planning lane (everything else);
  separate S2 workers, one owner. Priority: while an interaction request is pending, planning requests are deferred
  (the latest is sent once the interaction decision is applied) and planning proposals refused; released by its
  proposal, `DecisionFailure` or `interaction_timeout_s`; on failure, timeout or refusal the robot holds (GUARDED_WAIT
  `UNANSWERED`, said) and the interaction lane is asked again on the hold recheck; the planning lane never answers a
  change (2026-10-22). No interaction worker = one lane.
- **Progress keyframes (✓ 4.4):** when an accepted proposal moves past a running step (SWITCH_TO another subtask, or
  COMPLETE) the step becomes a `Keyframe` with the observation the decision used; snapshots carry all keyframes,
  kept across repairs and revisions. Evidence = the decider's judgement until postconditions (step 6). Always on (the
  owner's progress record), not a flag.
- ACK: active only on `activated` with newer `seq`; rejected / expired / dropped → fall back to the running command,
  ask again; `chunk_rejected` → logged only. Reaffirm is always `immediate`.
- Step done: postconditions hold (MJev / sensors; OneJev for ambiguous ones, step 6); sim truth = evaluation only.
- **Step-completion events (step 6; ✓ 2026-10-22):** the planning lane is asked when a step may be done, not on a
  timer. Sources: end-state checks (`StepWatcher`, MJev / sensors) and π0.5²'s own planner switching subtask
  (`ActionChunk.info` → `S1Runner.on_step`; in native mode the event is the keyframe). Phase boundaries are not a
  source (a slowdown is not completion). The heartbeat stays for liveness and lease renewal: 0.8 × lease when events
  cover progress, else 1 s.
- Heartbeat to the guard every tick; restart → new epoch, new command.

## 5. Execution runtime (`rojev/control/execution.py`)
- Modes `IDLE · RUNNING · TRANSITION · HOLD · STOPPED`; one action per tick (20 Hz in RoboCasa).
- Intake: current epoch, active/pending command, unexpired lease, chunk `dt` = control period; elapsed actions
  skipped; a new chunk replaces the queue from its first due time (the old queue up to then = hard prefix).
- Revoke drops a command's queued actions at once; `immediate` keeps the old command until the new chunk is due;
  `pause_hold` holds now.
- **Review (step 4):** the command keeps running; irreversible switches keep their value until renew / reaffirm / new
  command; lease shortened.
- **Decelerating hold (step 4):** on any hold after motion, continuous dimensions with a numeric hold value ramp to it
  over `stop_s` (KEEP dimensions repeat the last value as before); `Guard.stop` uses the same ramp.
- Irreversible switches change only while running an active command's chunk outside review.
- ACK `seq`; per-tick log.

## 6. Guard (`rojev/control/guard.py`)
Unchanged: bounds on every action; starvation watchdog; owner heartbeat; stop; survives runtime errors.

## 7. S1 path (`rojev/system1/`)
- `ControlLoop` on the backend's clock; RoboCasa wall-paced at 20 Hz (built).
- `S1Runner` (built): asks the policy for the target command with its instruction; places chunks at observation time.
- **Commitment policy (✓ step 4.2, `commit.py`):** how much of each chunk is committed before S1 is asked again.
  `FixedCommit` (whole chunk) → `PhaseCommit` (through the first low-speed point, PACE) → `ConsensusCommit` (K samples
  from one observation; commit while a majority moves like the committed sample; that sample = closest to the mean of
  the largest group agreeing on every irreversible switch). Disputed switch at step k ≥ floor → stop before it;
  inside the floor → majority decides. The floor is not a parameter: `S1Runner` measures request latency L (largest of
  the recent ones) and passes ⌈2L/dt⌉ + 1, and requests when less than max(`refill_s`, L + dt) is queued; starvation is
  counted by the runtime (`starved_ticks`). K samples batched in the service from step 7 (until then K× latency, which
  the measured floor absorbs if the chunk is long enough).
- **Execution monitor (✓ step 4.3, `monitor.py`):** robot plug-in `rojev.robot.EffectModel` (tracked quantities +
  commanded change + resolution); per tick, measured change over a window vs a linear map of this and the previous
  window's commands (lag-aware) fitted on nominal runs; per-axis residual scale ≥ resolution → score; split-conformal
  threshold, split by run (α per segment of a new run, reported as false alarms / min); command directions never
  exercised in calibration are not scored; one `Deviation` per excursion → owner (decision with trigger
  `deviation`) + `S1Runner.refill_now`. Calibration lives in the monitor module (`fit_monitor`,
  `executors/robocasa-pi052/calibrate_monitor.py`), not `eval/calibration.py`. Visual world-model version (CheckVLA)
  later.
- π0.5² service: planner mode (built); RTC prefix, FP8, K samples (step 7).

## 8. Object identity (`rojev/system2/perception/`, step 6)
- Route 1: MJev points → SAM 2 masks → `track_id`s. Route 2: Grounded SAM 2 (detector → SAM 2). Compare: correct
  object, tracking after occlusion, re-grounding rate, latency; MJev used where it wins (ambiguous spatial grounding).
- Lost / occluded past `track_expiry_s` / ambiguous → `unknown` → Jev observes / waits / asks.

## 9. Jev v1 (`rojev/system2/jev/`, step 6; interaction acts ✓ 2026-10-22, calibration fit + live OneJev pending GPU)
- **Interaction acts** (interaction lane): on an instruction change or a deviation, one OneJev Choice over
  {CONTINUE, REVISE, GUARDED_WAIT, ASK, ABORT} from new instruction + current step + progress keyframes + images.
  Calibrated (temperature refit); low margin → GUARDED_WAIT / ASK. Budgets: `max_asks`, `max_wait_s`; no repeated
  question without new evidence.
- **Planner for REVISE / next step** (planning lane): π0.5²'s own planner proposes subtask text; skill templates map it
  to supported subtasks + objects (tracks); the proposal carries pre- and postconditions.
- **Built (2026-10-22):** candidates in code per situation (`acts.candidates`); `Calibration` (T, margin; declared
  T = 1, margin 0.2 until fitted); abstain → ask (budget) else wait; `max_asks`, no repeat without a user turn;
  `min_dwell_s` for deviations; `hold` trigger → report once (`waiting_for_user`) then keep holding (D7); REVISE via
  `Reviser` (default `S1Replans` = SWITCH_TO(None)). End-state checks: `rojev/system2/verify` (one generic MJev
  question per step, `StepWatcher` → `StepEvent`).
- **Verifier checks:** postconditions and preconditions (MJev / sensors for concrete facts, OneJev for ambiguous ones);
  step done only when postconditions hold; a postcondition turning true posts a step-completion event to the owner
  (planning lane) and becomes the keyframe's evidence.
- **Acknowledge act:** a fast verbal or visible acknowledgement (≤ ~200 ms target) while REVISE is planned.
- ASK rendered from a template; the answer returns as `UserInstruction`.

## 10. Scenarios (`rojev/eval/scenarios.py`, step 5)
Instruction change at approach / grasp / carry / just before release; review timeout; ambiguous instruction; harmless
speech; queued-release race; restart; ACK drop / duplicate / delay; late or wrong advice; advice storm; object
displaced; blocked arm (deviation); slip; occlusion; duplicate objects; Jev delay; S1 delay. Each = a test (stand-ins)
and a live flag in `realtime.py`.

## 11. Tests → roadmap
| Roadmap | Check | Pass |
|---|---|---|
| T1 | unit tests + live RoboCasa run | full episode, full trace, lag reported |
| T1.11–T1.14 | review, commitment, monitor, keyframes unit + live tests | no pause on reaffirm; no irreversible change in review; smooth hold; commitment never starves above its floor and rejects a minority gripper decision; deviation detected at the calibrated false-alarm rate |
| T2 | `tests/test_faults.py` + live scenarios | no revoked/stale action applied; no gripper change in review or hold; active never backwards |
| T3 | fixed E13 | image-gain with CI |
| T4 | candidate recall: templates vs old filter | recall reported |
| T5 | swap / erase / contradict π0.5²'s subtask text | behaviour change measured |
| T6 | MJev evidence vs sim truth; MJev vs Grounded SAM 2 grounding | accuracy, tracking, latency |
| T7 | Jev delay injection | success and reaction latency vs delay |
| T8a | **commitment ablation with π0.5²** (needs GPU): fixed · phase · consensus K=3/5 · consensus+phase on RoboCasa tasks with grasps | success, starved ticks, committed length, disputed-switch rate, wrong grasps, extra inference cost |
| T8 | **revision benchmark**: injections at pre-grasp / contact / transport / pre-commit; goal swap, object swap, stop, ambiguous, harmless speech; arms native · scripted · oracle · Jev; baselines re-prompted π0.5², Hi Robot-style re-plan, SwitchVLA-style head, halt-and-restart, stop-and-go | success, wrong-goal commitments, reaction latency, wasted motion, act ECE, ask precision |
| T9 | Jev errors with 9B/27B vs new view | compute vs evidence split |

## 12. Build order (each step runnable; each builds on the previous)
1. ✓ messages + execution runtime + guard
2. ✓ real-time RoboCasa + control loop + S1 runner (stand-in policy live)
3. ✓ state owner + S2 worker + baselines scripted / oracle / native live
4. **✓ review mode + decelerating hold (4.1) → ✓ commitment policy (4.2) → ✓ execution monitor +
   `Deviation` + conformal calibration (4.3) → ✓ owner two lanes + progress keyframes (4.4)** (stand-ins; live with `wiggle`)
5. scenarios + fault tests (T2), including step-4 behaviours
6. perception (tracker; MJev vs Grounded SAM 2) → Jev v1 (interaction acts, verifier checks, templates,
   acknowledge) → Jev arm live
7. π0.5² service: RTC port, FP8, batched K samples → π0.5² live (needs GPU)
8. T3–T9 runs incl. the revision benchmark → `docs/paper/experiments.md`
Parameters (`review_s`, `stop_s`, `refill_s`, K, thresholds, `margin`, budgets, timeouts) in `config/runtime.yaml`,
defaults in `docs/features/v2-levers.md`.

## 13. Not in v1
**v1 = plan Route A** (p. 5): π0.5² picks the next step and moves; Jev only answers changes (continue / revise / wait / ask / abort) and ambiguous checks; MJev checks and grounds. **v2 = Route B, Helix-type:** small fast S1 conditioned on a learned Jev latent (training); promoted only if it beats v1 (plan's rule).
Training (S1-A…E, S2-B…E), latent route, adaptive depth, merged OneJev+MJev, world-model monitor (CheckVLA) and
consequence consensus (CA³C), VLA-Corrector-style learned critic, speculation (T10), SO-101 control, real hardware,
WebSocket transport.

## 14. Plug-and-play features (opt-in; the base loop runs without any of them)
Each feature is an implementation behind an interface, selected by one config field (`config/runtime.yaml`, CLI flags
in `realtime.py`). Off = the documented default behaviour, so every feature doubles as a benchmark ablation.
| Flag | Default | On | Off | Interface / module |
|---|---|---|---|---|
| `speculative_predict` | `False` | K samples per request, commit only the agreed prefix (consensus); later T10 preparation of likely next steps | one sample, fixed / phase commitment | `system1.commit.CommitPolicy` |
| `commit_policy` | `fixed` | `phase` (PACE boundaries) | fixed `refill_s` | `system1.commit.CommitPolicy` |
| `execution_monitor` | `False` | kinematic monitor + conformal threshold → `Deviation` | no deviation events | `system1.monitor.ExecutionMonitor` |
| `on_instruction_change` | `review` | `review`: keep reversible motion, freeze irreversible switches, short lease | `revoke`: stop at once (stop-and-go baseline) | `owner.StateOwner` |
| `stop_s` | `0.3` (app); library `0` | decelerating hold over `stop_s` | `0` = instant hold | `control.ExecutionRuntime` |
| interaction lane | none (one lane) | a second `DecisionWorker` for instruction / deviation triggers, with priority (`--interaction`) | every trigger to the planning decider | `owner.StateOwner(interaction=…)` |
