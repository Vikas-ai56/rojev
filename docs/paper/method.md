# Method: the system as built (2026-10-24)

Mechanisms verified against `rojev/` and `docs/features/`; numbers live in `experiments.md`, cited here by source.

## Terms (defined once)
| Term | Meaning |
|---|---|
| S1 | The policy that moves the robot: camera images + robot state + a task sentence → a *chunk* (a short list of timed actions) |
| XR-1 | Xiaomi-Robotics-1, RoboCasa365 checkpoint (5B), the default S1, served from a remote GPU; π0.5² stays a supported S1 (`--s1 pi052`) |
| S2 | Everything that decides *what* to do; it never sends actions |
| OneJev | OneJev1.6-4B, a vision-language model that answers a typed multiple-choice question with one probability per option |
| Jev | Our interaction layer on OneJev: turns what the user says into one act |
| Owner | The state owner: the only component that writes intent (task text, commands, holds, dialogue) |
| Command | An accepted operation for the robot, with an id, an epoch (restart number) and a lease (expiry) |
| Lockstep | Simulator mode in which simulated time stops while any model computes |

## 1. What the user gets
- **Say a task** at start; **speak at any time**: pause ("wait"), stop, resume ("go on"), switch to another task,
  or say something that keeps the task (a remark, "a bit slower").
- **Get spoken replies**: an acknowledgement at once, a question when the request is unclear, "I am holding still and
  waiting" when it holds.

| Guarantee | Mechanism (section) |
|---|---|
| Every user line is acknowledged before any model answers | owner says `ACK` on receipt (§4) |
| Speaking does not stop the robot by itself | the moving command goes *under review*: keeps moving, gripper frozen (§3, §4) |
| A change nobody understood never moves the robot | no answer by the deadline → hold and say so (§4) |
| A stop stays a stop | holds stand until a later change; no planner can restart the task (§4) |
| The user's words never become S1's prompt | a new task is one of S1's trained sentences, chosen verbatim (§5) |
| Nothing stale moves the robot | late actions skipped, revoked commands' actions dropped, epoch on every message (§3) |
| The gripper changes only on a running, unreviewed command's chunk | irreversible-switch gating (§3) |
| A dead model, network or owner leaves the robot holding still | guard watchdog, owner heartbeat, command leases (§3) |
| Questions are bounded | at most 2 per dialogue, never two in a row (§5) |

Not yet (v1 limits): knowing a task is done; starting a task not on the trained list; checking a task can run in
this scene; answering status questions ("are you done?") (`jev-interaction.md`, `campaign.md`).

## 2. Architecture at a glance
```
user ─▶ Owner ──snapshot──▶ interaction lane: Jev on OneJev   |   planning lane: NativePlanner (S1 plans steps)
          ▲ ◀─────────────── proposals ──────────────────────────────────────┘
          │ ACKs   │ commands (accept · renew · review · revoke)
          │        ▼
       Execution runtime ◀─ timed chunks ─ S1 runner ◀─ XR-1 / π0.5² (remote GPU)
          │
       Guard → control loop (20 Hz) → robot ─ newest frame ─▶ S1 runner, owner snapshots
```
- S2 reaches the robot only through owner commands; S1 never changes the task; imports point one way
  (protocol → robot → control → system1 / system2 → owner → eval, `tests/test_architecture.py`).

## 3. Execution runtime and guard: what keeps it safe
**Runtime** (`control/execution.py`): the only authority on what is physically active.
- Proposal ≠ command ≠ active: a moving command is *pending* until its first action is applied (ACK `activated`);
  wait / ask / abort hold at once. Why: the owner's record matches the robot, never a guess.
- Refuses stale input: wrong epoch, revoked or older command, lapsed lease, wrong control period or shape. Revoke
  drops a command's queued actions at once (a queued release never happens).
- Chunks are placed at their observation's time; actions already due are skipped, never replayed; a new chunk replaces
  the queue from its first due time. Why: a slow model costs its latency, never stale motion.
- Review: after the user speaks the command keeps reversible motion, irreversible switches freeze, its lease shrinks
  to the answer deadline. Why: no needless stop, no grasp or release on a goal that may be changing. Outside review,
  irreversible switches (gripper) still change only on a chunk action of a running command.
- New segments blend over 0.2 s; holds decelerate. Incoming out-of-bound actions are clamped and reported upstream; the
  runtime's own blends are projected silently (decision 2026-10-21: blends had been misreported as faults).

**Guard** (`control/guard.py`), inside the control loop, works if every thread above it is dead:
- owner heartbeat older than 2 s → hold, commands dropped (`dropped` ACK: the owner must re-issue);
- running without a valid action for 0.5 s → hold, command kept (resumes with its next chunk);
- runtime exception → hold, commands dropped; `stop()` is final for the epoch.

**Execution monitor** (opt-in): commanded vs measured hand motion, conformal threshold → `Deviation` → owner + an
immediate S1 refill. Contact-blind: a deliberate press and a blocked arm look alike (decision 2026-10-21).

## 4. State owner: single writer of intent (`owner/state_owner.py`)
- One thread, one event queue, a 50 ms tick; never waits for a model. Commands become active only on the runtime's ACK.
- User line: revision + 1, dialogue turn, `ACK` said, moving command under review, interaction lane asked. The task
  text is unchanged until an accepted decision carries new text (decision 2026-10-22: "wait" had become S1's task).
- Two lanes: changes (user line, deviation, hold recheck) → interaction; heartbeat and step events → planning.
  While a change is pending, planning is deferred and its proposals refused. Why: a routine decision never overrides
  the answer to what the user just said.
- Proposal refusal chain: epoch, revision, newer than the last applied, a request this owner made, lane priority,
  superseded, wrong base command, standing hold, expired (2 s), unchecked preconditions. Every refusal is logged.
- Holds: abort / wait / ask answering a change become a standing hold that remembers the interrupted command; no
  heartbeat meanwhile; released only by a moving answer to a later change (decision 2026-10-22: a planner restarted a
  task the user had stopped).
- Unanswered change (failure, refusal, 1.0 s deadline): hold `UNANSWERED`, said; planning never answers a change
  (decision 2026-10-22: an unread "stop" could move the robot).
- Late answer: awaited, then applied to the situation it was decided on; "continue" re-issues the operation in force
  then. Why: OneJev took 1.1–1.5 s, so every change cost two decisions (decision 2026-10-24).

## 5. Jev v1: two typed questions (`system2/jev/`)
Stateless: everything comes from the owner's snapshot. Per change:
1. Legal acts from a situation table, generated in code, one act per distinct effect:

   | Situation | Acts |
   |---|---|
   | moving | continue · revise · wait · ask · abort |
   | idle | continue · revise · ask · abort |
   | held | resume (if a command was interrupted) · continue (= keep holding) · revise · ask · abort |
   | motion deviation | continue · wait |

   No retry: XR-1 takes only a task sentence, so a retry equals continue (decision 2026-10-24).
2. **Question one (act)**: one OneJev Choice; inputs = every camera as its own image + task, what the user just said,
   current step, steps done, dialogue, hold. Options are neutral ids worded as what the user wants (live 2026-10-22:
   an option named "continue" captured "go on" while it meant "keep holding").
3. **Effect-based hesitation**: probabilities summed per effect (move · hold · switch); top-two effect margin below
   the threshold → ask (if allowed) else wait. Why: "stop" split 0.49 abort / 0.41 wait and asked although both hold
   (decision 2026-10-24).
4. **Question two (task)**, only for revise: one Choice over S1's trained task sentences + `none`; the chosen sentence,
   verbatim, becomes the task text; `none` or a low margin → ask / wait. Why: remarks, typos and "make it fast pls"
   had become XR-1's prompt (decision 2026-10-24).
5. **Calibration**, one per question: temperature by negative log-likelihood, abstain margin = smallest whose decided
   cases reach a target accuracy; written only if held-out runs reach that target. Today none ships: T = 1, margin
   0.2 are declared constants (the 2026-10-23 fit was withdrawn: one scene, no held-out set).
- Latency: one OneJev decision median 604 ms on a kept connection, 75–140 ms of it on the server (decision 2026-10-24).
- Measured (campaign, before step 3 was added): 82 user lines, 55 right, 2 unsafe, 24 safe misses, 1 unanswered by
  design; never revised on unlisted, gibberish or negated lines (`jev-interaction.md`).

## 6. S1 supply: chunks at observation time (`system1/runner.py`, `commit.py`)
- Latency L = observation age when its chunk is submitted, the largest of the last 20 requests.
- Request when less than max(refill, L + one tick) is queued; commit at least ⌈2L/dt⌉ + 1 actions. Derivation: with
  one request in flight the robot runs gap-free only if lead ≥ L and the commitment covers 2L (decision 2026-10-21:
  a configured constant was unsafe on RoboCasa).
- Deadline = one chunk's duration in real time (an older answer lies wholly in the past; a stall had starved a run
  for 105 s); none in lockstep, where it only abandoned good answers.
- Commitment: whole chunk (default); up to the first slow point (PACE, opt-in); agreed prefix of K samples
  (opt-in, not validated on a real policy).
- Continuation: the actions that run during inference are sent only to a policy that accepts them (π0.5² with
  real-time chunking); XR-1 does not.
- Model specifics stay in the executor (XR-1's 4-frame history, the switch mapping to the robot's action space).
- Remote path: one kept connection, WebP-90 frames, request ids + a server replay cache (each request runs at most
  once), token and size checked before the body (`remote-models.md`).
- Measured: XR-1 lockstep parity 18/20 on its published test cases vs 20/20 reported; request p50 511 / p90 731 ms
  against a 0.8 s chunk, so in real time over the WAN supply is the bottleneck (`system1-map.md` §12).

## 7. Timing: lockstep for evaluation, real time for responsiveness
- Real time: the loop ticks on the wall clock; models run on their own threads; late ticks are logged, never skipped.
- Lockstep: simulated time stops while S1 or either S2 lane computes, so model latency costs zero simulated time.
- Why lockstep for sim evaluation: S1 sits on a remote GPU (latency ≈ chunk length) and published benchmarks are
  scored this way; pausing S2 too keeps the owner's deadlines meaningful (smoke test: wait 3.05 s, resume 6.05 s,
  success 10.5 s) (decisions 2026-10-23).
- Cost: lockstep hides latency; real time is reported alongside, and hardware must pass in real time first.
- Run validity: every run samples its camera frames; noise or blank frames invalidate it. A robosuite teardown bug had
  fed S1 noise for a day; all earlier live results were withdrawn (decisions 2026-10-22, 2026-10-23).

## 8. Any embodiment: plug-ins, not pipeline code
| Plug-in | States | Used by |
|---|---|---|
| `ActionSpec` | per action dimension: bounds, hold value (or keep last), switch threshold, irreversible | runtime, guard, commitment |
| `EffectModel` | which measured numbers an action should move, and their resolution | execution monitor |
| Backend | `observe` / `step`, or a `rojev.robot.Robot` with its own clock (`joint_target_spec`) | control loop |

- Built: RoboCasa PandaOmron (Franka on a mobile base, 12-D, 20 Hz) in simulation; MuJoCo SO-101 with joint
  targets, exercised in tests with a stand-in S1 (no learned SO-101 policy).
- Not built: a shared assembly layer (`rojev/app`, spec only); any hardware run.

## 9. Composite tasks with known subtasks
- Protocol: a command may carry a step sentence (`Operation.subtask`); none = S1 sequences its own steps.
- XR-1 (default) receives the whole task sentence and sequences internally; it reports no steps.
- Planning-lane sequencers exist for step sentences: scripted (time budgets), oracle (simulator truth, evaluation
  only), commit-and-verify (pins a self-planning S1's chosen step until a check confirms it). Step events trigger the
  planning lane; keyframes keep steps done (with the observation judged) across revisions.
- Gap: no reliable step-done signal (MJev 0.50 / 0.45, OneJev 0.52 balanced accuracy on 222 labelled frames;
  decision 2026-10-22). Composite sequencing with XR-1: not validated.

## 10. Out of v1
- MJev: out of the loop; at chance as a done-checker (decision 2026-10-22).
- SAM 2.1: measurement only; a bad first mask never recovers (decision 2026-10-22).
- Subtask candidate builder (21k subtasks): not used; Jev answers changes only and chooses from trained sentences.

## 11. v2 direction (`docs/research/2026-10-23-v2-plan.md`)
- Same shell (owner, runtime, guard, lanes); one trained critic: progress / done, and Q(s, ℓ) = P(S1 succeeds on
  sentence ℓ from here), the capability and next-step feasibility check.
- Planning lane supervisor: COMPLETE on done; retry or an atomic step on stall; never moves on without done.
- Removes v1 limits: no completion signal, listed tasks only, no feasibility check, hand-written task list.
- Gated: atomic steering is built only if it beats XR-1 on the whole sentence (G1).
