# v3 roadmap: Jev × Helix (single source of truth)

Status: **testing without training** (decision 2026-10-20). Source plan: `docs/plan/jev_helix_implementation_plan.pdf`
(pp.1–18 = proposal; pp.19+ = reference only). v1 build spec (no training): `docs/plan/v3-v1-spec.md`. Update the checkboxes here as work lands; results go to
`docs/paper/experiments.md` (E14+), decisions to `docs/decisions.md`.

## 1. Architecture in one look
- **S1** (fast): sees cameras, moves the arm. "How do I do the current step?"
- **Jev(S2)** (slow) = OneJev (typed choices). "Is the goal still right? What next? Do I need to ask?" MJev (spatial
  evidence) is out of the v1 loop since 2026-10-22 (failed the done-check gate).
- **State owner**: the only writer of intent. Jev *proposes*; owner *accepts*; command is *active* only after the
  executor's ACK.
- **Executor + local guard**: timing, chunk validity, limits, stop. Works even if everything above hangs.
- Rules: advice is evidence, never authority · proposal ≠ command ≠ active · local stop is independent.
- S1↔Jev interface: **A = text** (baseline, now). B = learned vector (later, only if it beats text + point).

## 2. Amendments to the source plan (from the 2026-10-20 review)
- Messages: restart number (epoch) on **every** message; **Revoke** message; ACK sequence numbers + status query;
  defined `conditioning_id`, `candidates_rev`, advice schema; proposals carry **preconditions** (what must still hold).
- Operations: CONTINUE · SWITCH_TO(skill, target, dest) · RETRY · ABORT · GUARDED_WAIT · ASK · **OBSERVE**.
  `IGNORE_SUPERSEDED` is not a Jev output (owner drops stale advice by metadata; Jev judges only advice content).
- Hold/transition are **controllers**, not labels (section 6).
- Object identity: MJev localises → fast tracker keeps `track_id` alive → lost/expired = "unknown" → Jev looks/waits/asks.
- Candidates (replaces the BM25/transition filter): skill registry × tracked scene objects × argument types × generic
  preconditions + {CONTINUE, WAIT, ABORT, ASK}. Measure recall. RoboCasa: E9b-style templates over π0.5² skills.
- Jev runs on events (instruction change, step done/stalled, lease expiry) + heartbeat; switch needs a margin + minimum
  dwell.
- Change of mind: the owner puts the moving command in **review** at once (keeps reversible motion, freezes
  irreversible switches, short lease); Jev's interaction act decides continue / revise / wait / ask / abort; no answer →
  decelerating hold (spec §4, revision 2026-10-20).
- Jev v1 = interaction-act layer + verifier checks; π0.5²'s own planner proposes subtask text (research brief §2, §8).
- Execution side (borrowed, training-free): commitment policy (PACE, CA³C-lite) + kinematic execution monitor with
  conformal threshold (CheckVLA-lite) + progress keyframes.
- Simulator never pauses for inference (wall-clock mode for timing; virtual-time mode for data).

## 3. Test setups
| Setup | What it is | Tests |
|---|---|---|
| **RoboCasa (v1)** | full S1/S2 loop from open-source frozen parts: XR-1 (S1; π0.5² earlier) + OneJev 4B + our runtime; lockstep evaluation, real time measured | everything in v1 (T1–T8) |
| **Offline** | saved frames/decisions | Jev accuracy, candidate recall (T3, T4, T9) |
| **SO-101 sim (later)** | needs a trained S1 (S1-A); backend + realtime clock ready | after training |
Caveat: π0.5² is Panda/RoboCasa → model results may not transfer to SO-101.

## 4. Phases, tests and gates
### P0 — Foundations (v1 pipeline, RoboCasa; spec `docs/plan/v3-v1-spec.md`)
- [x] T1.1 Real-time sim clock for the SO-101 MuJoCo backend (2026-10-20; reused pattern for RoboCasa)
- [x] T1.2 Message contracts with validation (2026-10-20: commands/chunks/ACKs/advice; snapshot added with the owner)
- [x] T1.3 State owner event loop (never blocks on inference) (2026-10-20: `rojev/owner/state_owner.py` + S2 worker `rojev/system2/worker.py`)
- [x] T1.4 Execution runtime: command queue, Revoke, ACK, applied index, chunk expiry, merge (2026-10-20)
- [x] T1.5 Guard: action limits, watchdog, stop path, hold behaviour (2026-10-20; `docs/features/v3-runtime.md`)
- [x] T1.6 RoboCasa real-time adapter (env on its own 20 Hz thread) + S1Runner (2026-10-20: stand-in policy live; π0.5² live awaits GPU access; service unchanged, chunks wrapped by the runner)
- [ ] T1.7 Arms on the runtime: native (π0.5² own planner), scripted, oracle, Jev (2026-10-20: native / scripted / oracle live with a stand-in policy; Jev = step 5; π0.5² live awaits GPU)
- [ ] T1.8 Tracing: capture, decision, dispatch, ACK, first applied action
- [ ] T1.9 Perception: SAM 2 tracker; MJev-point vs Grounded SAM 2 grounding; skill templates (spec step 6)
- [ ] T1.10 RTC port to π0.5² (inference-time) + FP8 option + batched K samples (spec step 7)
- [x] T1.11 Review mode + decelerating hold (spec step 4) (2026-10-20: live, no pause on a change of mind)
- [x] T1.12 Commitment policy: fixed → phase (PACE) → consensus (K samples) (spec step 4); stand-in + simulated grasp
  tests only. **FLAG:** not validated on a real policy; π0.5² ablation = T8a (spec §11), required before enabling it
  in any benchmark arm
- [x] T1.13 Execution monitor (kinematic, conformal) + `Deviation` → owner + refill-now (spec step 4). **FLAG:**
  calibrated on stand-in runs that move one axis; recalibrate on clean π0.5² runs (all arm axes) before T8
- [ ] T1.15 Step-completion events → planning lane (postcondition watchers, S1 planner subtask switch, phase
  boundary); heartbeat demoted to liveness + lease renewal (spec §4, step 6) (2026-10-22: `StepEvent` + owner +
  `StepWatcher` with MJev end-state checks, π0.5² subtask switch as a source, heartbeat 0.8 × lease with events;
  phase boundaries rejected as a source; MJev thresholds (T6) open)
- [ ] T1.16 Jev v1 interaction lane (spec §9) (2026-10-22: `rojev/system2/jev` built + standing holds, dialogue,
  utterances in the owner; open: labelled calibration set + temperature fit (GPU), live OneJev latency vs
  `interaction_timeout_s`, ASK templates with object ids after perception)
- [x] T1.14 Owner two lanes (interaction / planning) + progress keyframes (spec step 4); keyframe evidence = decider
  judgement until postconditions (step 6)
- Open from the step-4 review (2026-10-21; each must be closed or measured before T8):
  - [ ] Batched K samples in the π0.5² service (step 7); until then speculative mode costs K× latency
  - [ ] Measure `phase` request load (≈ 4× fixed with the stand-in) on π0.5² (T8a)
  - [ ] Set `tolerance`, `smooth`, `slow_fraction` from π0.5² runs (T8a); today tuned constants
  - [ ] Position robots: chunk lacks the current pose, step-0 motion invisible to commitment (inside the floor)
  - [ ] Measure the grasp delay review mode causes (gripper frozen up to `review_s`) (T8)
  - [ ] Monitor: orientation and base motion; recalibrate on clean π0.5² runs covering all arm axes
  - [ ] All step-4 evidence is mechanism-level (stand-ins); performance claims need T8a / T8 on GPU
- **Gate:** π0.5² runs a full episode in real time under the runtime with Jev supervising; full trace.

### P1 — Fault injection (RoboCasa loop, stand-in models + live)
- [ ] T2 Harness injecting every case in section 6; one test per row
- **Gate:** zero invariant violations over thousands of injected schedules; hold/stop behaviour measured.

### P2 — Model checks (offline / RoboCasa), parallel with P0–P1
- [ ] T3 Fixed E13: same option order, scene-swap pairs, destination-aware metric, ≥200 frames; OneJev 4B vs 9B vs
      plain Qwen letter-score vs blank image vs perfect-state text → **does Jev use the image?**
- [ ] T4 Candidate recall: new generator vs old filter
- [ ] T5 S1 sensitivity: swap / erase / contradict π0.5²'s instruction → **does S1 listen?**
- [ ] T6 MJev evidence vs sim truth (held, where, done)
- **Gate:** T5 shows S1 follows instructions (else Jev cannot help yet); T3 image-gain measured with CI.

### P3 — Closed loop (RoboCasa)
- [ ] T7 Delay test: inject Jev delay, plot success vs delay
- [ ] T8 Revision benchmark (spec §11): injections at pre-grasp / contact / transport / pre-commit; goal swap,
      object swap, stop, ambiguous, harmless speech; arms native · scripted · oracle · Jev; external baselines
      (re-prompted π0.5², Hi Robot-style re-plan, SwitchVLA-style head, halt-and-restart, stop-and-go); metrics:
      success, wrong-goal commitments, reaction latency, wasted motion, act ECE, ask precision
- **Gate (kill/narrow Jev):** Jev must beat native and scripted on revision/recovery metrics.

### P4 — Error attribution (offline)
- [ ] T9 For Jev's errors: fixed by 9B/27B (compute) or only by a new view/answer (evidence)?
- **Gate:** decides whether "think deeper" is ever built (else build LOOK/ASK only).

### v2 — Helix-type route B (training; after v1 results)
- [ ] Small fast S1 (visual encoder + proprioception + projected Jev latent + action head) or π0.5² + trained latent path; compare against v1 route A on T8 (plan p. 5, 9)
- [ ] Task choice beyond the given list (v1 limit, `task_choice.CanonicalTask`, decision 2026-10-24): REVISE picks only
      from the trained sentences listed for the run. v2: a new task outside the list → a clean rewrite in S1's
      trained style (new compositions), confirmed with the user or gated by the feasibility head.
- [ ] Task feasibility in the current state (v1 limit): a listed task is chosen even if it cannot run now (door
      already closed, object absent). v2: the critic's Q(s, ℓ) head filters / reranks the list (plan in
      `docs/research/2026-10-23-v2-plan.md` §3–5).
- [ ] Status questions ("are you done?", "what are you doing?"): no act answers them in v1 (accepted 2026-10-24); needs
      an answer act backed by the completion / progress signal.
- [ ] Language: English only in v1 (accepted 2026-10-24). An utterance Jev cannot read should be asked again, never a
      new task; v1 handles this only probabilistically (effect hesitation turned "ruko" into a question).
- [ ] Task list per scene (v1: hand-written `--skills`): shortlist from the 4,740 trained sentences (RoboCasa365
      human300 `meta/tasks.parquet`) by the scene's objects and fixtures, then Q(s, ℓ).

### P5 — Speculation (after T8; no training)
- [ ] T10 Outatime-style preparation: Jev's next decision before the current step ends + π0.5² first chunks for the
      top-k next subtasks (k ≤ 4, covering most of Jev's probability), batched, lower priority than the live chunk;
      results carry obs_id and expire.
  - **Rule:** speculate in computation only. Nothing speculative moves the robot; only owner-accepted commands
    execute; gripper/contact actions need authorisation; no physical "backtrack". User changes → prepare the safe
    pause/hold, not full alternatives.
  - Measure: hit rate, latency saved per switch, GPU time wasted, S1 deadline misses.
  - **Gate (kill):** low hit rate or any S1 deadline slip.

### Later (deferred, needs training) — see section 7.

## 5. Scorecard (every closed-loop run)
- Task success (ordered) · disturbance recovery · false "done" · wrong-goal activation · stale-advice acceptance
- Candidate recall · Jev accuracy · image-gain (real − blank) · ECE/Brier
- Latency p50/p95/p99: event → ACK (≤200 ms), event → guarded response (≤70/≤100 ms), event → new goal (≤1 s)
- Deadline misses · switch reversals · asks/waits per episode · compute per successful episode
- Paired seeds; held-out scenes; bootstrap by episode; thresholds fixed before looking at results.

## 6. Failure cases → fix → test (T2)
| Failure | Fix |
|---|---|
| Queued "release in A" after "actually, B" | Revoke: executor drops all future actions of the old command |
| Old command after restart | Epoch on every message; executor rejects lower epoch |
| ACK lost / duplicated / late | Sequence numbers; owner never moves "active" backwards; status query |
| Waiting while holding | Hold by state: empty → stay; holding → keep grip, hold, lift clear; timeout → set down safely |
| Switch mid-grasp | Reversible (move) vs irreversible (close/open/release); irreversible needs a valid confirmed command |
| Jev flip-flops | Margin + minimum dwell; repeated reversals → WAIT/ASK |
| Endless ask/wait | Budgets per step; no repeat question without new evidence; exhausted → safe abort + alert |
| Object slips | Gripper width + MJev "held?"; retry budget; then abort/ask |
| Hidden / identical objects | Tracker identity + occlusion expiry → "unknown" → look/wait/ask |
| Advice storm | Admission limit; newest per topic; expiry |
| GPU contention | S1 priority/dedicated GPU; drop advice jobs first; log S1 deadline misses |
| Clock mismatch | One reference clock (robot side); measured offsets |
| Partial chunks | Log generated / sent / applied / observed separately |
| "Done" ownership | Jev proposes; owner confirms only on consistent evidence (Jev check + MJev; sim truth in eval only) |
| Late deep result after grasp/drop | Preconditions in the proposal; owner re-checks before accepting |

## 7. Deferred (with reason)
- S1/S2 training (S2-B..E, S1-A..E): after P3 shows where Jev helps. Notes kept: S1 = π0.5 (openpi) + SmolVLA twin;
  S2 = OneJev 4B (+9B check, 27B teacher); stop gradients into Jev; set-valued loss; scene-swap data.
- Latent interface (route B): only if it beats text + point; first a linear-probe gate.
- Adaptive depth / LEV: only if T9 shows compute-fixable errors.
- Merged OneJev+MJev backbone: new model; only if sequential MJev→OneJev limits accuracy/latency.
- Real SO-101 + deployment: needs trained S1, local GPU next to the arm, teleop demos.

## 8. Open questions
- GPU next to the SO-101? · Who records real demos? · Real camera layout? · OneJev base revision vs MJev @851bf6e.
