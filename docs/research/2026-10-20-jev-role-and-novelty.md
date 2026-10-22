# Jev's role, prior art and novelty (research brief, 2026-10-20)

Status: research findings, not yet adopted (decisions pending in §9). Sources checked against primary pages unless marked.

## 1. Problem with Jev as the main planner
- Typed decisions only: picks from a list someone else builds; cannot generate plans, questions or open-ended intent.
- As next-subtask selector: 40% (E13), image use unproven, zero-shot on robot data (OneJev-Data has no robotics).
- Unnatural for open-ended requests; a generative model or π0.5²'s own planner fits that job better.

## 2. Where Jev fits
- Small, fixed decisions where speed (~64 ms/question, OneJev README, H200) and calibration matter:
  - **Interaction act on a changed instruction:** CONTINUE / REVISE / GUARDED_WAIT / ASK / ABORT.
  - **Verifier:** step still valid? done? holding X? confident or ask?
  - **Speculation ranker (roadmap T10):** planner proposes next steps; Jev's probabilities pick which 2–4 to prepare; Jev confirms before irreversible actions.
- Source plan agrees: p.1 "interpret the revision; select a supported operation"; p.18 Jev's work = "integrates changed intent and advice with progress"; p.22 contribution = "grounded, timely revision under changing evidence".

## 3. MJev's role
- Weak use (arm D): judging abstract done/possible/blocked from subtask wording.
- Strong use = grounding: point to objects (seed SAM 2 tracker), point for S1 ("which cube"), concrete fact checks (holding, inside, open).
- Cost: ~1.3 s / 3 questions on T4 fp16; pointing = 3 passes; questions cannot share an image pass.
- Split: MJev locates and checks facts · Jev decides how to respond · generative planner plans · runtime keeps motion smooth and safe.

## 4. Prior art (closest)
| Work | Does | Lacks |
|---|---|---|
| Hi Robot (2502.19417) | Interjections any time; generative re-planning (~1 Hz or on speech) | Calibrated response choice, gating, latency metric |
| SwitchVLA (2506.03574) | Typed 3-class mid-motion switch (forward/rollback/advance, CE loss); mid-switch 95–96% vs π0 0% (real robot) | Calibration; ask/pause/abort |
| VITA-E (2510.21817) | Duplex control tokens, two parallel VLAs, halt and restart | Continuous revision; latency numbers |
| ELLSA (2510.16756, ICLR 2026) | Full-duplex vision/speech/action; barge-in = cancel | Interruption metrics |
| ARMS (2609.28429, CoRL 2026) | Always-on async streaming on π0.5 | Changed/cancelled instructions |
| UPS (2602.22474), KnowNo (2307.01928) | Calibrated act / ask | Not during motion |
| Helix (figure.ai/news/helix) | S1 200 Hz / S2 7–9 Hz, latent link | Nothing published on interruptions |
- Correction: BAS-VLA (2609.23650) is not about instruction switching.
- Verdict: "nobody has done mid-task revision" is false.

## 5. Open gap = defensible novelty
1. One calibrated act choice during motion over {continue, revise, guarded-wait, ask, abort} (+ fast acknowledge).
2. Instruction-authority rules (revision ids, revoke, leases, stale-goal suppression) — built (`rojev/owner`, `rojev/control`).
3. Gating irreversible actions while a revision is pending or ambiguous — built (`ExecutionRuntime`).
4. A revision benchmark measuring reaction latency and wrong-goal commitments — none exists.

## 6. Turing and human timing
- Turing 1950: imitation game = indistinguishable from a human, by teleprinter, no body; his sample machine pauses (30 s, wrong sum). No speed or embodiment requirement; much faster than human would fail.
- Embodied: Harnad 1991 Total Turing Test ("should be a robot … indistinguishable … from the way real people do it"); Zador et al. 2023 embodied Turing test. No latency figures.
- ⇒ Goal = human-like responsiveness, not "super fast".
- Targets (conversation research): acknowledge ≤ ~200 ms; > 300 ms lowers acceptance; ≥ 600–700 ms reads as hesitant (Levinson & Torreira 2015; Stivers 2009); full reply ~1 s with filler (Shiwa 2009, abstract only); stop within one control tick (guard). End-of-turn detection by silence costs ~500 ms → act on partial speech.

## 7. Full-duplex dialogue: who decides when to respond
- Inside the generator: Moshi (PAD tokens), Thinking Machines ("no separate dialog management component"; 200 ms micro-turns; 0.40 s FD-bench), GPT-Live, KAME front end (2510.02327, oracle stream from a back-end LLM).
- Separate small classifier: OpenAI Realtime semantic VAD (+ cancel/truncate on barge-in), VAP.
- Risk: a separate typed Jev layer is the "harness" style the frontier is leaving. Defensible for robots: irreversible actions need an auditable calibrated gate; no robot duplex model handles revisions yet; Jev's acts can later be folded in as a model head.
- Corrections: Thinking Machines' post has no "empty slot" wording ("both remain silent"); KAME exists.

## 8. Design implications
- **Review, not revoke:** on an instruction change keep reversible motion, freeze irreversible switches, shorten the lease, ask Jev first; confirmed → no pause; switch → blended handover; no answer → smooth slow-down.
- **Jev v1 = interaction-act layer** (`rojev/system2/jev/`) + generative planner for REVISE details.
- **Acknowledge act ≤ ~200 ms** (speech or visible slow-down) hides re-planning (fast front, slow back).
- **Streaming ASR + early end-of-turn.**
- **Owner:** two lanes (fast interaction lane, planning lane); protocol and control mostly unchanged.
- **Latency:** control loop unaffected; interaction decision ~0.1 s local GPU / ~0.25 s+ US cloud (published OneJev figures, unmeasured); REVISE + a few hundred ms while moving.

## 9. Revision benchmark (proposed T8)
- Inject at: pre-grasp, in contact, transport, just before an irreversible step. Kinds: goal swap, object swap, stop, ambiguous, harmless speech.
- Metrics: success, wrong-goal commitments, reaction latency (speech → first behaviour change; stop → zero velocity), wasted motion, act ECE, necessary vs unnecessary asks.
- Baselines: π0.5² re-prompted · Hi Robot-style re-planning · SwitchVLA-style head · VITA-E halt-and-restart · stop-and-go. Ablations: no calibration, no authority rules, no gating.
- Pending decisions: reframe the paper to "Jev decides how to respond to change"; adopt §8 and this benchmark in the roadmap.

## 10. Execution verification and failure detection (user sources, verified 2026-10-20)
| Work | What it does | Verified numbers |
|---|---|---|
| CheckVLA (2607.26789) | Frozen action-conditioned world model predicts expected effects of committed actions; conformal threshold → rewrite remaining actions with a latency-aware hard prefix; RoboCasa365 | Success 36.1% vs 27.6% periodic replanning; timely recall 77.9% vs 48.6% observation-only (5% false alarms); sim only, no latency reported |
| SAFE (2506.09937, NeurIPS 2025) | Failure probability from the VLA's internal features; trained on success/failure rollouts; unseen tasks; conformal; OpenVLA, π0, π0-FAST | Abstract gives none; code via vla-safe.github.io |
| Code-as-Monitor (2412.04455, CVPR 2025) | VLM writes code checking spatio-temporal constraints in real time (reactive + proactive) | +28.7% success, −31.8% time under severe disturbances |
| VLA-Corrector (2607.01804) | Latent vision monitor compares predicted vs actual feature evolution; truncates stale actions, corrective replanning; event-triggered action horizon | None in abstract. "~40M params", "NeurIPS 2026", code link: UNVERIFIED |
| DoReMi (2307.00329, IROS 2024) | LLM planner also writes constraints; VLM checks violations during execution; recovery | None in abstract |
| Foresight (2606.23085) | Failure detection from action-conditioned world-model latents, trained on task-level outcomes only; functional conformal thresholds | None in abstract |
| Already verified: SayCan (2204.01691), RoboCombine (2610.15425), RTC (2506.07339), training-time RTC (2512.05964) | Proposer–scorer; successor-skill start-state gap; committed-prefix chunking | — |
- Implication for novelty: **execution-side** verification ("did the world do what we expected?") is crowded (CheckVLA, SAFE, CaM, DoReMi, Foresight, VLA-Corrector) → "Jev as verifier" alone is not novel. The gap stays on the **intent side** ("the user changed their mind mid-motion"), §5.
- Implication for design: two kinds of change, two signals into the owner — execution deviation (S1-side monitor: SAFE-style failure head on π0.5² features or CheckVLA-style action-conditioned check) and intent revision (Jev interaction acts). CheckVLA's hard-prefix repair matches our runtime's committed-prefix rule; it runs on RoboCasa365, so it is a direct comparison point.
- Speculation (T10) comes after both signals exist: a prepared step can only be trusted if deviations and revisions are detected.

## 11. Five follow-up directions (user sources, verified 2026-10-20)
| Direction | Source check | Fit for us | When |
|---|---|---|---|
| A. Lightweight execution critic | VLA-Corrector (2607.01804): ~40M MLP corrector (38–42M, App. C.2); π0.5 MetaWorld 48.70 → 64.35 (Table 1), but 58.70 at horizon 50 and 54.32 at full data elsewhere → headline inconsistent; code github.com/ZJU-OmniAI/vla-corrector (checkpoint release not stated); monitor latency not reported | S1-side deviation signal into the owner; Jev decides recovery | Needs training on our rollouts → after v1 |
| B. Planner emits verifiable conditions | DoReMi (IROS 2024), Code-as-Monitor (CVPR 2025) verified (§10) | Maps onto `protocol.Precondition` + new postconditions ("cube_in_gripper"); simple checks by sensors/MJev, ambiguous ones by OneJev; owns "step done" | v1 (no training) |
| C. MJev vs Grounded SAM 2 | Grounded-SAM-2 repo: Grounding DINO / Florence-2 / DINO-X + SAM 2 tracking; live-camera continuous-ID script "not that stable now"; last update 2025-04-20 | Compare MJev-point → SAM 2 vs detector → SAM 2: correct object, tracking after occlusion, re-grounding rate, latency | v1 (T6 / T1.9) |
| D. Commit while futures agree | CA³C (2610.07949): K=3 candidate chunks, action-conditioned world model, commit while imagined futures agree; RoboCasa π0.5 52.9 → 56.2; best fixed horizon h=8/10 gives 92.9 vs 93.5 on LIBERO-Plus (π0.5 only) → small gain over a tuned fixed horizon; no latency reported | Speculation research; commit only on fresh evidence | Later (T10) |
| E. Phase-adaptive chunk length | PACE (2606.00537, NeurIPS 2026): training-free, re-plan at low-speed points of the predicted chunk; RoboTwin 2.0 (50 tasks) 57.8 → 64.2; real 50.7 → 70.4 (abstract/snippets) | `S1Runner` refill boundary: long commits in free motion, short near contact; complements irreversible gating | v1 candidate (no training) |
- Takeaway: B, C, E fit v1 without training; A after v1 (needs rollouts); D stays research (T10).
