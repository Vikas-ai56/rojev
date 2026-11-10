# Paper workspace: an interruptible robot from frozen parts

Everything paper-relevant. Engineering detail stays in `docs/features/`; decisions in `docs/decisions.md`.

## Working outline
- **Title (draft):** Talk to it while it works: a System 1 / System 2 robot runtime with typed, calibrated interaction.
- **Research question:** can a robot built from frozen open models (a VLA that acts, a typed-choice VLM that decides)
  take spoken changes mid-task (pause, stop, resume, switch task) safely and promptly, on any embodiment, without
  training?

| # | Claimed contribution | Support today | Still to measure |
|---|---|---|---|
| 1 | A runtime contract: one writer of intent; proposal ≠ command ≠ active; leases, review, frozen irreversible switches, independent guard | built; core tests pass (`system2-map.md` §9) | invariants over injected faults at scale (T2) |
| 2 | S1 supply from measured latency: chunks at observation time, gap-free floor and lead, one-chunk deadline | built; XR-1 parity 18/20 in lockstep | real time with S1 next to the robot |
| 3 | Jev v1: legal acts from a situation table, two typed questions, hesitation between effects, trained-task choice | built; campaign 55/82 right, 2 unsafe (before effect hesitation); live rerun: unsafe "ruko" gone, stop/go-on 4/4 (E30) | re-run campaign; held-out calibrated accuracy |
| 4 | Evaluation for interactive robots: lockstep with S1 and S2 paused vs real time; frame-integrity gate; per-line judge by robot state | built; the gate exposed a bug that voided a day of results | revision benchmark vs baselines (T8) |
| 5 | Embodiment plug-ins (`ActionSpec`, `EffectModel`, backends) and one shared assembly | RoboCasa Franka (sim) with XR-1 and π0.5²; LIBERO Franka through the shared assembly (tests only); MuJoCo SO-101 in tests with a stand-in S1 | GR00T on LIBERO on a GPU (a learned S1 on a second robot) |

**Not claimed:** new motor skills or any training; knowing when a task is done; tasks outside S1's trained sentences;
feasibility of a task in the current scene; gains over baselines (T8 not run); hardware results; real-time XR-1 over
a wide-area network; MJev or SAM 2.1 as working signals.

## Superseded (2026-10-17 outline)
- OneJev choosing every next subtask for a frozen VLA → Jev answers changes only (decisions 2026-10-21 "v1 = plan
  Route A"; next-subtask accuracy 40 % in E13).
- 21k-subtask candidate builder (coverage@K) → acts generated from the situation, new tasks from S1's trained
  sentences (decisions 2026-10-24 "One interaction act per effect", "Jev revises to a trained task sentence").
- MJev evidence before every decision → MJev out of the loop, at chance as a done-checker (decision 2026-10-22).
- π0.5² as the S1 → XR-1 by default, π0.5² supported (decision 2026-10-22; its failure evidence was later withdrawn
  with the frame bug and not re-run).
- Arms A–E as the main study → revision benchmark (T8 scorecard) and the user campaign (decisions 2026-10-22,
  2026-10-24).
- Paused / injected / async timing study → lockstep for sim evaluation, real time alongside (decision 2026-10-23).

## Files
| File | Contents |
|---|---|
| `README.md` | This page: question, claims, status |
| `method.md` | The system as built, user view first, then component by component |
| `experiments.md` | Every experiment with exact setup, numbers and run directories; withdrawn runs flagged |
| `appendix-engineering.md` | Engineering for reliability and reproducibility: S1 contracts, real-time correctness, evaluation integrity, transport, GPU setup, embodiment |
| `failure-cases.md` | Every failure found: symptom, root cause, fix (layer · file), evidence, status; lessons |
| `related-work.md` | Literature positions and verified numbers |
| `citations.md` | Index: key → source → used for |
| `references.bib` | BibTeX for every cited source |
| `sources/` | PDFs of key papers |

## Status (2026-11-10)
| Item | State |
|---|---|
| Runtime core (owner, runtime, guard, loop, S1 runner) | built, tested; since 2026-11-09 planning is asked on events only, the owner keeps commands alive, steps are kept in a task memory (done / undone) |
| XR-1 as S1, lockstep | 18/20 on 4 tasks × 5 seeds vs 20/20 reported |
| XR-1 in real time | supply-bound over the WAN (request p50 511 ms vs 0.8 s chunk); not validated |
| Jev v1 (act + task questions) | built; campaign run once; effect hesitation added after it, replayed on its 82 lines only |
| Jev calibration | gate built; nothing fitted yet (declared T = 1, margin 0.2) |
| Completion / feasibility signal | none (v2 critic, `docs/research/2026-10-23-v2-plan.md`) |
| Composite tasks with known subtasks | protocol, sequencers and a step planner (Embodied-R1.5) built; done checks run when the gripper changes; not validated |
| Shared assembly + LIBERO + GR00T as S1 | built 2026-11-10; tests only, no GPU run |
| Embodied-R1.5 (planner, done checker) | client and offline test built; not run on a GPU |
| Revision benchmark (T8) | scorecard built; no scored live run |
| MJev / SAM 2.1 | out of the loop / measurement only |
| SO-101 | MuJoCo backend + runtime tests; no learned S1 |
