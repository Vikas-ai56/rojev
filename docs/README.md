# rojev docs

Where each kind of document lives. Code map: [`CODEBASE.md`](CODEBASE.md).

## Start here
| Doc | What it is |
|---|---|
| [`plan/v3-roadmap.md`](plan/v3-roadmap.md) | **Single source of truth now**: architecture, plan amendments, phases P0–P5, tests T1–T10, gates, scorecard, failure → fix table |
| [`plan/v3-v1-spec.md`](plan/v3-v1-spec.md) | v1 build spec: full S1/S2 loop from open-source frozen parts (π0.5², OneJev, MJev; SAM 2 deferred, decisions 2026-10-21) + our runtime; Route A role split; build steps 1–8 |
| [`decisions.md`](decisions.md) | Every durable decision, newest last (decision · why · rejected · risks) |
| [`CODEBASE.md`](CODEBASE.md) | How to read the code |

## Research briefs: `research/` (findings, not decisions; adopted parts move to the roadmap / decisions)
| Doc | What it is |
|---|---|
| [`2026-10-20-jev-role-and-novelty.md`](research/2026-10-20-jev-role-and-novelty.md) | Jev's limits and better roles, MJev's role, prior art, novelty gap, Turing and human timing, full-duplex models, design implications, revision benchmark |

## Plans: `plan/`
| Doc | What it is |
|---|---|
| [`jev_helix_implementation_plan.pdf`](plan/jev_helix_implementation_plan.pdf) | Source plan for v3 (pp.1–18 proposal, pp.19+ reference) |
| [`visual_skill_agent_implementation_plan.md`](plan/visual_skill_agent_implementation_plan.md) (+ pdf) | v1 plan (SO-101 skill agent), superseded |
| [`motion_planner.md`](plan/motion_planner.md) | Notes on a spatial-reasoning / motion layer (v1 era) |
| [`../onejev_learned_policy_plan_v2.md`](onejev_learned_policy_plan_v2.md) | v2 plan (OneJev over a frozen RoboCasa policy), superseded by v3 |

## Features: `features/` (how each built part works and what was verified)
| Doc | Covers |
|---|---|
| [`v3-runtime.md`](features/v3-runtime.md) | v3 runtime core: messages, execution runtime (leases, review, blending, holds), guard, control loop, robot plug-ins, state owner (two lanes, keyframes), S2 worker, baseline deciders, v1 role split; runtime live results incl. π0.5² |
| [`s1-supply-and-commitment.md`](features/s1-supply-and-commitment.md) | S1 runner: chunk supply, measured latency, floor and lead time, commitment policies (fixed / phase / consensus) |
| [`execution-monitor.md`](features/execution-monitor.md) | Kinematic execution monitor: effect models, conformal calibration, `Deviation` → owner + refill, results and limits |
| [`remote-models.md`](features/remote-models.md) | Remote models: transport (HTTP/1.1 keep-alive, HTTP/3), frame codecs, π0.5² / OneJev / MJev services, WAN measurements |
| [`system1-map.md`](features/system1-map.md) | System 1 map (start here for S1): call map, control loop → guard → runtime ← runner ← π0.5², commitment, transport, monitor, messages, timeline, evidence |
| [`system2-map.md`](features/system2-map.md) | System 2 map (start here for S2): owner lanes / holds / dialogue, workers, planning lane, Jev v1, step watcher, messages to S1, timeline, evidence |
| [`scorecard.md`](features/scorecard.md) | T8 revision-benchmark scorecard: inputs, window and answer rule, metric definitions, aggregation, what is not validated |
| [`jev-interaction.md`](features/jev-interaction.md) | Jev v1 interaction lane (legal acts, calibration, abstention, budgets, REVISE), owner holds / dialogue / utterances / step events, MJev end-state checks + step watcher; flags, spoofs, tests, what is not validated |
| [`live-harness.md`](features/live-harness.md) | Live RoboCasa harness: `realtime.py` flags and outputs, 19-case scenario matrix, invariants, declared spoofs, viewer, video, matrix results |
| [`robot-interface.md`](features/robot-interface.md) | Robot interface (`rojev.robot`: MuJoCo SO-101, mock; hardware adapter later): limits, cameras, clocks and time modes |
| [`gpu-setup.md`](features/gpu-setup.md) | One-command GPU box setup (`scripts/gpu_setup.sh`) for S1 (π0.5²) + S2 (OneJev, MJev): steps, component hook interface, pitfalls, security |
| [`observability.md`](features/observability.md) | Phoenix tracing: setup, what spans contain, measurements, gotchas |
| [`v2-levers.md`](features/v2-levers.md) | Registry of every lever that changes output (current value, cost to change) |
| [`v2-pi052-executor.md`](features/v2-pi052-executor.md) | π0.5² set-up, steering point, subtask vocabulary (current); v2 OneJev arm and Kaggle runner (history) |
| [`v2-engineering-notes.md`](features/v2-engineering-notes.md) | v2-era build notes: code reuse and licences, MJev on Apple silicon, platform notes |
| [`phase0-install.md`](features/phase0-install.md) | Install gate of the v1 SO-101 plan: toolchain (current), OneJev 0.8B rows (stale) |

## Paper: `paper/` (only approved material)
| Doc | What it is |
|---|---|
| [`README.md`](paper/README.md) | Outline and status |
| [`experiments.md`](paper/experiments.md) | All experiment numbers (E1–E13+), newest last |
| [`method.md`](paper/method.md), [`related-work.md`](paper/related-work.md), [`appendix-engineering.md`](paper/appendix-engineering.md) | Paper sections |
| [`references.bib`](paper/references.bib), [`citations.md`](paper/citations.md), `sources/` | Curated citations and PDFs |

## Rules
| Doc | What it is |
|---|---|
| [`architecture.md`](architecture.md) | Binding architecture rules for v3 (roles, authority, time, safety, decisions, data, structure; retired v1–v2 rules) |
| [`HardwarePort.md`](HardwarePort.md) | Empty placeholder for the hardware port |

Conventions: docs are concise and point-wise; code changes update `CODEBASE.md`, the relevant `features/` doc and `decisions.md` in the same change.
