# rojev docs

Where each kind of document lives. Code map: [`CODEBASE.md`](CODEBASE.md).

## Start here
| Doc | What it is |
|---|---|
| [`plan/v3-roadmap.md`](plan/v3-roadmap.md) | **Single source of truth now**: architecture, plan amendments, phases P0–P5, tests T1–T10, gates, scorecard, failure → fix table |
| [`plan/v3-v1-spec.md`](plan/v3-v1-spec.md) | v1 build spec: full S1/S2 loop from open-source frozen parts (π0.5², OneJev, MJev; SAM 2 deferred, decisions 2026-10-21) + our runtime; Route A role split; build steps 1–8 |
| [`review/rojev-v1-review.html`](review/rojev-v1-review.html) | One-page v1 review (open in a browser): what, why, S1 / S2 flow, component choices, results, open blockers |
| [`review/rojev-v1-discussion.html`](review/rojev-v1-discussion.html) | v1 follow-up (2026-11-04): goals vs status, missing background reasoning, RynnUnify ideas, real-time plan, other designs, work order |
| [`target-system.md`](target-system.md) | Target design for the real robot: S1 / S2 split, memory, safety kernel, configuration, first-version targets, open decisions, status |
| [`decisions.md`](decisions.md) | Every durable decision, newest last (decision · why · rejected · risks) |
| [`CODEBASE.md`](CODEBASE.md) | How to read the code |

## Model bring-up: `model-bringup/` (read before plugging in any new or retrained model)
| Doc | What it is |
|---|---|
| [`README.md`](model-bringup/README.md) | Why it exists, rules of thumb |
| [`checklist.md`](model-bringup/checklist.md) | Ordered gates: claims and licence, serving, input / output contracts, env parity, observation integrity, runtime, reproduce-then-measure, own trained models |
| [`search-directions.md`](model-bringup/search-directions.md) | Where to look and how to bisect when a model underperforms |
| [`groot.md`](model-bringup/groot.md) | GR00T N1.7 on the SO-101: contract from source, our adapter, open checks |
| [`s2-bakeoff.md`](model-bringup/s2-bakeoff.md) | Embodied-R1.5 as planner and done checker on real SO-101 step cases + ArmNetBench attempts: case sets, run, scoring, pass bars, limits |

## Research briefs: `research/` (findings, not decisions; adopted parts move to the roadmap / decisions)
| Doc | What it is |
|---|---|
| [`2026-11-07-vla-steerability.md`](research/2026-11-07-vla-steerability.md) | Do VLAs follow step text the scene does not imply? Our GR00T probe, LangGap, ReSteer, MolmoAct2 / π0.7; the fix is same-scene multi-task data; probe as the S1 gate |
| [`2026-10-23-v2-plan.md`](research/2026-10-23-v2-plan.md) | v2 for new compositions: XR-1 whole-task + trained critic (done / progress, capability, next-step feasibility) + recovery; week-1 gates, 4-week recipe, π0.7 / Eka VFA / JEPA verified, flip triggers |
| [`2026-10-23-discarded-actions-as-prefix.md`](research/2026-10-23-discarded-actions-as-prefix.md) | VLA-Rolling (arXiv 2610.25043): reuse the discarded chunk tail as the next chunk's denoising start; fit for XR-1 and what to test |
| [`2026-10-22-s1-candidates.md`](research/2026-10-22-s1-candidates.md) | S1 candidates with verified results (π0.5, Xiaomi-Robotics-0/1, MolmoAct2, GR00T): which S1 for RoboCasa, LIBERO, real SO-101 / Franka; licences; flip triggers |
| [`2026-10-22-xr1-integration.md`](research/2026-10-22-xr1-integration.md) | Xiaomi-Robotics-1 RoboCasa365 checkpoint behind our runtime: exact inputs / outputs, env parity, pickle-server risk, history placement, plan |
| [`2026-10-22-mjev-keep-or-drop.md`](research/2026-10-22-mjev-keep-or-drop.md) | MJev-Spatial: what it is, what it has done here, literature on VLM step verification / grounding / tracking, options, A/B and flip triggers (verdict: drop from the v1 loop unless it wins grounding) |
| [`2026-10-22-trackers-vs-sam2.md`](research/2026-10-22-trackers-vs-sam2.md) | Trackers vs SAM 2.1 Tiny (EdgeTAM, EfficientTAM, SAM 3.1, …): accuracy, speed, licence, Mac; recommendation and acceptance gates |
| [`2026-10-22-native-runtime.md`](research/2026-10-22-native-runtime.md) | Python vs C++/Rust: measured per-tick cost and jitter; GC and GIL are the stalls; no rewrite for v1 |
| [`2026-10-20-jev-role-and-novelty.md`](research/2026-10-20-jev-role-and-novelty.md) | Jev's limits and better roles, MJev's role, prior art, novelty gap, Turing and human timing, full-duplex models, design implications, revision benchmark |

## Plans: `plan/`
| Doc | What it is |
|---|---|
| [`jev_helix_implementation_plan.pdf`](plan/jev_helix_implementation_plan.pdf) | Source plan for v3 (pp.1–18 proposal, pp.19+ reference) |
| [`../hardware-port.md`](hardware-port.md) | Moving to a real robot: lockstep is sim-only (kept for now), delay levers in order, what the simulator hides |
| [`../how-to-train.md`](how-to-train.md) | Training directions (recovery, handoffs, memory, delay, interface gradients) + 2026-10-23 addendum: measured delay budget, completion target and loop guard, memory, data integrity, S1 base, corrections |
| [`visual_skill_agent_implementation_plan.md`](plan/visual_skill_agent_implementation_plan.md) (+ pdf) | v1 plan (SO-101 skill agent), superseded |
| [`motion_planner.md`](plan/motion_planner.md) | Notes on a spatial-reasoning / motion layer (v1 era) |
| [`../onejev_learned_policy_plan_v2.md`](onejev_learned_policy_plan_v2.md) | v2 plan (OneJev over a frozen RoboCasa policy), superseded by v3 |

## Features: `features/` (how each built part works and what was verified)
| Doc | Covers |
|---|---|
| [`pipeline.md`](features/pipeline.md) | **Start here for reports**: the whole system on one page: components, data flow, triggers, memory, done-check cascade, liveness, S1 / S2 today, plug-ins, time modes, run records, what is tested and not validated |
| [`app-session.md`](features/app-session.md) | Shared assembly `rojev/app`: embodiment, config, plug-ins, `Session` (lockstep / real time), records, video |
| [`v3-runtime.md`](features/v3-runtime.md) | v3 runtime core: messages, execution runtime (leases, review, blending, holds), guard, control loop, robot plug-ins, state owner (two lanes, event-driven planning, owner-owned lease, task memory), S2 worker, baseline deciders, step planner; runtime live results |
| [`execution-monitor.md`](features/execution-monitor.md) | Kinematic execution monitor: effect models, conformal calibration, `Deviation` → owner + refill, results and limits |
| [`remote-models.md`](features/remote-models.md) | Remote models: transport (HTTP/1.1 keep-alive, HTTP/3), wire v3 and codecs, π0.5² / XR-1 / GR00T / LeRobot-policy / OneJev / MJev / SAM 2 / R1.5 services, `KeptClient`, chat VLM client, WAN measurements |
| [`plan/specs/any-robot/00-index.md`](plan/specs/any-robot/00-index.md) | SPEC suite: one assembly layer `rojev/app`, robots as executor plug-ins; slice 1 built (with differences), slice 2 not built, slice 3 replaced by LIBERO |
| [`system1-map.md`](features/system1-map.md) | System 1 map (start here for S1): call map (`Session` / LIBERO and the RoboCasa app), control loop → guard → runtime ← runner ← policy services, commitment, transport, monitor, messages, timeline, evidence |
| [`system2-map.md`](features/system2-map.md) | System 2 map (start here for S2): owner triggers / lanes / holds / lease / memory, workers, planning lane (step planner + R1.5 client), Jev v1, done-check cascade, messages to S1, timeline, evidence |
| [`jev-interaction.md`](features/jev-interaction.md) | Jev v1 interaction lane (legal acts, calibration, abstention, budgets, REVISE), owner holds / dialogue / utterances / step events, MJev end-state checks + step watcher; flags, spoofs, tests, what is not validated |
| [`robot-interface.md`](features/robot-interface.md) | Robot interface (`rojev.robot`: MuJoCo SO-101, mock; hardware adapter later) and robot plug-ins (`Embodiment`, `Backend`, `ActionSpec`; LIBERO, RoboCasa): limits, cameras, clocks and time modes |
| [`README.md`](features/README.md) | Levers: current knobs that change behaviour: default, where set (`SessionConfig` key / flag / constant), trade-off, owning page |

## Utilities: `utils/` (harness, S1 supply, scoring, tracing)
| Doc | Covers |
|---|---|
| [`s1-supply-and-commitment.md`](utils/s1-supply-and-commitment.md) | S1 runner: chunk supply, measured latency, floor and lead time, commitment policies (fixed / phase / consensus) |
| [`live-harness.md`](utils/live-harness.md) | Live RoboCasa harness: `realtime.py` flags and outputs, 19-case scenario matrix, invariants, declared spoofs, viewer, video, matrix results |
| [`scorecard.md`](utils/scorecard.md) | T8 revision-benchmark scorecard: inputs, window and answer rule, metric definitions, aggregation, what is not validated |
| [`observability.md`](utils/observability.md) | Phoenix tracing: setup, what spans contain, measurements, gotchas |

## Feature history: `features/history/` (superseded designs; what replaced them, lessons that still apply)
| Doc | What it was |
|---|---|
| [`v2-levers.md`](features/history/v2-levers.md) | v2 levers registry (chunk-synchronous OneJev supervisor); authored-task budgets still read by `tasks.py` |
| [`v2-pi052-executor.md`](features/history/v2-pi052-executor.md) | π0.5² set-up, steering slot, subtask vocabulary, Kaggle 2×T4 runner |
| [`v2-engineering-notes.md`](features/history/v2-engineering-notes.md) | v2 code-reuse plan and licences, MJev on Apple silicon, platform notes |
| [`phase0-install.md`](features/history/phase0-install.md) | G0 install gate of the SO-101 plan (local OneJev 0.8B) |

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
