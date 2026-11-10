# Target system: how rojev should look on a real robot (2026-11-07)

**Status:** a target design, not built unless the status table says so. The first version is the smallest system that
works on the arm; everything under "Later" is noted and decided by test results, not planned.

**Sources:**
- research 2026-11-06/07 (S2 for a real robot; GR00T; MolmoAct2);
- the cross-check against the Jev × Helix plan, `architecture.md`, the decisions and the owner's notes;
- `hardware-port.md` (safety backlog).

**Open decisions** are listed at the end and are not logged as decisions yet.

## Goal
A real robot (SO-101 first, then other arms) that:
- does **known skills in new combinations and new scenes** of environments it has been set up in;
- takes the user's changes mid-task;
- stays safe;
- needs only small fine-tunes per robot.

Not zero-shot: each environment is learned through a setup phase and memory.

## Principles
- **Real robot first.** The simulator proves plumbing only.
- **First working system first;** add parts only when a test shows the need.
- **Robot-specific means configuration, never code in the core.** Each robot is a profile plus small plug-ins. The core reads them.
- **Independent parts behind small interfaces,** each maintainable alone.
- **Models sit behind one client interface per role,** so any serving engine plugs in. The engine is undecided (scara_inf is a candidate).
  - S1: `Policy`.
  - Planner and Jev: `Decider`.
  - Done checker: `Verifier`.
- **The owner is the only writer of intent and evidence.** Models only propose or report evidence; nothing reaches the motors without the owner and the safety kernel.

## Layout
```
 robot side (local CPU, works when the network does not)          model side (GPU; served by any engine)
 ┌──────────────────────────────────────────────────────────┐     ┌─────────────────────────────────────┐
 │ robot adapter (profile: joints, units, cameras, limits)   │     │ S1 policy  (GPU in the arm's LAN or  │
 │ control loop (robot rate) → safety kernel → motors        │◀────│            region: half-chunk budget)│
 │ execution runtime (chunks on frame time, late = skipped)  │     │ S2 planner (on events, ≤ ~1 Hz)      │
 │ owner: intent, task state, ledger, memory writes          │────▶│ done checker                         │
 │ S1 runner (pacing) · S2 workers (planning, interaction)   │     │ Jev (user changes)                   │
 │ recorder (episodes, step labels, event log)               │     └─────────────────────────────────────┘
 └──────────────────────────────────────────────────────────┘
```

## Configuration, not code
| File (per …) | Holds |
|---|---|
| **Robot profile** (per robot) | Joints, limits, units, control rate; cameras and depth source; calibration; safety-kernel parameters; skill list (step labels S1 was trained on, each with preconditions, effects and measured success) |
| **Environment profile** (per environment) | Static memory: object catalog, fixtures and their places, keep-out zones |
| **Model endpoints** (per deployment) | Per role: engine, address, auth, deadline |
| **Runtime settings** (per deployment) | Leases, timeouts, review window, ask budget, check cadence |

## Sensing (assumed SO-101 defaults)
- **Cameras:** LeRobot's standard SO-101 setup, two RGB cameras (fixed `front` + `wrist`), no depth. Declared in the robot profile, so other hardware is a profile change.
- **3D without depth:** objects on the table are placed in 3D from the calibrated front camera plus the measured table plane. Stacked or tall objects stay approximate and are marked as such.
- **Depth:** a depth camera, when the hardware has one, is an optional profile entry. S1 models keep using RGB.
- **Timing:** every sensor reading carries its own timestamp on one clock. A stale or broken camera holds the robot.

## S1: how to move
| First version | Later (if tests show the need) |
|---|---|
| A VLA fine-tuned per robot on step-labelled demos: GR00T N1.7 or MolmoAct2, chosen by a bake-off on the arm | Points / traces as extra input; real-time chunking; handoff-trained skills |
| Takes step text (a trained label) + cameras + joints, returns action chunks | Longer history |
| Whole-task mode kept as the control | — |

## S2: what to do, and whether it worked
| First version | Later (if tests show the need) |
|---|---|
| **Planner:** a generative embodied VLM (Embodied-R1.5 first). Wakes on task start, step done, stall/failure and user change. Proposes the next step from the skill list, with the target object | Advice lane (slow reasoning as dated, expiring advice); OBSERVE / THINK acts; failure diagnosis; imagined outcomes before risky steps |
| **Done checker:** a frozen progress/success model, threshold set on held-out real episodes | Fine-tuned on our episodes if it misses the bar; learned feasibility from episodic memory |
| **Task state + ledger** (memory, below) | 3D object graph; identity across cameras |
| **Jev:** user changes (continue / revise / wait / ask / abort) and typed checks | Speech recognition, multilingual input, status answers, encoding images once for all questions |
| **Anti-flip-flop:** minimum dwell on a step; no repeated question without new evidence | — |

## Memory
| Kind | Holds | Lifetime | Written | Read |
|---|---|---|---|---|
| **Static** | Object catalog (names, appearance), fixtures and their places, keep-out zones, calibration | Per environment, across days | Only at setup or when the user confirms | Every task |
| **Task** | Goal and plan; step ledger (done / failed / undone, with evidence); where each object is now and its state (open, held); this task's dialogue | One task, reset after | Ledger: only on checked effects. Object state: from fresh observations, dated; old ones expire to `unknown` | Planner, owner (preconditions) |
| **Episodic** | Past outcomes, failures, corrections | Append-only, across tasks | From day 1, everything (failures included) | Later: feasibility statistics, training |

**Repetition:** the planner compares the goal's required effects with the effects achieved in the task state.
- A step whose effect already holds is skipped.
- An undone effect is redone.
- A legitimate repeat (two apples) runs twice.

## Safety kernel (configured per robot)
| First version | Later |
|---|---|
| Layer 0: servo limits and an e-stop | Layer 2: whole-chunk check for self-collision and keep-out zones |
| Layer 1: non-finite values, position, velocity and acceleration limits, across chunk boundaries | Layer 3: stall / overload from servo current and temperature; gripper force cap |
| Heartbeat, stale-data and network-loss hold; a stop path that never waits for a model | Layer 4: preconditions checked by the owner |
| Human supervision in the room during first trials | Layer 5: reactions (keep grip, back off, low torque); human detection |

**Before the arm moves under a model:**
- the four deferred controller fixes (decision 2026-11-06);
- per-sensor timestamps;
- a hold that keeps the grip.

## Data and evaluation
- **Setup protocol per environment:** calibration check, table plane, keep-out zones, object catalog confirmed by the user, a few baseline trials.
- **Recorder:** LeRobot datasets with step labels (`sub_tasks`), handoff and recovery demos, and an event log with sent vs applied commands.
- **Evaluation fixed before data collection:**
  - held-out layouts and objects;
  - baselines: S1 alone on the whole task, and oracle steps;
  - trial counts with confidence intervals;
  - mid-task-change metrics (the revision benchmark).

## First-version targets (proposed; counts and splits set before testing)
| Part | Target |
|---|---|
| S1 | Changing the step text changes what the arm does (swap / erase / contradict) on most trials; skill success measured from the previous skill's end state |
| Planner | Correct next step on a labelled set of real decision points; impossible steps and hallucinated objects reported separately |
| Done checker | False "done" vs missed "done", per attempted step, at a threshold set on held-out real episodes |
| Interaction | Acknowledgement ≤ 200 ms; guarded response measured per path; wrong-goal commitments and ask precision |
| System | Success on new 2–3-step combinations of trained skills in held-out layouts, against both baselines |
| Safety | A hazard log, stop and violation rates under scripted faults, measured stopping time |

## Open decisions (from the cross-check; not logged yet)
1. **World state:** held by the owner as dated evidence (reverses the retired explicit world model, `architecture.md`).
2. **Per-skill preconditions and effects** as authored data in the robot profile (earlier: generic preconditions only).
3. **Done checker:** frozen first, then fine-tuned on real episodes if it misses the bar (earlier: frozen).
4. **Planner choosing every step** in step mode, with whole-task mode as the control (earlier: S1 sequences, Jev answers changes).
5. **S1's GPU in the arm's LAN or region** (cloud is fine for S2).

## Status (checked against the code, 2026-11-09)
| Part | Status |
|---|---|
| Owner, leases, ACKs, execution runtime, control loop, S1 runner, transport | Built (simulator) |
| Safety | Part of layer 1 only (position and velocity clamps, holds, stop, non-finite S1 output refused) |
| S1 | GR00T adapter built, untested on a GPU; MolmoAct2 not started |
| Planner | Event-driven planning lane and `StepPlanner` built (stand-in model, simulation tests); R1.5 client not wired; bake-off ready, not run (`model-bringup/s2-bakeoff.md`) |
| Done checker | Cascade built (switch cues → check → quick recheck to confirm; fallback period); no real-robot model wired; nothing emits COMPLETE yet; R1.5's yes / no check is in the same bake-off |
| Task state / ledger | Task memory built: goal + step ledger (DONE / UNDONE with evidence), owner sole writer, read by deciders; episodic log of finished tasks (JSONL). Object states not yet (no perception source writes them) |
| Jev | Works for whole-task S1 in simulation; p50 961 ms against a 1.0 s deadline; no fitted calibration; never on real cameras |
| Recovery | Kinematic monitor → deviation → answer, contact-blind |
| Robot adapter | Shared assembly `rojev/app` built (any robot plug-in: embodiment + backend); LIBERO plug-in in progress; simulated arm in tests; no hardware adapter yet |
| Recorder, setup protocol, evaluation protocol | Not built |
