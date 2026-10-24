# Architecture rules (v3, Jev × Helix)

Binding rules for all code. Sources: `plan/jev_helix_implementation_plan.pdf` (pp.1–18), the review amendments in `plan/v3-roadmap.md` §2, `plan/v3-v1-spec.md`, and `decisions.md` (2026-10-20 entries). Code map: `CODEBASE.md`.

## Roles
1. **Two systems, separate jobs.** S1 (visuomotor policy: XR-1 by default, π0.5² supported) answers *how do I do the current step from what I see now*. S2 (Jev = OneJev typed choices; MJev and trackers measured, not in the v1 loop) answers *is the goal still right, what next, do I need to ask*. Jev never emits motor actions; S1 never changes the user's goal.
2. **One state owner.** The only writer of instruction revision, desired command, evidence/progress and outstanding jobs. It never blocks on inference.
3. **Advice is evidence, not authority.** Slow workers (deep models, advisors) return dated evidence; only the owner turns proposals into commands.

## Authority and consistency
4. **Proposal ≠ command ≠ active.** Jev proposes; the owner accepts (`AcceptedGoal`); a command is active only when the execution runtime applies its first action and ACKs it. Active never moves backwards (ACK `seq`).
5. **Epoch on every message.** Nothing from before a restart can act; a restart requires a new command.
6. **Decisions expire and carry their assumptions.** Each proposal has its observation time, preconditions and expiry; the owner re-checks preconditions on the newest snapshot before accepting. Commands hold a lease; `Revoke` drops their queued actions at once.
7. **Elapsed actions are skipped, never replayed.** A new chunk replaces the queue from its first due time; segments are blended.

## Time
8. **Each layer runs on its own clock.** The control loop never waits for a model; S1 and S2 run asynchronously. In simulation, `lockstep` (simulated time stops while any model thinks) is the evaluation mode while the models run on a remote GPU (decision 2026-10-23); `realtime` keeps physics running during inference and measures responsiveness; hardware runs real time only.
9. **Latency is measured per path**: event → ACK, event → guarded response, event → new goal; never a single number.

## Safety
10. **No physical backtracking.** Speculation is computation only; nothing speculative moves the robot. Irreversible switches (gripper, contact) change only while running an active command's chunk. A user change of mind gets an immediate guarded pause from the owner, then a new goal from Jev.
11. **The guard is independent.** Bounds on every action, starvation watchdog, owner heartbeat, stop path; it works if every thread above it is dead. Bound clamps are reported, never silent.
12. **Limits are hard checks** from the robot model plus sourced extras, enforced in front of every backend; planner or policy preferences are not guarantees.

## Decisions
13. **"Do nothing" is a real choice.** CONTINUE, GUARDED_WAIT, OBSERVE, ASK and ABORT are first-class operations. Missing evidence → look again or ask; more computation only helps when the evidence is already there.
14. **No flip-flopping.** Switch only past a probability margin and a minimum dwell; budgets on asks and waits; no repeated question without new evidence.
15. **Candidates come from what S1 can actually do**: executor-supported skills × tracked objects × argument types, generic preconditions only. No per-task or per-object hand-coded checks. Candidate recall is measured separately.
16. **A model probability is not a success probability.** Recalibrate after every model, quantization or option-set change.

## Data
17. **No privileged data reaches a model.** Simulator truth (`SimGroundTruth`, `tasks.py` checks) is for evaluators, baselines and training labels only.
18. **Object identity is tracked**, not re-guessed: MJev points seed a tracker; lost, occluded or ambiguous → `unknown`.

## Structure
19. **Write it as if it were hardware.** The simulator is one backend behind the robot interfaces (`rojev/robot`); the runtime is robot-agnostic and a robot plugs in through its `ActionSpec`. Porting means a new backend and spec, not new pipeline code.
20. **Open-source frozen components first** (v1); train only where tests show the need. Jev must beat the native VLA (with full history) and a scripted supervisor, or its role is narrowed or removed.
21. **Deep modules, one-way dependencies.** `rojev/` is split by the architecture: `protocol` (shared contracts) → `robot` → `control` → `system1` / `system2` → `owner` → `eval`; `observability` beside; v2 code quarantined in `legacy_v2` until deleted. Each package exposes a small interface in `__init__`; imports only point down the stack (`tests/test_architecture.py`). A new file goes in the package that owns its responsibility, never the nearest folder.
22. **Code rules:** `rojev/` is core only (demos in `examples/`, checks in `tests/`, experiments in `playground/`); no stub modules; no inline comments; docs (`CODEBASE.md`, `features/`, `decisions.md`) change with the code.

## Retired (v1–v2 rules no longer binding)
- Explicit `world_version` world model, `MotionPlanner` plug-ins, `rojev/skills/` library, SO-101 kinematic approach rules: the SO-101 planner track is paused; S1 is a learned policy.
- SayCan-style product of OneJev usefulness and feasibility gates: OneJev's probability is not a success estimate (rule 16).
