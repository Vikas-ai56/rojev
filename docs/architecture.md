# Architecture rules

> **2026-10-16 pivot:** the active direction is `docs/onejev_learned_policy_plan_v2.md` (OneJev supervising a frozen SmolVLA in RoboCasa). Rules 2–5, 8 and 10 still apply to the supervisor. Rules 7, 9 and 11 (planners, skills library, planner-side limits) are **paused** with the SO-101 track; `rojev.robot` stays as infrastructure.

Sources: `docs/plan/visual_skill_agent_implementation_plan.md`, `docs/plan/motion_planner.md` (design advice, not measured findings), and user decisions of 2026-10-16.

1. **Write it as if it were hardware.** The simulator is one backend behind the same interfaces a real SO-101 will use: sensors in (joint states, camera frames with timestamps and calibration), commands out (joint targets). MuJoCo has two roles. As the **robot model** (MJCF kinematics and collision geometry for IK and collision checks), planners may use it on hardware too, the way MoveIt uses a URDF. As **the world** (stepping physics, true object poses), it appears only in the sim backend and the evaluator. Porting to hardware means swapping the backend; the ingestion pipeline stays the same.
2. **Three separate jobs:** *what* the object is (OneJev + perception), *where* it is in robot coordinates (state estimation), and *how* to reach it (feasibility, planning, control). OneJev does only the first, plus skill choice.
3. **Two clocks:** OneJev decides at skill boundaries and on failure. Control runs continuously at the controller rate.
4. **World state is explicit and versioned** (`world_version`). Every estimate carries its frame, its source (`sim_ground_truth`, `monocular_plane`, `onejev`, …) and its uncertainty. Skills read it and never compute hidden geometry themselves.
5. **No privileged data reaches OneJev.** OneJev's input filter drops `sim_ground_truth`-sourced fields. The evaluator alone reads ground truth, and nothing it reads is fed back.
6. **Feasibility is SayCan-style:** usefulness (OneJev) × deterministic gates (`target_known`, `reachable`, `grasp_available`, `path_valid`). OneJev answers over all options and everything is logged; the validator rejects invalid picks afterwards. Don't multiply in learned success estimates until both sides are calibrated.
7. **Planners are plug-ins.** One `MotionPlanner` base class takes a `MotionRequest{target_id, target, frame, world_version, permitted_contact}` and returns a trajectory or an explicit failure. Implementations are interchangeable: mink (local differential IK) and OMPL (global collision-free paths) to start. The SO-101's lift, elbow and wrist-flex axes are exactly parallel (verified), so the arm is a flat chain swung by the pan joint. A target is a position plus an approach **pitch** within the vertical plane through the target, plus an optional roll (`wrist_roll`). Approach yaw cannot be chosen.
8. **Two kinds of replanning:** a small target move updates the low-level target without asking OneJev again. Unreachable, blocked or changed task → OneJev reconsiders. A stale `world_version` means stop, correct locally, or replan.
9. **Skills live in `rojev/skills/`:** one extensible base class (preconditions → plan via `MotionPlanner` → execute → verify → `SkillResult`) plus a registry.
10. **Sim first, then degrade:** start from simulator ground-truth estimates, with that use logged. Later swap in noisy or monocular estimates (calibrated fixed camera, ray–plane) and measure the degradation.

11. **Limits are hard checks:** joint range ∩ ctrlrange and forcerange from the model, plus sourced extras such as velocity. They are enforced at three points regardless of planner: on the request/goal, on every returned trajectory before it runs, and in the safety filter in front of every backend, with every clamp reported. Planner costs are preferences, not guarantees.

Roadmap, not now: monocular estimation, IBVS, stereo, moving objects, real drivers.
