# Slice 3: a second robot, MuJoCo SO-101, through the same `Session`

**Status:** SPEC, not built. Grounded on `v3-runtime` @ `0b5caa9`. One PR, after slice 1 (independent of slice 2).

## Purpose
Prove "a new robot = embodiment + backend + policy adapter + tasks" with a robot of a different kind: a 6-joint
position-controlled arm with its own clock (MuJoCo SO-101), not a 12-dim velocity-controlled mobile manipulator.

## What the executor supplies (`executors/so101-mujoco/`)
| File (new) | Content | Built from (exists today) |
|---|---|---|
| `pyproject.toml` + `uv.lock` | uv project depending on `rojev` (editable) | `executors/robocasa-pi052/pyproject.toml` pattern |
| `embodiment.py` | `Embodiment("so101", joint_target_spec(robot, switches={"gripper": …}, irreversible=("gripper",)), cameras, start=robot's measured joints, JointTargetEffect(…))` | `rojev/robot/action_spec.py:100`, `rojev/robot/effect.py:31`, `rojev/robot/so101.py:16` (`simulated`) |
| `stand_ins.py` | SPOOF (declared): `JointScriptPolicy` (joint-space waypoints per named step, chunked at the control rate; K samples by waypoint noise) | `examples/scripted_trajectory.py` waypoints, `tests/test_system1_runner.py:242–270` ramp |
| `run.py` | parse (`rojev.app.cli` + `--scene --steps --viewer`) → `RobotBackend(so101.simulated(...))` → `Session` → `run` → `write` | `rojev/control/robot_backend.py:15` |
| `tests/test_so101_session.py` | below | — |

Planner: `ScriptedSequence` over the authored steps (from `rojev.eval`); interaction: `AlwaysContinue` / keyword
rules (imported from the RoboCasa harness? **no**: the keyword stand-in moves to `rojev.eval` only if both executors
need it; otherwise `AlwaysContinue` suffices here). No step verifier (no MJev for this scene).

## Invariants preserved
- No change in `rojev/` for this robot (if one is needed, it is a core bug fixed in its own PR, never an SO-101 branch).
- The robot clock paces the loop (`ControlLoop` with `backend.clock`, `rojev/control/loop.py:85–92`); the safety
  filter in `Robot.write` still applies.

## Test plan
- Lockstep (simulated time) 3 s run: records complete, invariants green (`switches_held` on the SO-101 gripper),
  commands follow the scripted steps, `say=[(1.0, "wait"), (2.0, "go on")]` holds and resumes under `AlwaysContinue` /
  rules.
- Realtime 3 s run marked `timing` (wall clock).
- Same `rojev.eval.invariants.check` as RoboCasa, with the SO-101 `ActionSpec`.

## Done criteria
`cd executors/so101-mujoco && uv run pytest -q` green; diff of `rojev/` in this PR is empty.

## Not in scope
Real SO-101 hardware (Feetech driver, process isolation, driver-level hold-on-silence: research brief 2026-10-22);
a trained SO-101 S1.

## Docs in this PR
`docs/CODEBASE.md` (executor rows), new `docs/features/so101-mujoco.md` (what it proves, spoofs declared),
`docs/README.md`, `docs/decisions.md` (D-A5).
