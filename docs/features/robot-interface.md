# Robot interface: `rojev.robot`

Code: `rojev/robot/` (`base.py`, `types.py`, `limits.py`, `mock.py`, `so101.py`, `action_spec.py`, `effect.py`,
`sim/`). Contract tests: `tests/test_robot_contract.py` (24; the six shared contracts run on both backends),
`tests/test_robot_sim_world.py` (4). Runtime side: [`v3-runtime.md`](v3-runtime.md).

## Two plug-in points
- **`Robot`**: a joint-target arm with timestamped sensors. Backends: `MujocoRobot` (simulated SO-101) and `MockRobot`.
  It reaches the runtime through `rojev.control.RobotBackend` plus `joint_target_spec(robot, switches, irreversible)`.
- **Runtime plug-ins** for robots that are not a `Robot` (the RoboCasa PandaOmron, an environment stepped by the loop):
  an `ActionSpec` (`action_spec.py`) and, for the execution monitor, an `EffectModel` (`effect.py`); PandaOmron's live in
  `executors/robocasa-pi052/embodiment.py`.
- A hardware SO-101 adapter is not built; the single assembly layer for any robot is specified, not built
  (`docs/plan/specs/any-robot/`, decisions 2026-10-22).

## `Robot` shape
```
Robot (base.py)        connect / disconnect · read() → RobotObservation · write(JointCommand) → CommandResult
 ├─ MujocoRobot        sim/mujoco_robot.py; time_mode lockstep | paced | realtime
 └─ MockRobot          mock.py; commands become state (ros2_control mock style)
Clock                  SimClock (lockstep, paced) · RealtimeSimClock (physics thread) · MonotonicClock · ManualClock
Rate(clock, hz)        fixed-rate loop on the robot clock; same code on sim and hardware
```
- **Units:** canonical joint values = the MJCF joint coordinates (radians, gripper hinge included); each backend
  converts its own. Gripper: increasing `q` opens it.
- **Timestamps:** every sample carries `Stamp{acquired, received, seq}` on the robot clock; the simulator never stamps
  wall time.
- **Threads:** `read` / `write` are serialized (one I/O lock), so several threads may share a robot; every OpenGL
  context lives on one GL thread (`sim/gl.py`; a macOS CGL context used from a second thread blocks forever).
- **Cameras:** `CameraSpec(name, width, height, fps, depth, latency_s)`; each frame comes with `CameraInfo` (K,
  distortion, parent frame, `T_parent_optical` in ROS optical convention) computed from the model. Depth is float32
  metres along the optical axis. Fixed cameras only; tracking cameras and sensorsize intrinsics raise.
- **Time modes:**
  - `lockstep` (default): physics advances only inside `clock.wait_until`; deterministic. Tests, data generation.
  - `paced`: lockstep, never faster than the wall clock.
  - `realtime`: physics on its own thread with the wall clock; `clock.lag_s` / `max_lag_s` report falling behind;
    physics-thread errors (e.g. viewer closed) are raised in the caller's next `wait_until`.
  - `Rate.sleep` drops ticks missed while the caller was busy (no catch-up burst) and returns the overrun.
- **Ground truth:** `MujocoRobot.ground_truth` (`SimGroundTruth`: body poses, segmentation pixel counts, contacts) is
  for evaluators and tests only.

## Limits: hard, model-sourced, independent of planners
- `JointLimits.from_model`: position = joint range ∩ the driving actuator's `ctrlrange`; effort = actuator
  `forcerange`; velocity from `config/so101.yaml` (4.40 rad/s, an unverified STS3215 retailer value). Each records
  its source.
- `SafetyFilter` (inside `Robot.write`): clamps each target into its range and to `max_velocity × dt` from the measured
  position; every clamp is reported in `CommandResult.violations`. `write` before any `read` is refused.
- `JointLimits.trajectory_violations` checks position and velocity over a timed trajectory (for planners' output).
- `MujocoRobot` refuses any actuator that is not a position servo.

## Runtime action description (`ActionSpec`)
- Per dim: name, bounds, hold value (number, or `KEEP` = repeat the last value), switch `threshold`, `irreversible`.
- Used by the runtime for clamping (`limit`), holding (`hold`, refuses to guess `KEEP` dims without a seed), and
  freezing irreversible switches (`irreversible_change`, `keep_switches`).
- `joint_target_spec`: one `KEEP` dim per joint with the robot's position limits; actions are joint targets.
- PandaOmron (12-D, 20 Hz): arm and torso deltas and base velocity hold 0; gripper and base mode are `KEEP` switches at
  0.5; the gripper is irreversible.

## Scene
- `scenes/pick_place.xml` holds only the world (floor, light, fixed `overhead` camera, box, `robot_mount` frame).
- `rojev.robot.sim.compose()` attaches the vendor SO-101 MJCF with `MjSpec.attach`, leaving both files unmodified, and
  copies the robot's solver options (`attach` would silently keep MuJoCo's defaults).
- The box sits at (0.20, 0.12, 0.03), beside the home pose, so the overhead camera sees it.

## Verified
- Contract tests pass on both backends in the full suite re-run 2026-10-24: stamps follow the clock; an in-limit ramp
  has no violations and converges; out-of-range and over-speed targets are clamped and reported; write before read is
  refused; time modes (lockstep waits for the caller, realtime runs while it blocks, missed ticks dropped, physics errors
  reach the caller, physics stops on disconnect); camera rate, latency and calibration (the box projects onto its
  pixels).
- `make sim` (2026-10-16, SO-101 scripted trajectory): 0 safety clamps, final error ≤ 0.0006 rad; lockstep headless
  14.50 s sim in 4.0 s wall; realtime headless 14.52 s sim in 14.48 s wall.
- Runtime plug-in: `tests/test_system1_runner.py::test_any_rojev_robot_plugs_in_so101_lockstep` drives the SO-101
  through `ControlLoop`; `test_so101_realtime_keeps_its_clock_and_lag_budget` (`-m timing`) in real time.

## Not validated
- No hardware backend; joint unit tables (offset, sign, scale) for the real arm are not written.
- macOS MuJoCo warns `ARB_clip_control unavailable` (no reverse-Z depth); depth error was ≤ 3 mm at ~0.8 m, re-check
  before relying on it for fine geometry.
