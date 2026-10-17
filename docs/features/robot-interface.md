# Robot interface (Phase 1 / M1)

Code: `rojev/robot/`. Contract tests: `tests/test_robot_contract.py`, run against every backend.

## Shape

```
Robot (base.py)            connect / disconnect · read() → Observation · write(JointCommand) → CommandResult
 ├─ MujocoRobot            simulator backend; physics advances only inside SimClock.wait_until
 ├─ MockRobot              commands become state (ros2_control mock style)
 └─ (later) a LeRobot SO-101 adapter, behind the same interface
Clock                      SimClock (sim time, optional real-time pacing, paused()) · MonotonicClock · ManualClock
Rate(clock, hz)            the fixed-rate loop; identical code on sim and hardware
```

- **Units:** canonical joint values are the robot model's MJCF joint coordinates (radians, gripper hinge included). Each backend converts its own units. On the real arm this is a small per-joint table (offset, sign, scale; the gripper's direction).
- **Gripper:** increasing `q` opens it (the moving jaw is 17 mm from the gripper frame at q = 0 and 102 mm at 1.745).
- **Timestamps:** every sample carries `Stamp{acquired, received, seq}` on the robot clock. The simulator never stamps wall time.
- **Cameras:** a `CameraSpec(name, width, height, fps, depth, latency_s)` per sensor. A frame comes with `CameraInfo` (K, distortion, parent frame, `T_parent_optical` in ROS optical convention), computed from the model. Depth is float32 metres along the optical axis. Only fixed cameras are supported; tracking cameras and sensorsize intrinsics raise errors. On macOS, MuJoCo warns `ARB_clip_control unavailable … depth accuracy will be limited` (no reverse-Z depth). Measured error was still ≤ 3 mm at about 0.8 m, but re-check before relying on depth for fine geometry.
- **Ground truth:** `MujocoRobot.ground_truth` (`SimGroundTruth`) provides body poses, segmentation pixel counts and contacts. It is for evaluators and tests only.

## Limits: hard, model-sourced, independent of planners

- `JointLimits.from_model`: position = joint range ∩ the driving actuator's `ctrlrange`; effort = actuator `forcerange`. Velocity comes from `config/so101.yaml`: 4.40 rad/s, an unverified STS3215 retailer value. Each kind of limit records its source.
- `SafetyFilter` (applied inside `Robot.write`): clamps each target into its range and to at most `max_velocity × dt` from the measured position. **Every clamp is reported** in `CommandResult.violations`; a correct pipeline produces none.
- `JointLimits.trajectory_violations` checks position and velocity over a timed trajectory. Planners must apply it to their own output (M3).
- `MujocoRobot` refuses any actuator that is not a position servo, since its ctrl would not be a joint target.

## Scene

`scenes/pick_place.xml` holds only the world: floor, light, the fixed `overhead` camera, the box, and the `robot_mount` frame. `rojev.sim.compose()` attaches the vendor SO-101 MJCF with `MjSpec.attach`, leaving both files unmodified. The robot's solver options are copied across, because `attach` would otherwise silently keep MuJoCo's defaults (timestep 0.002, no elliptic cone, impratio 1). The box is at (0.20, 0.12, 0.03), beside the home pose: at (0.22, 0) the arm hid it completely from the overhead camera.

## Verified

- 16 contract tests on both backends: stamps follow the clock; an in-limit ramp produces no violations and converges within 0.01 rad; out-of-range and over-speed targets are clamped and reported; write before read is rejected.
- Camera: a 10 fps / 50 ms latency camera delivers 9–10 frames per second, each with exact latency. The box's true top face projects to within 1 px of its rendered pixels, and depth at that pixel matches within 3 mm.
- Example (`make sim`): 0 safety clamps, final error ≤ 0.0006 rad, viewer exactly real time (14.52 s wall for 14.50 s sim). Headless takes 4.0 s, dominated by rendering the 640×480 camera at 30 fps.
- OneJev now sees the overhead sensor camera. With the box hidden it answers `observe` (0.55), P(box) = 0.03, and the evaluator passes. The earlier `approach_box` failure came from the old free-camera view; viewpoint matters.
