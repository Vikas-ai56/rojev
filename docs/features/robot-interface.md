# Robot interface and robot plug-ins

Code: `rojev/robot/` (`base.py`, `types.py`, `limits.py`, `mock.py`, `so101.py`, `action_spec.py`, `effect.py`,
`sim/`). Contract tests: `tests/test_robot_contract.py` (24, 1 of them `timing`, run with `-m timing`: the real-time physics thread catches up in bursts, so its 10 ms check depends on OS scheduling; the six shared contracts run on both backends),
`tests/test_robot_sim_world.py` (4). Runtime side: [`v3-runtime.md`](v3-runtime.md).

## Plug-in points
- **`Robot`**: a joint-target arm with timestamped sensors (simulated SO-101, mock). It joins the runtime through
  `RobotBackend` + `joint_target_spec`.
- **Backend**: anything that can return an observation and apply one action (RoboCasa, LIBERO). The loop steps it.
- **`ActionSpec`**: the robot's action layout: bounds, what "hold still" means, switches (gripper), which switches
  cannot be undone. Switches also give the done checker its cue.
- **`Embodiment`**: what the shared assembly needs from a robot: name, `ActionSpec`, camera names, start command. The
  robot plug-in builds its own backend (it needs the task and seed) ([`app-session.md`](app-session.md)).

| Robot | Assembly |
|---|---|
| RoboCasa Franka on a mobile base (12 numbers, 20 Hz) | its own older wiring |
| LIBERO Franka (7 numbers, 20 Hz) | shared assembly |
| SO-101 in MuJoCo (joint targets) | tests only |

## LIBERO plug-in
- **Action:** LeRobot's LIBERO action: hand position and rotation changes (6) + gripper (-1 open, +1 close). Zero
  change holds the arm; the gripper keeps its last value and is treated as a switch that cannot be undone.
- **Environment:** LeRobot's own LIBERO environment and image / state processing, so S1 sees exactly what LeRobot's
  evaluation would give it. Episode *e* uses LeRobot's *e*-th start state and seed. Success = LeRobot's own check.
- **Evaluation:** one run of the shared assembly per episode, a fresh environment each time, the whole task sent to
  S1 (no S2 steps yet), stopped on success or at LeRobot's step limit; results saved after every episode.
- **Where it runs:** LIBERO installs only on Linux, so the simulator runs on the GPU machine next to the S1 service.
- **Tests:** a fake LIBERO under LeRobot's real code, on the laptop.

## SO-101
- Units: recorded data and S1 model I/O use LeRobot's (body joints in degrees around the calibration midpoint,
  gripper 0–100); `so101.lerobot_units()` converts to and from the core's radians from `config/so101.yaml` `lerobot`
  (simulator values declared; the real arm's come from its calibration). Cameras for S1: fixed `front` (scene) and the
  arm's `wrist_cam` (decision 2026-11-07).
- A hardware SO-101 adapter is not built.

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
- LIBERO Panda (7-D, 20 Hz): six hand deltas hold 0; the gripper is a `KEEP` switch at 0.0, irreversible.