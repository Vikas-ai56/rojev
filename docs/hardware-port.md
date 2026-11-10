# Moving to hardware: what changes and what to focus on

Short list of what the simulator lets us defer and a real robot will not. Gates for any model: `model-bringup/`.
Robot interface: `features/robot-interface.md`; SO-101 executor spec: `plan/specs/any-robot/03-so101-executor.md`.

## Lockstep is a simulator mode (kept for now)
- In simulation we evaluate in **lockstep**: simulated time stops while S1 thinks (`realtime.py --lockstep`). It is
  needed now because S1 runs on a remote GPU (round trip p50 0.51 s, p90 0.73 s, E23; gap-free supply of a 16-action
  chunk needs ≤ 0.375 s) and because published
  benchmarks are scored that way. Real time (`realtime.py` default) is kept and reported alongside.
- **A robot has no lockstep.** Before hardware, the same tasks must pass in real time with 0 starved ticks.

## Delay on the robot (in order)
1. S1 on a GPU next to the robot (Helix runs S1 and S2 onboard); smaller model (XR-1-2B), fewer denoising steps.
2. Runtime alignment we already have: chunks at observation time, past actions skipped, early requests, blending.
3. Inference-time real-time chunking on the flow-matching S1 (no training).
4. If still not enough: async post-training of S1 with the robot's measured delay distribution (`how-to-train.md`).

## Other things the simulator hides
- **Cameras:** freezes, drops, black / stale frames. The sim run check (`harness/frame_check.py`) detects noise /
  blank; on hardware it moves to the robot layer, adds staleness by timestamp, and the guard holds on bad input.
- **Completion:** no simulator success signal; needs the trained completion output; budget / watchdog only as a
  safety stop.
- **Clocks:** camera, joint and model timestamps on one timeline (stale joints → wrong camera pose).
- **Calibration and slop:** hand-eye offsets and joint backlash (SO-101) turn into centimetres at the object.
- **Contract:** the robot's own action / state layout and rate; never truncate a Panda action to fit an SO arm.
- **Safety:** local stop path with measured stopping time, independent of any model; the full layer is below.

## Robot safety layer (backlog: research, then spec; nothing beyond the first row is built)
Built today: per-joint position and per-tick velocity clamps (`SafetyFilter`), holds on starvation / owner loss,
`stop`, and a non-finite S1 action failing its request (GR00T adapter). Damage also comes from commands that are only
impossible after an earlier action (a grasp changes the wrist pose, a target past reach stalls a servo), so each layer
below is a per-robot plug-in (a safety model: limits, kinematic model, keep-out zones, telemetry thresholds, reactions)
checked by a generic core guard; S2 touches only layer 4.

| Layer | Catches | Robot-specific input |
|---|---|---|
| 0 Hardware | servo angle / torque / overload limits (STS3215 EEPROM), e-stop | per arm, set once |
| 1 Command | non-finite, position, velocity, **acceleration / jerk**, across chunk boundaries | joint limits |
| 2 Whole chunk, before execution | forward kinematics over every future target: self-collision, table / walls (keep-out zones measured at calibration) | URDF / MJCF, zones |
| 3 Live telemetry | tracking error, load / current, temperature, voltage → back off, lower torque; gripper effort cap once grasped | thresholds fitted on nominal runs |
| 4 Task (S2 + owner) | a step whose preconditions do not hold (drawer closed, object absent); S2 points checked for reachability (IK) | skill list with preconditions / effects |
| 5 Reaction | hold keeping grip, slow back-off along the last safe path, low-torque mode, then ask; never just drop | per robot |

Research first (real hardware only): STS3215 protections and telemetry rates, LeRobot's limits, lab practice for VLA
safety (safety filters, chunk validation, Gemini Robotics' semantic + collision layers), chunk-checking libraries
within a 30 Hz budget (MuJoCo, Pinocchio + coal, cuRobo), stall / contact detection from servo current.

## Other open items for the SO-101 (2026-11-07)
- Calibration: the arm's sign / offset / gripper span for `config/so101.yaml` `lerobot` (simulator values declared).
- S1: GR00T parity with its reference client and p50 / p95 latency on an Ampere+ GPU near the arm (`model-bringup/groot.md`); accept the gated `nvidia/Cosmos-Reason2-2B` terms once; MolmoAct2 adapter; bake-off on the arm.
- Data: recorder for LeRobot datasets with `sub_tasks` step labels and `modality.json`; front + wrist cameras on rigid mounts.
- Runtime: the any-robot assembly layer (`plan/specs/any-robot/`) to run S1 / S2 in the loop on the SO-101; the
  deferred controller fixes (decision 2026-11-06).
