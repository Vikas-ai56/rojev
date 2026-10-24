# Moving to hardware: what changes and what to focus on

Short list of what the simulator lets us defer and a real robot will not. Gates for any model: `model-bringup/`.
Robot interface: `features/robot-interface.md`; SO-101 executor spec: `plan/specs/any-robot/03-so101-executor.md`.

## Lockstep is a simulator mode (kept for now)
- In simulation we evaluate in **lockstep**: simulated time stops while S1 thinks (`realtime.py --lockstep`). It is
  needed now because S1 runs on a remote GPU (0.35–0.55 s round trip vs an 0.8 s XR-1 chunk) and because published
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
- **Safety:** local stop path with measured stopping time, independent of any model.
