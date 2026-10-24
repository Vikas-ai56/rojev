# Execution monitor: kinematic check with a split-conformal threshold (step 4.3)

Code: `rojev/system1/monitor.py` (`ExecutionMonitor`, `MonitorModel`, `Recording`, `fit_monitor`, `regressors`),
`rojev/robot/effect.py` (`EffectModel`, `JointTargetEffect`), `executors/robocasa-pi052/embodiment.py`
(`PandaOmronEffect`), `executors/robocasa-pi052/calibrate_monitor.py`. Tests: `tests/test_system1_monitor.py` (52).
Opt-in: `realtime.py --execution-monitor MODEL.json`; off in the v1 runs.

## What it checks
- Per control tick: did the tracked quantities move as the applied actions command, the way they do on nominal runs?
  (CheckVLA's check without its world model; training-free.)
- Catches: blocked arm, slip, collision, being pushed. Misses anything that leaves the tracked quantities unchanged
  (object slipping from the gripper, wrong object; on PandaOmron also orientation and base faults).
- Advisory: a deviation asks for a fresh chunk and a decision; it never stops or holds the robot itself.

## Robot plug-in: `EffectModel`
| Member | Meaning |
|---|---|
| `tracked(state)` | measured quantities, from the observation state vector |
| `command(state, action)` | what the action asks of them, any units (the monitor fits the map; no controller gains) |
| `resolution` | smallest real change per tracked quantity (> 0); lower bound of the residual scale |
- `JointTargetEffect(resolution)`: any joint-target `Robot`; tracked = joint positions, command = target − measured.
- `PandaOmronEffect`: tracked = end-effector position relative to the base (`state[0:3]`); command = action dims 0:3
  (OSC position delta) and 10 (torso); resolution 0.1 mm (the simulator has no sensor noise).

## Model (`regressors`, `MonitorModel`)
- Per tick k ≥ 2w − 1 (w = `window` ticks): y = tracked change over the last w ticks; x = commands summed over this
  window and the one before (2n columns), so the map absorbs controller gains and tracking lag.
- Fit: least-squares `gain` (2n × m) on nominal runs; `scale` = residual std per axis, ≥ `resolution`.
- Score = RMS over axes of (y − x·gain) / scale; one number per tick.
- `exercised`: regressors ever non-zero in the fit. A window using a never-exercised direction is not scored
  (`score = None`, counted in `uncovered`). Magnitude is not boxed; saturation beyond the calibrated range shows as alarms.

## Calibration (`fit_monitor`, `calibrate_monitor.py`)
- Split by run: runs 0, 2, 4, … fit gain and scale; runs 1, 3, 5, … calibrate (a segment split gave 0.17 false alarms
  per segment at α 0.05; decisions 2026-10-21).
- Calibration runs are cut into full `segment`-tick segments, each giving its maximum score. Threshold = the
  ⌈(n+1)(1−α)⌉-th smallest of the n maxima → a segment of a new nominal run alarms with probability ≤ α.
- Exact for runs exchangeable with the calibration runs; approximate per segment (segments of one run correlate).
- α is per segment; `alarms_per_minute(hz) = α · 60 · hz / segment` states it in time.
- Refused: resolution ≤ 0; segment < 2 windows; fewer than 2 runs; fewer than ⌈1/α⌉ − 1 full calibration segments;
  a recording with real-time factor below `--min-realtime-factor` (0.98) or none; recordings that mix policies.
- `Recording.conditions` (from `realtime.py --monitor-record`): real-time factor, policy, latency, task, seed, commit,
  starved ticks. Recalibrate when the robot, controller, policy or task family changes.

```
uv run python realtime.py … --monitor-record runs/r1.npz          # nominal runs, no faults
uv run python calibrate_monitor.py runs/*.npz --window-s 0.5 --segment-s 5 --alpha 0.1 --out monitor.json
uv run python realtime.py … --execution-monitor monitor.json [--block-at T --block-s D]
```

## Deviation flow
1. `ExecutionMonitor.observe(TickRecord)` in the control-loop thread (`before` / `after` captures: no tick lost).
2. Score above threshold → one `Deviation(epoch, cmd_id, t, score, threshold, source="kinematic")` per excursion
   (re-armed when the score drops back).
3. `realtime.py` `on_deviation`: `owner.post(d)` and `runner.refill_now(d.cmd_id)`.
4. Owner: about the active command → `snapshot.deviation`, trigger `deviation` (interaction lane when configured);
   otherwise logged as `stale deviation`. Jev offers only continue or wait on a deviation (decisions 2026-10-24).
5. Runner: the next step asks for a chunk on the newest observation even with actions queued.
- Summary: `monitor.threshold`, `max_score`, `uncovered_ticks`, `deviations`; per tick `score` in `ticks.jsonl`.

## Evidence
Tests (`tests/test_system1_monitor.py`) pass in the full suite re-run 2026-10-24. Simulated planar arm through the real
runtime, guard and loop (50 Hz, first-order lag + sensor noise, window 10, segment 50, α 0.05):
| Check | Result |
|---|---|
| Blocked at the fastest point of the motion | detected within one window, 20/20 seeds |
| Pushed without a command (5 ticks) | detected within one window, 20/20 seeds |
| Blocked while not asked to move | no alarm |
| False alarms, 10 calibrations × 8 runs, 40 held-out runs | mean rate within the conformal bound (α + 3 standard errors) |
| Online vs offline scores, unexercised directions, resolution bound, refusals, JSON / npz round trip | pinned |

| Live RoboCasa | Result | Status |
|---|---|---|
| Scripted S1 calibration (`results/monitor/skills-20261021/monitor.json`: window 0.5 s, segment 5 s, α 0.1, 12 segments) | threshold 2.54; residual scale ≈ 2.9 cm (the runs press into contact while grasping) | stands: the scripted S1 reads simulator poses, not images (rule of E15) |
| 2 s block during a grasp phase (scenario `blocked`) | peak score 2.15 < threshold 2.54: not detected | same; decisions 2026-10-21 |

## Limits and open items
- Contact-blind: one threshold cannot separate intended contact from a fault; contact in calibration runs inflates it.
  Open: contact-aware calibration from a real signal (perception, force, world-model prediction), no hand tuning.
- Never calibrated or run with XR-1 (or π0.5² on clean frames).
- Calibration refuses runs below real-time factor 0.98, but lockstep runs with a remote S1 run slower than real time
  (XR-1 parity: 0.601), so lockstep recordings cannot calibrate it as built.
- Position only on PandaOmron (no orientation, no base).
- Spurious numpy matmul `RuntimeWarning` in the RoboCasa env (numpy 2.2.5 pin; results equal `einsum`; decisions
  2026-10-21).
