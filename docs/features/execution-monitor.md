# Execution monitor: kinematic check with a split-conformal threshold (step 4.3, 2026-10-21)

Code: `rojev/system1/monitor.py` (`ExecutionMonitor`, `MonitorModel`, `Recording`, `fit_monitor`, `regressors`), `rojev/robot/effect.py` (`EffectModel`, `JointTargetEffect`), `executors/robocasa-pi052/embodiment.py` (`PandaOmronEffect`), `executors/robocasa-pi052/calibrate_monitor.py`. Tests: `tests/test_system1_monitor.py` (52, simulated planar arm through the real runtime, guard and control loop, lockstep). Opt-in (`realtime.py --execution-monitor`).

## What it checks
- Per control tick: did the tracked quantities move as the applied action commands, the way they do on nominal runs? (CheckVLA's check without its world model; training-free.)
- Catches: blocked arm, slip, collision, being pushed. Misses anything that does not change the tracked quantities (object slipping from the gripper, wrong object, orientation and base faults on PandaOmron).
- Advisory: a deviation asks for a fresh chunk and a decision; it never stops or holds the robot.

## Robot plug-in: `EffectModel`
| Member | Meaning |
|---|---|
| `tracked(state)` | measured quantities, from the observation state vector |
| `command(state, action)` | what the action asks of them, any units (the monitor fits the map; no controller gains) |
| `resolution` | smallest real change per tracked quantity (> 0): sensor resolution or noise floor; lower bound of the residual scale, so an axis that never moved in calibration cannot turn noise into a huge score |
- `JointTargetEffect(resolution)`: any joint-target `Robot`; tracked = joint positions, command = target − measured.
- `PandaOmronEffect`: tracked = end-effector position relative to the base (`agent_pos[0:3]`); command = OSC position delta + torso (action 0:3, 10); resolution 0.1 mm (the simulator has no sensor noise). Orientation and base not monitored.

## Model (`regressors`, `MonitorModel`)
- Per tick k ≥ 2w − 1 (w = `window` ticks): y = tracked change over the last w ticks; x = commands summed over this window and over the window before (2n columns), so the map absorbs controller gains and tracking lag.
- Fit: least squares `gain` (2n × m) on nominal runs; `scale` = residual std per tracked axis, ≥ `resolution`.
- Score = RMS over axes of (y − x·gain) / scale; one number per tick.
- `exercised`: regressors ever non-zero while fitting. A window that uses a never-exercised direction is not scored (`score = None`, counted in `uncovered`). Magnitude is not boxed: a larger command in an exercised direction is still predicted; saturation beyond the calibrated range shows as alarms.

## Calibration (`fit_monitor`, `calibrate_monitor.py`)
- Split by run: runs 0, 2, 4, … fit gain and scale; runs 1, 3, 5, … calibrate. Splitting by segment underestimated a new run's residual (measured 0.17 false alarms per segment at α 0.05).
- Calibration runs are cut into `segment`-tick segments (only full ones); each gives its maximum score. Threshold = ⌈(n+1)(1−α)⌉-th smallest of the n maxima → a segment of a new nominal run alarms with probability ≤ α.
- Segments of one run are correlated: exact for runs exchangeable with the calibration runs, approximate per segment (held-out test).
- α is per segment; `alarms_per_minute(hz) = α · 60 · hz / segment` states it in time.
- Refusals (`ValueError` / `p.error`): resolution ≤ 0; segment < 2 windows; fewer than 2 runs; fewer than ⌈1/α⌉ − 1 full calibration segments; a recording with real-time factor below `--min-realtime-factor` (0.98) or none recorded; recordings that mix policies.
- `Recording.conditions` (written by `realtime.py --monitor-record`): real-time factor, policy, latency, task, seed, commit policy, starved ticks.
- Recalibrate when the robot, controller, policy or task family changes.

```
uv run python realtime.py … --monitor-record runs/r1.npz          # nominal runs, no faults
uv run python calibrate_monitor.py runs/*.npz --window-s 0.5 --segment-s 5 --alpha 0.1 --out monitor.json
uv run python realtime.py … --execution-monitor monitor.json [--block-at T --block-s D]
```

## Deviation flow
1. `ExecutionMonitor.observe(TickRecord)` in the control-loop thread (`before` / `after` observations, so no tick is lost).
2. Score above threshold → one `Deviation(epoch, cmd_id, t, score, threshold)` per excursion (re-armed when the score drops back).
3. `realtime.py` `on_deviation`: `owner.post(d)` and `runner.refill_now()`.
4. Owner: about the active command → kept as `snapshot.deviation`, decision with trigger `deviation` (interaction lane when configured, else planning); about another command / epoch → logged as `stale deviation`.
5. Runner: next step asks for a chunk on the newest observation even with actions queued ([`s1-supply-and-commitment.md`](s1-supply-and-commitment.md)).
- Summary: `monitor.threshold`, `max_score`, `uncovered_ticks`, `deviations`; per tick `score` in `ticks.jsonl`.

## Evidence
Tests re-run 2026-10-22 (`uv run pytest tests/test_system1_monitor.py`): all pass. Simulated planar arm: 50 Hz, first-order lag + sensor noise, window 10 ticks, segment 50, α 0.05.
| Check | Result |
|---|---|
| Blocked at the fastest point of the motion | detected within one window, 20/20 seeds |
| Pushed without a command (5 ticks) | detected within one window, 20/20 seeds |
| Blocked while not asked to move | no alarm |
| False alarms, 10 calibrations × 8 runs, 40 held-out runs (280 segments) | mean rate within the conformal bound (α + 3 standard errors) |
| Online vs offline scores, never-exercised directions, resolution bound, refusals, JSON / npz round trip | pinned |

| Live RoboCasa (`realtime.py`, 2026-10-21) | Result |
|---|---|
| Calibration: stand-in policy 0.3 s, 6 real-time runs × 30 s, window 0.5 s, segment 5 s, α 0.1 | 15 calibration segments, ≤ 1.2 false alarms / min; seed 2004 refused (kitchen steps 54 ms > 50 ms budget) |
| Arm blocked at 12.0 s | one deviation at 12.5 s, one forced refill |
| Held-out nominal run (seed 2007) | 0 alarms, max score 2.15 vs threshold 3.93; 0 starved ticks in every run |
| Cost | none measurable: real-time factor 1.0, step p50 41 ms with and without the monitor |
| Scenario matrix, scripted S1 (step 5a) | `blocked` fails: calibration runs press into contact while grasping (residual scale 2.9 cm vs 0.3 cm in the earlier calibration); a 2 s block in a grasp phase peaks at 2.15 vs threshold 2.54 |

## Limits and open items
- Contact-blind: one threshold cannot separate intended contact from a fault; contacts inside calibration runs inflate it. Open (step 6): contact-aware calibration from a real signal (perception, force, world-model prediction); no hand tuning (decisions 2026-10-21).
- Stand-in runs exercise only the y axis → recalibrate on clean π0.5² runs. Not validated with π0.5².
- Position only on PandaOmron (no orientation, no base).
- Spurious numpy matmul `RuntimeWarning` in the RoboCasa env (numpy 2.2.5 pin; results equal `einsum`; decisions.md).
