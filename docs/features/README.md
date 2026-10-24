# Levers: the knobs that change the robot's behaviour (2026-10-24)

Overview of all the Features:

Current values only, read from the code. Library default = the constructor default; `realtime.py` = the live harness flag (`executors/robocasa-pi052/realtime.py`, `parse`). A changed value goes in `docs/decisions.md`; the v2 registry is in [`history/v2-levers.md`](history/v2-levers.md).

## Owner and decision lanes ([`system2-map.md`](system2-map.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Planning heartbeat | 1.0 s; 0.8 × lease when step events cover progress (`--verify`, or `--arm native\|commit` with a remote S1) | `StateOwner(decide_every_s)` · `--decide-every` | slow reaction vs wasted S2 calls |
| Proposal expiry | 2.0 s after the decision | `DecisionWorker(proposal_ttl_s)` (no flag) | acting on an old decision |
| On instruction change | `review` (keep reversible motion, freeze irreversible switches); `revoke` | `--on-change` | stop on every rephrase (revoke) |
| Review window | 1.0 s; with an interaction lane the window is `interaction_timeout_s` | `--review-s` | old goal too long vs pausing |
| Interaction deadline | 1.0 s, then the robot holds (`UNANSWERED`); a late answer is applied as decided | `--interaction-timeout` | must exceed Jev's latency in real time |
| Hold recheck | off | `--hold-recheck S` | wait budget never re-asked (off) |
| Dialogue in snapshots | last 8 turns | `StateOwner(dialogue_turns)` (no flag) | context vs prompt size |
| Commit-and-verify release | 25 s without a check | `CommitVerifyPlanner(max_commit_s)` · `--max-commit` (`--arm commit`) | stuck vs unverified step released |
| Authored-sequence budget | `chunks_per_subtask` × 2.5 s (5 / 5 / 4 chunks, p75 demo subtask) | `tasks.py` · `--arm scripted\|revisable` | baseline arms only |

## Jev interaction ([`jev-interaction.md`](jev-interaction.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Lane | none | `--interaction none\|continue\|rules\|jev` | — |
| Calibration (per question `act`, `task`) | T = 1, margin 0.2: declared, not fitted (2026-10-23 fit withdrawn) | `Calibration` · `--jev-calibration FILE` from `calibrate_jev.py --target-accuracy` (0.9) | asks too often vs acts on weak guesses |
| Asks per change | 2 | `JevInteraction(max_asks)` · `--max-asks` | — |
| Deviation dwell | 1.0 s after a step start → continue | `JevInteraction(min_dwell_s)` (no flag) | start-of-step false alarms |
| Revision target | `canonical` (trained task sentences; `--skills FILE`) with a remote S1, else `rewrite` | `--reviser` | — |
| Chooser | `onejev` (one kept connection); `keywords` stand-in | `--chooser` | — |

## Step checks and monitor ([`system2-map.md`](system2-map.md), [`execution-monitor.md`](execution-monitor.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Step checker | none; `mjev`, `onejev` or `truth` (spoof) | `--verify` | — |
| Check cadence, threshold, confirmations | 1.0 s, P(done) ≥ 0.7 on 2 consecutive checks (declared, not fitted) | `StepWatcher` · `--verify-every --done-threshold --confirmations` | false "done" vs late release |
| Execution monitor | off; on with `--execution-monitor M.json` | `ExecutionMonitor` | blocked arm unnoticed (off); contact-blind |
| Monitor calibration | window 0.5 s, segment 5 s, α 0.1, runs below real-time factor 0.98 refused | `calibrate_monitor.py --window-s --segment-s --alpha --min-realtime-factor` | alarm rate vs sensitivity |

## Harness and evaluation ([`live-harness.md`](live-harness.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Time mode | real time; sim evaluations run `--lockstep` (decision 2026-10-23) | `--lockstep` | lockstep hides S1 / S2 latency |
| Frame check | every 1.0 s; noise > 25 (neighbour difference) or blank < 3 (std) → run not scored | `harness/frame_check.py` `NOISE`, `BLANK` · `--frame-check-every` | thresholds far outside 963 real frames |

## S1 supply and commitment ([`s1-supply-and-commitment.md`](s1-supply-and-commitment.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Refill lead (lower bound) | 1.0 s | `S1Runner(refill_s)` (no library default) · `--refill` | low: gap before the first measured latency · high: acts on older frames |
| Lead time, commitment floor | derived: lead max(`refill_s`, L + dt), floor ⌈2L/dt⌉ + 1, L = max of the last 20 latencies | `S1Runner.lead_s`, `floor`, `latency_window` 20 (no flag) | not configured |
| Request deadline | one chunk duration (horizon × dt); none with `--lockstep` | `S1Runner.deadline_s` | stalled request abandoned (`timeouts`) |
| Commitment policy | `fixed` (whole chunk); `phase`: first slow point after the floor, `slow_fraction` 0.3 | `make_commit_policy` · `--commit-policy` (no flag for `slow_fraction`, `max_commit`) | late reaction near contact (fixed) |
| Speculative predict | off; on: K = `samples` 3, `tolerance` 0.5 × median speed, `smooth` 3 | `ConsensusCommit` · `--speculative-predict --samples --tolerance` | K draws per request (one batch) vs committing to an outlier chunk |
| XR-1 frame history | 4 frames, 2 ticks apart | `realtime.py` `XR1_HISTORY` (= `xr1_inputs.LENGTH, INTERVAL`) | model input contract; not tunable |

## S1 service and wire ([`remote-models.md`](remote-models.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Frame codec | `webp90` | `wire.DEFAULT_CODEC`; `"codec"` key in the service JSON entry | request size vs fidelity (within sampling noise, 8 frames) |
| Transport | HTTP/1.1 keep-alive (`http://` url from `gpu_setup.sh`); `h3://` + `cert_pem` by hand | service JSON `url` → `open_transport` | aioquic not better on the WAN run |
| π0.5² planner memory | off | `serve.py --planner-memory` | decision 2026-10-22 kept it off; its A/B (E19) is withdrawn (corrupted frames), not re-run |
| π0.5² real-time chunking | off | `serve.py --rtc H` | uses the continuation; latency not measured |

## Execution and safety ([`v3-runtime.md`](v3-runtime.md))
| Lever | Default | Set at | Trade-off |
|---|---|---|---|
| Chunk blend | 0.2 s (quintic, continuous dims) | `ExecutionRuntime(blend_s)` · `--blend` | jerk at switches vs lag |
| Decelerating hold | library 0 (instant); `realtime.py` 0.3 s | `ExecutionRuntime(stop_s)` · `--stop-s` | jerky vs long stops |
| Command lease | 3.0 s | `StateOwner(lease_s)` · `--lease` | robot runs on a command nobody renews |
| Starvation watchdog | 0.5 s → hold | `Guard.watchdog_s` (no flag) | coasting on no chunks |
| Owner heartbeat timeout | 2.0 s → hold | `Guard.owner_timeout_s` (no flag) | acting with no supervisor |
| Transition to a new command | `immediate` (blended at the new chunk); `pause_hold` | `StateOwner(transition)` (no flag) | slow vs unsafe switches |
| Robot action spec | per robot: bounds, hold value, switches, irreversible dims (PandaOmron: gripper) | `executors/*/embodiment.py` | wrong hold, gripper released on wait |

## How to change one
- Attribute the failure to one stage first (trace, oracle arm), then change config before swapping a model, a model before training.
