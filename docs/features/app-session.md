# Shared assembly: `rojev/app` (`Session`)

One assembly of the runtime for any robot and any models. A robot plug-in supplies its `Embodiment` and backend; model
plug-ins supply S1, the planning decider, and optionally the interaction decider and the done checker. Nothing in
`rojev/app` names a robot, simulator or model. Decisions: `decisions.md` 2026-10-22 (D-A1…A5), 2026-11-09, 2026-11-10.
Whole system: [`pipeline.md`](pipeline.md).

## Contract
| Piece | What it is |
|---|---|
| `Embodiment(name, spec, cameras, start, effect)` | What the core must know about a robot: `ActionSpec`, camera names, start command (holds it still), optional `EffectModel` |
| `SessionConfig` | Every runtime knob with its measured default; `from_mapping` refuses unknown keys |
| `load_config(path)` | YAML: `runtime` section → `SessionConfig`; other sections belong to plug-ins |
| `build({factory, args})` | A plug-in from configuration: `package.module:callable` called with `args` |
| `Session(embodiment, backend, policy, planner, instruction, config, out, interaction, verifier, video, observers)` | Builds runtime, guard, loop, S1 runner, a worker per lane, owner (event-driven planning, lease, memory, episodic log), step watcher + `SwitchCues` when a verifier is given |
| `Session.run(seconds, say, stop_when)` → `RunResult(ended, sim_s, wall_s, ticks, complete, memory, out)` | Ends on `seconds` of robot time, COMPLETE, `stop_when()`, or an error (records still written) |

## Time modes
- **Lockstep** (default; how benchmarks are scored): one thread alternates owner (events, then tick), one S1 supply
  step and one control tick; no tick while a decision lane or the done checker is busy or an answer is unhandled.
- **Real time:** every part on its own thread against the robot's clock.

## Users
- LIBERO: one run per episode, S1 alone (no S2 steps yet); GR00T on the GPU: 49/50 on LIBERO-goal (E33, lockstep).
  Real time (`config/runs/libero-realtime.yaml`) runs on the Mac with the fake simulator; on the GPU not yet.
- Tests: a stand-in robot with the step planner and a done checker.

## Settings to watch
- No refill setting: how many actions of each chunk run is the model's own `execute_steps`, sent with each chunk
  (GR00T on LIBERO: 8 of 16). In lockstep the robot runs exactly that many; a model without it runs whole chunks.
- `watchdog_s` (10 s) suits planners that wait for events; the scripted and oracle baselines need about 1 s.

## Records (under `out`)
`config.json`, `owner.jsonl`, `ticks.jsonl`, `decisions.jsonl`, `s1_requests.jsonl`, `utterances.jsonl`,
`episodes.jsonl`, `summary.json` (ended, timing, starved ticks, violations, error, memory, commands, ACK counts,
liveness lapses, S1 / lane / verifier stats), `view.mp4` with `video=True` (cameras + panel: time, command, latest
decision, steps done; frames buffered, encoded after the run).

## Not here
- Model adapters and robot plug-ins (executors, services).
- Motion monitor, partial-chunk commitment, frame check, done-check logs: still only in the RoboCasa app.
- The legacy RoboCasa app (`executors/robocasa-pi052/realtime.py`) keeps its own assembly until it moves onto `Session`.
- Model-delay emulation in lockstep (RoboCasa harness only).

## Tests
`tests/test_app_session.py` (6): a stepped stand-in robot; lockstep records and rate; a step planner with a cue-driven
done checker finishes a two-step task; acknowledgement and `stop_when`; a failing backend; video; plug-ins and
settings. The LIBERO tests also run a still policy through the real `Session`.
