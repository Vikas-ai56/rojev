# Slice 2: the RoboCasa executor on `Session` (behaviour-preserving)

**Status:** SPEC, not built. Grounded on `v3-runtime` @ `0b5caa9`. One PR, after slice 1.

## What stays in the executor (RoboCasa specifics only)
| Kept | Where today |
|---|---|
| `RoboCasaBackend` (env, blocked fault, viewer publisher, truth reads) | `realtime.py:75–118` |
| `PANDA_OMRON` embodiment: `Embodiment("panda-omron", PANDA_OMRON, CAMS, START, PandaOmronEffect())` | `embodiment.py`, `realtime.py:63` (`START`) |
| S1 choice `remote | skills | wiggle` (`make_policy`), `WigglePolicy` | `realtime.py:134–158`, `:252–258` |
| planners `fixed | scripted | oracle | revisable | native` (`make_planner`), `FixedCommand` | `:120–132`, `:260–279` |
| interaction `continue | rules | jev` (`make_jev`, `make_interaction`) | `:281–301` |
| verifier `mjev | truth` (`make_verifier`) | `:303–313` |
| video panel + `VideoRecorder` as a `Session` observer; viewer | `:161–171`, `:370–372` |
| task-specific summary: `task_success`, `success_t`, `truth_now` | part of `summarize` `:467–513` |
| flags `--task --seed --policy --arm --subtask --latency --sample-noise --service --block-at/-s --viewer --video --trace --out`; everything else from `rojev.app.cli` | `:180–250` |

`main` becomes: parse → backend → embodiment → policy / planner / interaction / verifier → `Session(...)` →
`run(seconds, say)` → `write(out, extra_summary)` → optional trace export.

## Per-file change map
| File | Change | Drift risk |
|---|---|---|
| `executors/robocasa-pi052/realtime.py` | ~500 → ~200 lines as above | **high**: flag names, defaults and record fields are used by `scenarios.py:175`, `gpu_suite.py:46–60`, `calibrate_monitor.py`, `calibrate_jev.py`, `label_jev.py`, the scorecard |
| `executors/robocasa-pi052/harness/recording.py`, `invariants.py` | deleted (moved in slice 1); `harness/__init__.py` re-exports nothing it no longer owns | medium: imports in `scenarios.py:23`, tests |
| `executors/robocasa-pi052/scenarios.py` | `check(run, PANDA_OMRON)` from `rojev.eval.invariants` | low |
| `executors/robocasa-pi052/gpu_suite.py` | imports unchanged except moved names | low |
| `executors/robocasa-pi052/tests/test_harness.py` | imports from `rojev.app` / `rojev.eval` | low |

## Invariants preserved
- Same CLI (names, defaults); same record files and fields (D-A4); same thread order.
- Declared spoofs stay in the executor harness (skills, keyword deciders, truth verifier).

## Test plan
- Executor tests green.
- **Record-compatibility check:** one scripted-S1 run before and after (`--seed 1000 --arm revisable --interaction
  rules --say 5:wait --say 9:go\ on --seconds 12`): identical file set, identical keys per record type, identical
  owner event kind sequence.
- Scenario matrix: same pass set as the last run (17/19; `blocked`, `step-events` as before).
- `--score-only` on the new run works (scorecard reads the same records).

## Done criteria
All above green; `realtime.py` has no runtime wiring left (grep: no `ExecutionRuntime(`, `StateOwner(`,
`DecisionWorker(`, `S1Runner(` in the executor).

## Docs in this PR
`docs/features/live-harness.md` (realtime = RoboCasa specifics on `Session`), `docs/features/system1-map.md` +
`system2-map.md` §1 Entrypoint (→ `rojev/app/session.py`), `docs/CODEBASE.md` (realtime row, removed harness rows).
