# Slice 1: `rojev/app`, the assembly layer

**Status:** SPEC, not built. Grounded on `v3-runtime` @ `0b5caa9`. One PR.

## Public interface (`rojev/app/__init__.py`)
```python
Embodiment(name, spec: ActionSpec, cameras: tuple[str, ...], start: np.ndarray | None, effect: EffectModel | None)
SessionConfig(lease_s, decide_every_s, review_s, on_change, interaction_timeout_s, hold_recheck_s, refill_s, blend_s,
              stop_s, commit: CommitPolicy, monitor: MonitorModel | None, monitor_record: bool,
              verify_every_s, done_threshold, confirmations, realtime: bool)
Session(embodiment, backend, policy, planner, instruction, config, interaction=None, verifier=None,
        observers: tuple[Callable[[TickRecord], None], ...] = ())
  .run(seconds: float, say: list[tuple[float, str]]) -> RunRecords     lifecycle: gc freeze, start, schedule, stop
  .write(out: Path, extra_summary: dict) -> dict                      every record file + summary.json
add_session_args(parser) / session_config(args, embodiment) -> SessionConfig       shared CLI (open decision 2)
Speaker, RecordingPolicy, RecordingCommit, RecordingDecider, RecordingVerifier, InteractionCases
```
`RunRecords` = the in-memory records (owner events, ticks, S1 requests, commits, decisions, utterances, verify checks,
stats) that `write` serialises; the same object feeds `rojev.eval.scorecard` / `invariants` without a file round trip.

## Per-file change map
| File | Change | Drift risk |
|---|---|---|
| `rojev/app/__init__.py` (new) | exports above | low |
| `rojev/app/embodiment.py` (new) | `Embodiment` dataclass + validation (cameras non-empty, start matches `spec.size`) | low |
| `rojev/app/session.py` (new) | what `realtime.py:334–466` does today, robot-free: runtime (`last=start`), guard, loop (`realtime` flag), recording wrappers, runner (`on_step` → watcher claim or owner), workers, owner, monitor (if `effect` and `monitor`/`monitor_record`), watcher (if `verifier`), deviation → owner + `refill_now(cmd)`, per-tick record (`realtime.py:360–373` fields), observers, thread lifecycle and `say` schedule, gc freeze | **high**: the cyclic late-binding between owner / runner / watcher closures (`realtime.py:352–392`) must be kept exactly |
| `rojev/app/records.py` (new) | `RunRecords`, `write`, generic summary (= `summarize` `realtime.py:467–513` minus RoboCasa truth / success) | **high**: field names are the contract (D-A4) |
| `rojev/app/recording.py` (moved) | from `executors/robocasa-pi052/harness/recording.py:27–145` (all five wrappers) | medium: `InteractionCases` imports `system2.jev.chooser.situation` |
| `rojev/app/speech.py` (moved) | `Speaker` from `realtime.py:315–328` | low |
| `rojev/app/cli.py` (new) | runtime / S2 / monitor / verify / commit flags from `realtime.py:180–250` minus `--task --seed --policy --arm --viewer --video --block-*` | medium: defaults must stay identical |
| `rojev/eval/invariants.py` (moved) | from `harness/invariants.py`; `check(records, spec)` over `RunRecords` or loaded dirs | low |
| `rojev/eval/__init__.py` | export invariants | low |
| `tests/test_architecture.py` | `"app": {protocol, robot, control, transport, system1, system2, owner, eval, observability}`; `"eval"` gains `robot` | low |
| `tests/test_app_session.py` (new) | below | — |

## Invariants preserved
- `rojev/app` contains no simulator or robot names; it only sees `Embodiment`, `Backend`, `Policy`, `Decider`,
  `Verifier`.
- Thread order and shutdown order exactly as `realtime.py:398–413` (start loop → runner → workers → owner → responder →
  watcher; stop reversed).
- The deviation path posts to the owner and refills only the supplied command.

## Test plan (stand-in robot, no simulator)
- A 2-dim `ActionSpec` stand-in backend + scripted policy + `FixedCommand`: 2 s run → records have every file, ticks at
  the control rate, invariants green (`rojev.eval.invariants.check`).
- With a keyword interaction + `say=[(0.5, "wait"), (1.2, "go on")]`: hold then resume in owner events.
- Monitor on/off and verifier on/off build the right parts (no thread for absent parts).
- Records round trip: `write` then `Run.load` gives the same invariant results.
- Architecture test green.

## Done criteria
Core suite green; new tests green; `rojev/app` imports nothing from `executors/`.

## Docs in this PR
`docs/features/app-session.md` (new: contract, lifecycle, record format), `docs/CODEBASE.md` (rows for `rojev/app/*`,
`rojev/eval/invariants.py`, layering line, reading order step "app"), `docs/architecture.md` (top layer),
`docs/decisions.md` (D-A1…A4), `docs/README.md`.
