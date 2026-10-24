# Any robot: one assembly layer, robots as executor plug-ins (spec suite index)

**Status:** SPEC, not built. Grounded on `v3-runtime` @ `0b5caa9` (2026-10-22). Not security-sensitive (no new external
surface; services and transport unchanged).

## Goal
A new robot = a new executor that supplies only **its embodiment, its backend, its S1 policy adapter and its tasks**.
Everything else (wiring the runtime, guard, loop, S1 runner, S2 workers, owner, monitor, step watcher, speech,
recording, outputs, lifecycle) is one core module used by every executor.

## Why (what is true today)
- The core is robot-agnostic by contract (`ActionSpec`, `Backend`, `Robot` + `RobotBackend`, `EffectModel`), proven on
  stand-ins, MuJoCo SO-101 (tests only) and RoboCasa PandaOmron.
- But the assembly lives in one RoboCasa script: `executors/robocasa-pi052/realtime.py` `main` (`:334–466`) builds
  every part by hand, and `summarize` (`:467–513`) + the per-tick log (`:360–373`) define the record format. A second
  robot today means copying ~500 lines.
- Generic run tooling sits in the RoboCasa harness: recording wrappers (`harness/recording.py:27–145`), invariants
  (`harness/invariants.py:23–87`), `Speaker` (`realtime.py:315`).

## Sizing
Medium-large: 1 new core package (`rojev/app`, ~5 files), 1 new executor (`executors/so101-mujoco`, ~4 files), ~8
existing files changed (`realtime.py`, `scenarios.py`, `gpu_suite.py`, `calibrate_*`, `harness/*`, architecture test),
docs. Crosses the layering boundary (new top layer). No new dependencies, no migrations.

## Slices (one PR each, in order)
| # | Spec | Delivers | Depends on |
|---|---|---|---|
| 1 | [`01-app-layer.md`](01-app-layer.md) | `rojev/app`: `Embodiment`, `SessionConfig`, `Session` (build + run + records), shared CLI flags, recording wrappers, `Speaker`; `rojev/eval/invariants.py`; tests with a stand-in robot | — |
| 2 | [`02-robocasa-on-app.md`](02-robocasa-on-app.md) | `realtime.py` reduced to RoboCasa specifics on top of `Session`; records byte-compatible; scenarios / GPU suite / calibration unchanged in behaviour | 1 |
| 3 | [`03-so101-executor.md`](03-so101-executor.md) | `executors/so101-mujoco`: the second robot through the same `Session` (MuJoCo SO-101, joint targets, stand-in S1); invariants green | 1 |

Slices 2 and 3 are independent after 1.

## Decisions locked (decision log 2026-10-22, "Any robot: …" entries)
- **D-A1** `rojev/app` is the only assembly: the top layer of the stack, depends on everything below, nothing depends
  on it; executors depend on it.
- **D-A2** `Embodiment` = what the core needs to know about a robot: `ActionSpec`, `EffectModel` (optional),
  camera names, start command; the backend is built by the executor (it needs task, seed, scene).
- **D-A3** Generic run tooling moves into the core: recording wrappers + `Speaker` → `rojev/app`; invariants →
  `rojev/eval/invariants.py` (eval may now depend on `robot`).
- **D-A4** The record format (file names, fields) is the contract between a run and every tool that reads runs
  (invariants, scorecard, calibration, GPU suite); slice 2 must keep it byte-compatible.
- **D-A5** The SO-101 proof uses a declared stand-in S1 (a scripted joint-space policy); no trained S1 exists for the
  SO-101 (spec "Not v1").

## Open decisions (user)
1. Package name: `rojev/app` (proposed) or `rojev/session`.
2. Generic CLI flags in the core (`rojev.app.cli`, proposed: every executor gets the same runtime / S2 / monitor
   flags) or per executor.
3. Real SO-101 hardware driver: out of scope here (proposed); this suite proves the sim path only.

## Invariants preserved (every slice)
- One-way dependencies (`tests/test_architecture.py`); `rojev/` never imports an executor; core never bends to a
  simulator (no RoboCasa / MuJoCo names in `rojev/app`).
- Runtime authority rules unchanged (owner = single writer of intent; runtime = what is active; guard last).
- No inline comments; deep modules (small `__init__` exports); uv only; docs in the same PR.

## Docs map (whole suite)
`docs/CODEBASE.md` (new package rows + reading order + layering line), `docs/architecture.md` (layer diagram),
`docs/features/live-harness.md` (realtime now on `Session`), new `docs/features/app-session.md`,
`docs/features/system1-map.md` / `system2-map.md` (entrypoint section → `Session`), `docs/README.md` index,
`docs/decisions.md` (D-A1…A5).
