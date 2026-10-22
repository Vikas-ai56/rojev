# rojev

SDK for a simulation-first visual skill agent: MuJoCo SO-101 simulation and a OneJev (GGUF/llama.cpp) skill selector, traced to a local Phoenix server. Plan: `docs/plan/`. File map: `docs/CODEBASE.md`.

## Setup (macOS, Apple Silicon)

```bash
brew install python@3.12 llama.cpp uv
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git vendor/mujoco_menagerie
git -C vendor/mujoco_menagerie sparse-checkout set robotstudio_so101
git -C vendor/mujoco_menagerie checkout $(cat config/menagerie.commit)
git clone https://github.com/OmniJev/OneJev.git vendor/OneJev && git -C vendor/OneJev checkout $(cat config/onejev.commit)
uv sync
make test
```

## Run

| Command | What |
|---|---|
| `make phoenix` | Trace UI at http://localhost:6006 |
| `make qev` | OneJev selector at http://127.0.0.1:8000 (downloads ~1.3 GB of weights the first time) |
| `make sim-viewer` | Scripted SO-101 trajectory in the MuJoCo viewer |
| `make sim ARGS="--onejev"` | Same, headless; OneJev observes each checkpoint frame |

## Package

`rojev` is split along the architecture (map: `docs/CODEBASE.md`; layering enforced by `tests/test_architecture.py`):

| Package | Purpose |
|---|---|
| `rojev.protocol` | Shared contracts: observations, commands, decisions |
| `rojev.robot` | Robot interface for sim and hardware (timestamped joints/cameras, clocks, hard limits, `ActionSpec`); `rojev.robot.sim` = MuJoCo backend |
| `rojev.control` | Real-time layer: control loop, execution runtime, guard, robot adapter |
| `rojev.system1` | Visuomotor policy contract, remote client, chunk runner |
| `rojev.system2` | Decider contract, decision worker, OneJev / MJev clients |
| `rojev.owner` | State owner: the single writer of intent |
| `rojev.eval` | Baseline deciders (evaluation only) |
| `rojev.observability` | OpenTelemetry → Phoenix |
| `rojev.legacy_v2` | Quarantined v2 supervisor (removed after roadmap T8) |
