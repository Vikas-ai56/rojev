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

| Module | Purpose |
|---|---|
| `rojev.robot` | Robot interface for sim and hardware: timestamped joint/camera data, clock + fixed-rate loop, hard joint limits + safety filter, MuJoCo and mock backends |
| `rojev.sim` | MuJoCo world utilities: scene composition, contacts, segmentation ground truth |
| `rojev.onejev` | OneJev System One call, recorded as a span with the exact image sent |
| `rojev.tracing` | OpenTelemetry → Phoenix setup |
