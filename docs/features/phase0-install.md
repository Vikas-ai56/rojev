# Phase 0 — install (G0)

Plan reference: §1 and gate **G0 — install** (viewer, actuator inventory, PNG, OneJev Choice response).
**Status: G0 passed on 2026-10-16** (MacBook Air M5 `Mac17,3`, 16 GB, macOS 26.6.2).

## Installed

| Component | Version / pin | How |
|---|---|---|
| Xcode CLT | pre-installed (`/Library/Developer/CommandLineTools`) | — |
| Homebrew Python | 3.12.15 | `brew install python@3.12` |
| Git | 2.56.0 (Homebrew) | `brew install git` |
| llama.cpp | 0.6.0, build 11429 (`d81235049`) | `brew install llama.cpp` |
| uv | 0.12.2 (Homebrew, pre-installed) | — |
| `.venv` | Python 3.12.15 (Homebrew interpreter) | `uv sync --python-preference only-system` |
| MuJoCo | 3.16.0 (plan range `>=3.1.3,<4`) | uv (`pyproject.toml`, exact pins in `uv.lock`) |
| numpy / pillow / pytest / pyyaml | 2.5.4 / 12.3.0 / 9.1.1 / 6.0.3 | uv (pytest in the `dev` group) |
| MuJoCo Menagerie SO-101 | `config/menagerie.commit` | sparse clone of `robotstudio_so101` only |
| OneJev `qev` client | 0.1.2 @ `config/onejev.commit` | editable path source in `[tool.uv.sources]` (no PyTorch; GGUF path) |
| OneJev **1.5** 0.8B Q8_0 GGUF + f16 mmproj | `config/onejev_weights.txt` (snapshot + sha256) | first `qev serve` (HF cache `~/.cache/huggingface/hub`) |

Plan deviation: OneJev 1.5 instead of 1.0 — see `docs/decisions.md`.

## Verified

Simulator / asset
- `scene_box.xml` compiles: nq 13, nv 12, nu 6, timestep 0.005, one camera (`wrist_cam`).
- Actuator inventory (index → joint, ctrlrange): 0 `shoulder_pan` ±1.91986 · 1 `shoulder_lift` ±1.74533 · 2 `elbow_flex` ±1.69 · 3 `wrist_flex` ±1.65806 · 4 `wrist_roll` [-2.74385, 2.84121] · 5 `gripper` [-0.17453, 1.74533]. Each actuator drives the same-named joint.
- Green box default position (0.5, 0, 0.03) — Phase 1 must not assume this is reachable.
- `python -m mujoco.viewer --mjcf=…/scene_box.xml` stays up; the `mjpython` passive viewer runs and steps physics.
- The plan's render smoke script wrote a 640×480 PNG showing arm + box (checked by eye). The same check now lives in `test_offscreen_render_shows_arm_and_box`; the script was removed.
- 2 s of physics from the `pickup` keyframe stays finite with no MuJoCo warnings.

OneJev
- Server start-up: weights + mmproj + tokenizer downloaded; `vision=True`; qev's GGUF-vs-tokenizer token-equality check passed.
- The plan's §1.4 OneJev smoke request → `{'approach_box': 0.934, 'observe': 0.031, 'stop': 0.034}`. Now `test_onejev_returns_probability_for_every_option`; the script was removed.
- Image grounding check (same request, rendered frame with box shown vs. hidden):

  | Frame | P(green box visible) | P(robot arm visible) | P(approach_box) |
  |---|---|---|---|
  | box shown | 0.984 | 0.983 | 0.934 |
  | box hidden | 0.140 | 0.986 | 0.658 |

- Same request twice → identical probabilities (deterministic).
- Latency 195–390 ms per request (one image, 1–3 questions) after warm-up.
- Memory at steady state: llama-server 1079 MB phys footprint + qev 245 MB ≈ **1.3 GB**; system memory free 31% with normal desktop load. This is a `footprint` reading after load, not a peak trace through loading.

Re-run: `uv run pytest tests/test_install.py -v` → 10 pass with a server up (9 + 1 skip without one). Re-verified after the pip → uv switch: 10/10, smoke output identical.

Fresh machine: clone `vendor/` at the pinned commits (plan §1.2 git commands), then `uv sync`. Use `uv run mjpython …` for the passive viewer.

## Running the selector server

```bash
uv run qev serve --gguf mradermacher/OneJev1.5-0.8B-GGUF:Q8_0 --host 127.0.0.1 --port 8000
```

Pass `--host 127.0.0.1`: by default qev binds `0.0.0.0`, which exposes the API to the local network. Weights are cached, so later starts take about 10 s.

## Known gaps carried into Phase 1+

- No `calibration.json` upstream → identity calibration. Treat probabilities as uncalibrated until Phase 4 measures Brier/calibration.
- With the box hidden, `approach_box` still got 0.658. The explanation first given here ("the instruction pulls the action") was untested. It was later measured: a neutral instruction lowers both P(approach) and P(box visible). The measurements are in `observability.md`. Evidence questions must not share the instruction-laden state.
- `qev` loads the f16 mmproj (205 MB), not the Q8 projector (116 MB) the plan quotes.
