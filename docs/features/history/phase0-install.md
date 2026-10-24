# Phase 0 install gate G0 (historical)

Historical (v1 SO-101 plan, superseded 2026-10-16 by the v2 pivot (RoboCasa) and 2026-10-19 by the OneJev 1.6 4B pin; see [`../gpu-setup.md`](../gpu-setup.md), [`../robot-interface.md`](../robot-interface.md)).

## What it was (2026-10-16, passed)
- Mac toolchain for the SO-101 plan: Homebrew Python 3.12, uv, llama.cpp, MuJoCo 3.x, Menagerie SO-101 (`config/menagerie.commit`), qev from `vendor/OneJev` (`config/onejev.commit`); versions in `config/toolchain.txt`, exact packages in `uv.lock`.
- OneJev 1.5 0.8B Q8_0 GGUF served locally by `qev serve` on llama.cpp (~1.3 GB resident, 195–390 ms per request).
- Gate checks now live in `tests/test_install.py` (scene compiles, actuator inventory, stable physics, offscreen render; live OneJev test runs only when a qev server answers at `$QEV_BASE_URL`).

## What replaced it
- OneJev 1.6 4B (`config/onejev_weights.txt`) served by qev on the GPU box ([`../gpu-setup.md`](../gpu-setup.md)); the local 0.8B path survives only in `tests/test_install.py` and the `Makefile` `qev` target.
- Image grounding check (box shown vs hidden: P(green box) 0.984 vs 0.140) = experiment E1 in [`../../paper/experiments.md`](../../paper/experiments.md).

## Lessons that still apply
- qev binds `0.0.0.0` and has no auth: always pass `--host 127.0.0.1` (kept in the OneJev GPU hook).
- No `calibration.json` upstream: OneJev probabilities are uncalibrated until measured (Jev calibration, [`../jev-interaction.md`](../jev-interaction.md)).
- Ask evidence questions without the instruction in the state: with the box hidden, P(approach) stayed 0.658 because the instruction pulled it (E2).
- Fresh machine: clone `vendor/` at the pinned commits, `uv sync`; the MuJoCo passive viewer needs `uv run mjpython`.
