# Codebase map

`rojev` SDK for the simulation-first visual skill agent (plan: `docs/plan/visual_skill_agent_implementation_plan.md`). Current stage: **v2 G0** (frozen RoboCasa/SmolVLA executor). The SO-101 track (Phase 1 M1 done: `rojev.robot`) is paused.

Rule: `rojev/` holds only reusable SDK code. Demos and scripts go in `examples/`, checks go in `tests/`.

## SDK — `rojev/`

| Path | What it is |
|---|---|
| `rojev/__init__.py` | Package version |
| `rojev/robot/` | Robot interface (sim now, SO-101 later): `base.py` (`Robot`, `Clock`, `Rate`), `types.py` (timestamped `JointState`, `CameraFrame` + `CameraInfo`, `JointCommand`, `CommandResult`), `limits.py` (`JointLimits`, `SafetyFilter`), `mujoco_robot.py` (`MujocoRobot`, `SimClock`, `SimGroundTruth`, `camera_info`), `mock.py`, `so101.py` (factory from `config/so101.yaml`). See `docs/features/robot-interface.md` |
| `rojev/sim.py` | MuJoCo world utilities for the backend and evaluators: `compose` (world XML + vendor robot via MjSpec, robot solver options kept), `contacts_with`, `visible_pixels` |
| `rojev/onejev.py` | `system_one()`: OneJev call recorded as an OpenInference LLM span (exact image sent, state, questions, answers, debug logits, `onejev.*` numeric attributes); passes qev options such as `debias` |
| `rojev/tracing.py` | `setup(project)`: OpenTelemetry → local Phoenix; raises the image-redaction limit |

Still to come, each added only when its logic exists (rules: `docs/architecture.md`): `rojev/skills/` (base class + registry, every skill lives here),
world state / planning scene (`world_version`), feasibility gates, motion contract (`MotionRequest` → `MotionSolver`), selector, validator, evaluator.

## Around the SDK

| Path | What it is |
|---|---|
| `examples/scripted_trajectory.py` | Fixed SO-101 joint trajectory through `rojev.robot` at 30 Hz, traced per segment (safety clamps, errors, floor contacts); optional observe-only OneJev checkpoints with a ground-truth evaluator; failures marked ERROR; viewer runs in real time |
| `tests/test_install.py` | G0 install gate (independent of `rojev`) |
| `tests/test_sim.py` | `rojev.sim` unit tests |
| `tests/test_robot_contract.py` | Contract every robot backend must pass (mujoco + mock) |
| `Makefile` | One-line commands: `phoenix`, `qev`, `sim`, `sim-viewer`, `test` |
| `README.md` | Setup + run + module table |
| `pyproject.toml` / `uv.lock` / `.python-version` | uv project `rojev` (uv_build, flat layout). `qev` is an editable path source at `vendor/OneJev`. `arize-phoenix` server is in the `dev` group |
| `config/menagerie.commit`, `config/onejev.commit` | Pinned vendor sources |
| `config/toolchain.txt` | Homebrew Python / llama.cpp versions and host |
| `config/onejev_weights.txt` | OneJev 1.5 0.8B GGUF + tokenizer snapshot hashes, sha256 of weight files |
| `docs/plan/` | Implementation plan (md + pdf) |
| `docs/decisions.md` | Decisions log |
| **`docs/paper/`** | **Single place for the paper:** README (outline/status), related-work, method, experiments (all numbers + runs), appendix-engineering, references.bib, citations.md, sources/ (PDFs) |
| `rojev/supervisor/` | Supervisor core: `candidates.py` deterministic candidate builder (transition graph, BM25 similar tasks, object overlap, canonical wording) |
| `docs/features/v2-engineering-notes.md` | Build notes moved out of the paper: code reuse + licences, MJev on M5, platform notes |
| `executors/robocasa-pi052/drive.py` | π0.5² RoboCasa driver on LeRobot's language runtime: success + subtask log; subtask source own/direct (OneJev hook) |
| `executors/robocasa-pi052/kaggle/` | Kaggle 2×T4 runner: `kaggle_run.py` (staged setup + runs), `push.py` (bundle/push/status/pull), `kernel-metadata.json` |
| `playground/candidates/coverage.py` | E9 coverage@K evaluation (held-out episodes / held-out tasks) |
| `playground/candidates/templates.py`, `coverage_templates.py` | E9b template + slot-filling builder and its coverage run (experiment, not adopted) |
| `data/vocab/pi052_robocasa_sequences.jsonl` | Ordered subtask timeline per π0.5² training episode (32,043) |
| `tests/test_candidates.py` | Candidate builder unit tests |
| `playground/archive/` | Archived experiment scripts referenced by `docs/paper/experiments.md` |
| `docs/architecture.md` | Binding architecture rules (from the plan + motion_planner.md) |
| `docs/features/phase0-install.md` | What G0 installed and how it was verified |
| `docs/features/observability.md` | Phoenix usage, span contents, measured results |
| `scenes/pick_place.xml` | World only (floor, overhead camera, box, `robot_mount`); robot attached by `sim.compose` |
| `config/so101.yaml` | Robot values the MJCF lacks (control rate, max joint velocity), each with its source |
| `docs/features/robot-interface.md` | M1: robot interface, limits, cameras, what was verified |
| `models/`, `data/`, `results/` | Plan layout dirs |
| `data/phoenix/` (gitignored) | Phoenix trace store (SQLite) |
| `executors/robocasa/` | Separate uv project for the frozen executor (RoboCasa + robosuite + LeRobot SmolVLA); conflicting pins kept out of the rojev env |
| `config/lerobot.commit`, `config/robocasa.commit`, `config/robosuite.commit` | Pinned executor sources |
| `docs/onejev_learned_policy_plan_v2.md` | Active research plan (pivot 2026-10-16) |
| `playground/timing/` | Experiments: decision-timing approaches (T1–T3 × S1–S4), chosen by results |
| `executors/robocasa/checkpoint.py` | Reads camera mapping and training instruction vocabulary from a checkpoint |
| `executors/robocasa/rollout.py` | Frozen-policy rollouts with an explicit instruction (mirrors lerobot-eval otherwise) |
| `executors/robocasa/capability_table.py` | G0-C runner (lerobot-eval per task; superseded for instructions by rollout.py) |
| `executors/robocasa-pi052/` | Separate uv env for π0.5² (LeRobot steerable branch); `extract_subtask_vocab.py` |
| `data/vocab/pi052_robocasa_subtasks.json` | π0.5² training subtask vocabulary (21,326 strings, episode counts, tasks) |
| `docs/features/v2-pi052-executor.md` | π0.5² executor: setup, steering point, subtask vocabulary |
| `docs/features/v2-g0-executor.md` | v2 G0 progress: executor env, checkpoint facts, pending downloads |
| `vendor/` (gitignored) | `mujoco_menagerie` (sparse), `OneJev`, `lerobot`, `robocasa`, `robosuite` |
| `.venv/` (gitignored) | Built by `uv sync` |
