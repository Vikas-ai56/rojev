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
| `docs/research/hierarchical-timing-and-mjev.md` | Research notes: sync vs async in hierarchical VLA systems, reusable code, MJev on M5 |
| `docs/features/v2-g0-executor.md` | v2 G0 progress: executor env, checkpoint facts, pending downloads |
| `vendor/` (gitignored) | `mujoco_menagerie` (sparse), `OneJev`, `lerobot`, `robocasa`, `robosuite` |
| `.venv/` (gitignored) | Built by `uv sync` |
