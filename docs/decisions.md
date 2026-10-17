# Decisions

## [2026-10-16] Project root is `~/rojev`, not `~/visual-skill-agent`
**Decision:** Phase 0 layout from the plan (§1.2, §4) lives in this repo (`/Users/kvikas/rojev`) instead of creating `~/visual-skill-agent`.
**Why:** The repo already exists with the plan under `docs/plan/`; a second directory would split code from its plan and decision log.
**Rejected alternatives:** Creating `~/visual-skill-agent` verbatim as in the plan.
**Risks / edge cases found:** All plan commands use relative paths (`vendor/...`, `data/raw/...`), so they must be run from the repo root.

## [2026-10-16] Vendor sources and venv are not committed; reproducibility comes from pins
**Decision:** `.venv/`, `vendor/`, `models/`, `data/raw/`, `data/episodes/`, `results/runs/` are gitignored. Reproducibility is carried by `config/menagerie.commit`, `config/onejev.commit`, `uv.lock` (was `config/python.freeze.txt`), `config/toolchain.txt`.
**Why:** Plan §4 marks `.venv/` local-only; vendored clones and model weights are large and re-fetchable at a pinned commit.
**Rejected alternatives:** Git submodules for `vendor/` (heavier, and the Menagerie clone is sparse); committing vendor files.
**Risks / edge cases found:** `qev` is an editable path source; restoring requires re-cloning OneJev at `config/onejev.commit` first.

## [2026-10-16] uv replaces pip for the Python environment
**Decision:** `pyproject.toml` + `uv.lock`, environment built with `uv sync`, commands run with `uv run`.
**Why:** User preference; the lockfile gives exact, reproducible resolution.
**Rejected alternatives:** pip venv + `pip freeze`.
**Risks / edge cases found:** The uv lock resolved the same versions as the pip set that passed G0. G0 was re-run under uv and passed 10/10.

## [2026-10-16] G0 is a pytest gate, not only one-off scripts
**Decision:** `tests/test_install.py` encodes the G0 checks (MuJoCo version range, no `mujoco-py`, real SO-101 meshes, compiled dimensions, recorded actuator inventory, keyframe physics stability, render shows arm + box by pixel test, `llama-server` present, `qev` client/CLI). The live OneJev Choice test auto-skips unless a `qev` server answers at `$QEV_BASE_URL`.
**Why:** Plan says "never assume the six control indices correspond to a guessed joint order" — the inventory is now asserted, so a Menagerie update that reorders actuators fails loudly.
**Rejected alternatives:** Only running the plan's ad-hoc snippets (not repeatable).
**Risks / edge cases found:** The render check uses colour thresholds on the default free camera (green ≈1.8k px vs 500 threshold; 0 px with box hidden). A Menagerie change to camera/lighting/colours would need the thresholds re-checked, not loosened blindly.

## [2026-10-16] Use OneJev 1.5 0.8B Q8_0 instead of the plan's OneJev 1.0
**Decision:** Selector weights are `mradermacher/OneJev1.5-0.8B-GGUF:Q8_0` (tokenizer `OmniJev/OneJev1.5-0.8B`), chosen by the user at the weights checkpoint. Supersedes the plan's `mradermacher/OneJev-0.8B-GGUF:Q8_0`.
**Why:** Same footprint (1082 MB Q8_0), same `qev` serving path, and a newer release with substantially higher reported decision scores at 0.8B (upstream Decision Index 42.71 vs 22.23 — upstream's own benchmark, not verified here).
**Rejected alternatives:** 1.0 as in plan; installing both.
**Risks / edge cases found:** Any Phase 3/4 numbers are for the 1.5 Q8 policy and must be reported as such; 1.0 can still be added later as a separate baseline policy. 1.5 was three days old at selection.

## [2026-10-16] Serve OneJev on loopback only
**Decision:** Start the selector with `--host 127.0.0.1`. qev's default is `0.0.0.0`.
**Why:** The simulator and the client run on the same Mac. Binding all interfaces exposes an unauthenticated inference API to the local network.
**Rejected alternatives:** Keep qev's default.
**Risks / edge cases found:** The G0 validation server ran with the default `0.0.0.0` bind, and this was noticed afterwards. It was up for a few minutes and then stopped.

## [2026-10-16] OneJev weight download held for explicit approval (resolved: see entry above)
**Decision:** Everything up to the GGUF weights is installed and verified; `qev serve --gguf …` (which downloads weights) has not been run.
**Why:** User asked to stop before the OneJev weights. Also found drift: the plan pins `mradermacher/OneJev-0.8B-GGUF:Q8_0` (OneJev 1.0), but upstream released OneJev 1.5 on 2026-10-13 (`mradermacher/OneJev1.5-0.8B-GGUF`, same 1082 MB Q8_0 size; upstream reports 0.8B Decision Index 42.71 vs 22.23 for 1.0).
**Rejected alternatives:** Silently swapping to 1.5 (would deviate from the plan's evaluated policy without sign-off).
**Risks / edge cases found:** (a) Neither `OmniJev/OneJev-0.8B` nor `OmniJev/OneJev1.5-0.8B` ships `calibration.json`, so the server runs with identity calibration. (b) `qev`'s `resolve_gguf` prefers the **f16** mmproj (205 MB) over the Q8 one (116 MB) the plan quotes. (c) Weights land in `~/.cache/huggingface/hub`, not `models/`.

## [2026-10-16] Keep HTTP/1.1 on loopback for the selector; no HTTP/3
**Decision:** The client calls the server over loopback with plain HTTP (urllib). `httpx2` stays as a dependency of `huggingface_hub`, used only for downloads.
**Why:** Measured transport ≈0.4 ms vs ≈73 ms median decision (≈0.6%); inference dominates.
**Rejected alternatives:** HTTP/3/QUIC. Its gains come from lossy, high-RTT networks, and neither uvicorn nor llama-server supports it.
**Risks / edge cases found:** Re-measure if the selector moves off-box.

## [2026-10-16] Phoenix (local) for tracing, via OpenTelemetry
**Decision:** Arize Phoenix in the project env. Spans are emitted with standard OTel calls, and traces are stored in `data/phoenix/`.
**Why:** Free and local, with no Docker, and no data leaves the Mac. Because the spans are standard OTel, they could later go to Langfuse or LangSmith instead.
**Rejected alternatives:** LangSmith (free-tier limits, cloud only); Langfuse self-hosted (Docker stack too heavy for 16 GB).
**Risks / edge cases found:** The gRPC :4317 port binds all interfaces and can't be configured. Images over 32k chars are redacted unless the limit is raised.

## [2026-10-16] Repo is the `rojev` SDK; `src/` removed
**Decision:** `rojev/` (flat layout, uv_build) holds only reusable code. Demos go in `examples/`. The G0 smoke scripts were dropped because tests cover them. The Phoenix server moved to the dev group.
**Why:** User wants the core package kept clean for the skills library and backbone.
**Rejected alternatives:** `src/rojev/` src-layout (user asked for `rojev/`); stub modules for future phases (dead code).
**Risks / edge cases found:** `rojev.sim` defaults to the repo's `vendor/` path, so a non-editable install needs `ROJEV_SO101_DIR`.

## [2026-10-16] Failures are explicit span errors, judged against simulator ground truth
**Decision:** A separate evaluator span compares answers to segmentation ground truth and sets ERROR plus the reason, which propagates to checkpoint and episode. Motion segments fail on joint error > 0.01 rad or floor contact.
**Why:** The user asked that a wrong decision never pass silently. Ground truth comes from the simulator, not from the `--hide-box` flag.
**Rejected alternatives:** Phoenix annotations added after export (more moving parts); visibility taken from the CLI flag (misses occlusion).
**Risks / edge cases found:** The "approach iff visible" rule is specific to this demo's instruction and is not a general skill validator.

## [2026-10-16] Architecture rules from the plan + motion_planner.md (after discussion)
**Decision:** Follow `docs/architecture.md`: a hardware-shaped backend interface (sim now, SO-101 later), layered world state → OneJev → gates → pluggable planner → skills.
**Why:** User decisions on scope, DoF, gating and planners. The hidden-box failure is what a feasibility gate catches.
**Rejected alternatives:** Skills doing their own geometry; a sim-only data path; one hard-wired IK solver.
**Risks / edge cases found:** 5-DoF planar chain: a target is position + approach pitch (+ optional roll); yaw is set by the target's direction from the base. motion_planner.md is untested advice.

## [2026-10-16] All skills in `rojev/skills/` behind one base class
**Decision:** One extensible `Skill` base class plus a registry in `rojev/skills/`, built when Phase 1 needs it.
**Why:** User requirement; it keeps the skill contract in one place.
**Rejected alternatives:** Skills spread across controllers or examples.
**Risks / edge cases found:** The base class must not hide perception or planning (rule 4).

## [2026-10-16] Own Robot interface; gym only as a wrapper on top
**Decision:** `rojev.robot.Robot` (LeRobot-shaped API, ros2_control-style read/write, injected clock) with MuJoCo and mock backends; a LeRobot adapter comes later.
**Why:** No MuJoCo SO-101 implements LeRobot's Robot API (gym_hil is a Franka; LeIsaac needs Isaac/NVIDIA), and a gym env merges robot, world and task.
**Rejected alternatives:** gym env as the base layer; depending on LeRobot internals (its motor bus is being rewritten on main).
**Risks / edge cases found:** Canonical units are MJCF joint radians, so real units need a per-joint table; the velocity limit is an unverified retailer value.

## [2026-10-16] Hard joint limits from the model, enforced outside planners
**Decision:** `JointLimits` (range ∩ ctrlrange, forcerange, sourced velocity) is checked on requests, on trajectories and in `SafetyFilter`; every clamp is reported.
**Why:** User requirement: planner cost terms (mink etc.) are not guarantees.
**Rejected alternatives:** Trusting the solver's constraint handling.
**Risks / edge cases found:** The MJCF has no velocity limits; joint ranges currently equal ctrlrange exactly.

## [2026-10-16] Scenes composed with MjSpec; robot solver options carried over
**Decision:** `scenes/*.xml` hold only the world; `sim.compose` attaches the vendor robot and copies its timestep, integrator, cone, impratio and iterations.
**Why:** `<include>` from another directory breaks mesh paths, and `attach` silently keeps the parent's default solver options.
**Rejected alternatives:** Copying or editing vendor XML; `<include>` with absolute paths.
**Risks / edge cases found:** Any new robot option must be added to `ROBOT_OPTIONS`.

## [2026-10-16] Pivot to v2: OneJev as a probabilistic supervisor over a frozen learned policy
**Decision:** Follow `docs/onejev_learned_policy_plan_v2.md`: RoboCasa + frozen `lerobot/smolvla_robocasa` as executor, OneJev selecting subgoals. The headline is probabilistic (selective) supervision from OneJev's typed readout. The SO-101 track (M2–M5: world state, mink/OMPL, skills) is paused.
**Why:** The research question is supervisor value with the executor held fixed; a hand-built motion stack would confound it.
**Rejected alternatives:** Two parallel tracks (executor-agnostic supervisor); continuing the SO-101 planner work.
**Risks / edge cases found:** A weak executor (card reports no success rates; light 20k-step fine-tune) could leave no room to improve; 2.5 s open-loop chunks; Mac feasibility untested, so G0 starts with a Mac test.

## [2026-10-17] Plug-in VLA executors on RoboCasa: SmolVLA first, π0.5 second
**Decision:** The executor is an interface (`reset / set_subgoal / next_action / cancel`) with backends per policy. It starts with `lerobot/smolvla_robocasa` and `lerobot/pi05_robocasa`, both through LeRobot's PyTorch API in the same RoboCasa env. An OpenPi websocket client is a later backend.
**Why:** The user wants any VLA to plug in without touching the core; both checkpoints share cameras, 16-D state and 12-D action, so swapping them is a config change.
**Rejected alternatives:** Moving to LIBERO for OpenPi checkpoints (changes the environment, confounding the comparison); OpenPi's JAX runtime (NVIDIA-only).
**Risks / edge cases found:** π0.5 is 3B (~6.6 GB bf16, estimated) and may not fit beside MJev on the 16 GB Mac.

## [2026-10-17] MJev-Spatial answers at every decision point, after OneJev
**Decision:** At each decision point (subgoal boundary, failure or retry, never per control step), OneJev answers first and MJev-Spatial-4B second, every time.
**Why:** Real scenes are complex and we do no direct CV; MJev supplies the spatial evidence (relations, distances, points).
**Rejected alternatives:** Calling MJev only on OneJev uncertainty (user: real scenes always need it); classical CV.
**Risks / edge cases found:** The 16.45 GB fp32 checkpoint plus 9.32 GB base vision does not fit the ~9 GB Mac budget unquantized. The released `checkpoint.json` says step 100; predict.py uses 1024 image tokens vs 512 in training. Details: `docs/research/hierarchical-timing-and-mjev.md`.

## [2026-10-17] Literature positioning: dropped tools and prior art
**Decision:** Drop 3DCodeVerse and VideoJev. The plan's own memory + monitor loop handles visual history, using OneJev with multi-image keyframes. Position the work against EmbodiedJev.
**Why:** EmbodiedJev uses text-only Jev with no learned executor and no calibration or risk–coverage analysis; showing what it cannot do is the research value.
**Rejected alternatives:** VideoJev as the progress monitor (4B, overlaps the planned feedback loop).
**Risks / edge cases found:** New System One papers appear weekly; re-check novelty before writing up.
