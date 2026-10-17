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

## [2026-10-17] MJev-Spatial answers at every decision point, before OneJev
**Decision:** At each decision point (subgoal boundary, failure or retry, never per control step), MJev-Spatial-4B answers first; its spatial evidence goes into OneJev's state, then OneJev decides.
**Why:** OneJev's action decision must rest on scene evidence; we do no direct CV, so MJev supplies it (relations, distances, points).
**Rejected alternatives:** Calling MJev only on OneJev uncertainty; MJev after OneJev; classical CV.
**Risks / edge cases found:** The calls are sequential, so decision latency = MJev + OneJev. The 16.45 GB fp32 checkpoint plus 9.32 GB base vision does not fit the ~9 GB Mac budget unquantized. The released `checkpoint.json` says step 100; predict.py uses 1024 image tokens vs 512 in training. Details: `docs/paper/related-work.md (§4) + appendix-engineering.md (§C)`.

## [2026-10-17] Literature positioning: dropped tools and prior art
**Decision:** Drop 3DCodeVerse and VideoJev. The plan's own memory + monitor loop handles visual history, using OneJev with multi-image keyframes. Position the work against EmbodiedJev.
**Why:** EmbodiedJev uses text-only Jev with no learned executor and no calibration or risk–coverage analysis; showing what it cannot do is the research value.
**Rejected alternatives:** VideoJev as the progress monitor (4B, overlaps the planned feedback loop).
**Risks / edge cases found:** New System One papers appear weekly; re-check novelty before writing up.

## [2026-10-17] Executors are robot-agnostic plug-ins
**Decision:** `Executor` interface + per-robot `EmbodimentAdapter` (observation/action mapping) + policy backend (LeRobot by repo id, later OpenPi client). Franka/RoboCasa first.
**Why:** Any robot and any VLA must plug into the low level without touching the core.
**Rejected alternatives:** Executor code specialised to RoboCasa or SmolVLA.
**Risks / edge cases found:** Action heads are embodiment-specific; a new robot needs a policy trained for it.

## [2026-10-17] SmolVLA: load the final trained weights only
**Decision:** Load `smolvla_robocasa` with `load_vlm_weights=False`; only config/tokenizer come from the SmolVLM2 repo.
**Why:** The base VLM weights would be overwritten by the checkpoint anyway.
**Rejected alternatives:** Default load (downloads 2.03 GB of weights that are then discarded).
**Risks / edge cases found:** Must assert zero missing/unexpected keys at load.

## [2026-10-17] GPU work on Kaggle (2×T4)
**Decision:** Use Kaggle 2×T4 for the MJev fp32 reference (port validation) and π0.5 runs.
**Why:** No local GPU; the Mac's ~9 GB cannot hold either.
**Rejected alternatives:** Buying or renting a GPU (not possible now).
**Risks / edge cases found:** T4 has no bf16 (fp16/fp32 only); weekly GPU quota and 12 h sessions.

## [2026-10-17] Decision timing chosen by experiment, not upfront
**Decision:** Implement the timing approaches in `playground/timing/` behind one interface and choose on measured results.
**Why:** The user won't choose without results; no paper compares sync vs async subgoal replanning directly.
**Rejected alternatives:** Committing to sync or async now.
**Risks / edge cases found:** Needs a working executor (G0-A) first.

## [2026-10-17] Executor eval settings come from the checkpoint, never from docs
**Decision:** Camera-key mapping is read from each policy's `policy_preprocessor.json`; the scene split matches its training data (`target` for `smolvla_robocasa`).
**Why:** The LeRobot RoboCasa docs swap two cameras relative to how `smolvla_robocasa` was trained; `lerobot-eval` overrides the stored map.
**Rejected alternatives:** Hardcoded mappings copied from docs or CI.
**Risks / edge cases found:** Earlier G0-A/G0-C results are invalid; worth reporting upstream to LeRobot.

## [2026-10-17] Executor rollouts use our own loop, not lerobot-eval
**Decision:** `executors/robocasa/rollout.py` mirrors `lerobot_eval.rollout`, but the instruction is a parameter checked against the checkpoint's training vocabulary.
**Why:** lerobot-eval always sends RoboCasa's sentence; smolvla_robocasa was trained on task names (A/B: 1/5 vs 0/5, same seeds). The supervisor must also control instruction text.
**Rejected alternatives:** Patching lerobot-eval; monkeypatching the env's task_description.
**Risks / edge cases found:** Must be kept in sync with LeRobot's rollout when the pinned commit changes.

## [2026-10-17] Position against Steerable Policies + LeRobot's external-VLM planner
**Decision:** Treat arXiv 2602.13193 and LeRobot's `rollout/planner.py` as the closest prior/concurrent work. Our contribution = typed probabilistic supervision (OneJev) instead of generated JSON.
**Why:** Same architecture (external model steers a VLA's subtask); the decision interface is the difference.
**Rejected alternatives:** Ignoring it (it's the obvious reviewer comparison).
**Risks / edge cases found:** Their planner could become a baseline arm (generative VLM) or a host for OneJev; pi052 lives on an unreleased branch.

## [2026-10-17] Executor: π0.5² (`lerobot/pi052_robocasa`), OneJev supplies its subtask
**Decision:** Use pi052_robocasa (LeRobot branch `feat/smolvla-on-steerable` @9d00293, own executor env). Steer it by subclassing the policy so `_generate_low_level_subtask` returns OneJev's choice. Run on Kaggle T4 in fp16, validated against an fp32 CPU reference.
**Why:** It is trained on natural language + subtask annotations (32k episodes); SmolVLA was ~1/5 per task. fp16 is the only single-T4 fit.
**Rejected alternatives:** SmolVLA (too weak); pi05_robocasa (training data undocumented); Mac bf16 (over the ~9 GB budget).
**Risks / edge cases found:** Unreleased branch; Gemma fp16 overflow (hence validation; fallback = fp32 across 2×T4); PaliGemma tokenizer gated; 4.14B params (2.82B bf16 + 1.32B fp32).

## [2026-10-17] Code reuse: what we borrow
**Decision:** Per `docs/features/v2-engineering-notes.md (§1)`. Import π0.5² (actions, its own planner as Arm A, subtask vocabulary as OneJev's candidates); port LeRobot's external-VLM planner (Arm C, Qwen3.5-0.8B) and RoboCombine's switching/metrics; MemER and Goal2Skill as ideas only; SkillNet not used.
**Why:** Avoid rebuilding what exists, and keep every arm on identical candidates and identical low-level weights.
**Rejected alternatives:** Forking π0.5² to change its planner; copying MemER code (no licence).
**Risks / edge cases found:** The planner lives on another unreleased branch (port, pin commit); the subtask vocabulary must be extracted from the dataset's `language_persistent` column.

## [2026-10-17] Scope the claim to recombination + decision quality, not new skills
**Decision:** The research claims compositional recombination of trained subtasks plus better switch/retry/abort decisions. No new-skill or cross-domain claims. A language-following check on π0.5² comes before supervisor work.
**Why:** Deep research (`docs/paper/related-work.md (§1–2)`): no VLA shows zero-shot new skills or cross-domain transfer; hierarchy helps on held-out compositions, but the low level is the bottleneck.
**Rejected alternatives:** Claiming generalization to new task domains.
**Risks / edge cases found:** π0.5² may ignore subtask text (the vision shortcut seen in LIBERO-PRO/CRAFT), which would make the selector moot.

## [2026-10-17] No standalone language-following test for π0.5²
**Decision:** Drop the separate Kaggle diagnostic (blank/gibberish/swapped subtask); language following is measured inside the supervisor runs.
**Why:** User call: the actual workflow exercises the same question with the real subtask sequence.
**Rejected alternatives:** Standalone E10 before supervisor work.
**Risks / edge cases found:** If π0.5² ignores subtask text, we learn it later, after the supervisor is built; arm E (oracle subtasks) bounds it.

## [2026-10-17] Candidate filter: keep the simple E9 builder, park templates
**Decision:** Use the E9 exact-string builder; template slot-filling (E9b) stays in `playground/`, not in `rojev/`.
**Why:** E9b @10 .405 vs .398 (no real gain); filter is an add-on, the working pipeline comes first; OneJev handles wording.
**Rejected alternatives:** Growing hand-written word lists (synonyms, relations) to cover edge cases.
**Risks / edge cases found:** New-task recall stays ~40% @10; reachability is the limit (75% exact vs 89–97% with templates).

## [2026-10-17] π0.5² driver on LeRobot's language runtime; Kaggle kernel with embedded bundle
**Decision:** `drive.py` reuses `lerobot.runtime` (env, loading, PI052 adapter) + success recording; Kaggle kernel gets pyproject/uv.lock/drive.py embedded, vendor repos cloned at pinned SHAs, `uv sync --frozen`.
**Why:** Official pi052 path (no re-implementation); OneJev plugs into its subtask hook; the lockfile, not Kaggle's image, defines the env.
**Rejected alternatives:** Own rollout loop (as for SmolVLA); private Kaggle dataset for code (stale-version risk); pip installs on Kaggle's image.
**Risks / edge cases found:** Runtime skips actions when `ctrl_hz>0` (set 0); T4 lacks bf16 (probe, fp32 2-GPU fallback); ~10 GB assets downloaded per session; CLI-pushed runs can't read Kaggle secrets → gated tokenizer files ship as a private dataset, passed as a local dir via LeRobot's processor overrides + `config.tokenizer_name` (global HF-offline breaks the FAST tokenizer).

## [2026-10-17] Steer π0.5² through the runtime subtask slot, not a subclass
**Decision:** OneJev's subtask enters via LeRobot's runtime `set_context("subtask", …)` (direct-subtask mode); replaces the planned `_generate_low_level_subtask` subclass.
**Why:** The official runtime already exposes this hook; no override of model code.
**Rejected alternatives:** Subclassing PI052Policy.
**Risks / edge cases found:** Hook lives on an unreleased branch (pinned @9d00293).

## [2026-10-17] Curate the paper bibliography
**Decision:** `docs/paper/references.bib` keeps 29 entries: load-bearing research papers, SmolVLA/π0-family papers/blogs/checkpoints, OneJev and MJev-Spatial. Codebases, tool docs, one-off side references removed; build notes moved to `docs/features/v2-engineering-notes.md`.
**Why:** User call: the paper folder holds only what the paper cites.
**Rejected alternatives:** Citing every source touched (141 entries).
**Risks / edge cases found:** Removed sources (e.g. EmbodiedJev, Jev blog, Goal2Skill) are no longer cited; re-add deliberately if the paper text needs them.

