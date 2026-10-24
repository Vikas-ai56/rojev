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

## [2026-10-18] Sim on the Mac, π0.5² as a cloud service behind a tunnel
**Decision:** RoboCasa runs locally (live window); π0.5² runs on Kaggle (`serve.py`), reached via a Cloudflare quick tunnel + bearer token; lossless .npz requests, one per 50-step chunk.
**Why:** User call: a clean split, a visible sim; Kaggle accepts no inbound connections and CLI runs cannot read secrets (rules out ngrok/Tailscale tokens).
**Rejected alternatives:** Sim on Kaggle (no live view); JPEG images (changes policy input); ngrok / Tailscale.
**Risks / edge cases found:** Quick tunnels have no uptime guarantee; ~0.3 s extra round trip; service idles out after 30 min.

## [2026-10-18] Arms B and E, and the first task set
**Decision:** B = authored sequence, fixed budget per subtask (p75 demo duration); E = same sequence, switch on ground-truth checks; tasks MicrowaveThawingFridge, MoveFridgeToFreezer, LoadCondimentsInFridge.
**Why:** User choices; ground truth logged for every arm attributes failures to arm vs executor.
**Rejected alternatives:** Human key-press oracle (not scalable); OneJev-only switching for B (kept as a possible ablation).
**Risks / edge cases found:** Noisy π0.5² training labels; "place … on the middle shelf" not in its vocabulary.

## [2026-10-18] Drop tianshou from the robocasa dependency override
**Decision:** Remove `tianshou==0.4.10` from the executor env's robocasa metadata override.
**Why:** Only `robocasa/scripts/bench_speed.py` imports it; via tensorboard it pins protobuf<5, blocking rojev's tracing (protobuf≥5).
**Rejected alternatives:** Separate processes for sim and supervisor; dropping tracing.
**Risks / edge cases found:** RoboCasa's speed benchmark script no longer runs in this env.

## [2026-10-18] Oracle (arm E) reads current state and may go back
**Decision:** E's next subtask = the one after the latest subtask whose ground-truth check holds now; "pick up" = grasped AND off its source fixture/container.
**Why:** E10: grasp contact fired with the steak still on its plate; forward-only E kept saying "place" after a drop.
**Rejected alternatives:** Grasp-only check; forward-only sequence; distance-lift thresholds.
**Risks / edge cases found:** Relies on checks being states (held/inside/closed/on); a non-persistent check would need its own rule.

## [2026-10-18] Ground-truth checks are evaluation-only; baselines leave the supervisor package
**Decision:** Arms B/E move to `rojev/eval/baselines.py`; `rojev/supervisor` keeps only the arm contract and the real supervisor. Sim checks (`tasks.py`) are used for scoring, the arm-E ceiling and as training labels, never as supervisor input.
**Why:** User: per-subtask hand-coded checks do not generalise (unknown number of subtasks, continuous actions outside the kitchen); the model must judge from images and be trained/given better input instead.
**Rejected alternatives:** Validating previous subtasks with per-subtask checks inside the supervisor.
**Risks / edge cases found:** Arm D's recovery depends on OneJev re-reading the image and on earlier subtasks being in the shortlist; to be measured.

## [2026-10-18] Arm D on two GPUs: MJev service (own env, bf16 trunk) + speculative π0.5²
**Decision:** New notebook rojev-armd: MJev-Spatial on GPU 1 (separate env: torch 2.9, transformers 5.17.0), π0.5² on GPU 0; OneJev local. MJev trunk stored bf16, readout fp32; base model pinned to its `base_revision`; arm D overlaps the "continue" chunk with the decision.
**Why:** MJev's fp32 language weights (16.4 GB) exceed a T4; its pins conflict with π0.5²'s env; most chunks are "continue", so overlap halves chunk latency.
**Rejected alternatives:** One shared env; fp32 MJev split across both GPUs (contends with π0.5²); MJev on the Mac (MLX port, weeks).
**Risks / edge cases found:** Upstream pins inconsistent (transformers 5.17.0 needs huggingface-hub≥1.5, MJev pins <1) → relaxed via metadata override; upstream inference runs fp32 weights, so bf16 is validated vs fp32 on CPU (`check_precision.py`); ~37 GB of weights on a 57 GB disk.

## [2026-10-18] Arm D evidence = MJev judges candidate subtasks with one template
**Decision:** MJev answers "Robot step: '<s>'. Which is true right now?" (done/possible/blocked) for current + recent + top-M candidates; one OneJev choice decides (continue / switch / redo / complete).
**Why:** User: evidence must be general, not hand-picked facts (gripper only). Subtask-level judgements cover presence, completion and state for any language subtask.
**Rejected alternatives:** Fixed fact questions (gripper, door); object parsing + per-object questions; separate Q1 done-check.
**Risks / edge cases found:** MJev not trained on this template → accuracy measured vs sim ground truth (`evidence_vs_truth`); M bounds what is judged.

## [2026-10-18] Evaluation not tied to authored tasks
**Decision:** Arm D runs on any of the 300 RoboCasa pretrain tasks, scored by RoboCasa's own task success; `tasks.py` stays for 3 diagnosis tasks (per-subtask truth, arms B/E). MJev precision inputs are seeded random scenes with filter-chosen subtasks (`make_testdata.py`).
**Why:** User: avoid overfitting to hand-picked tasks/scenes.
**Rejected alternatives:** Authoring sequences and checks for many tasks; hand-picked test scenes.
**Risks / edge cases found:** Tasks without `tasks.py` have no per-subtask attribution, only task success and MJev evidence logs.

## [2026-10-18] Add flash-linear-attention to the MJev env
**Decision:** `flash-linear-attention` (Linux) in `services/mjev`; keep bf16, 3 cameras.
**Why:** E12: precision and image-token levers barely change MJev latency (≈ 8 s/question); transformers falls back to PyTorch Gated-DeltaNet without FLA.
**Rejected alternatives:** fp16 (≈ 10% faster only); 2 cameras (answers change, Δp 0.22); fewer image tokens (no effect at 256²).
**Risks / edge cases found:** FLA's Triton kernels on T4 (sm_75) untested; `causal_conv1d` stays on its fallback (needs a CUDA build).

## [2026-10-18] FLA did not fix MJev latency; profile before the next change
**Decision:** Keep FLA in the MJev env (harmless, used), but make no further speed change until a stage profile (`profile_mjev.py`) shows where the ≈ 24 s per request goes.
**Why:** E12b: with FLA, bf16 24.4 s and fp16 21.8 s per request (unchanged); two guessed causes (precision, FLA) were wrong.
**Rejected alternatives:** More guessed fixes (causal_conv1d build, fewer cameras) without measurement.
**Risks / edge cases found:** Remaining candidates: `causal_conv1d` fallback, CPU preprocessing, vision encoder, MJev's Python loops.

## [2026-10-18] OneJev decision quality measured on demo frames; next lever is training
**Decision:** Use the E13 demo-frame benchmark as OneJev's yardstick; adopt ≤ 8 distinct options; plan OneJev training on image-dependent next-subtask labels; evaluate 4B as the decision model.
**Why:** E13: blank image ≈ real image accuracy (0.8B and 4B) → OneJev ignores the cameras for this decision; 4B 40% vs 0.8B 22.5%; distinct options 30% vs 22.5% (0.8B).
**Rejected alternatives:** More prompt wording changes (do not make it use the image); separate camera images (no gain).
**Risks / edge cases found:** n = 40; mradermacher 4B GGUF fails in llama.cpp 0.6.0 (missing blk.32 tensor) → prithivMLmods @eb1a6d6 used; 4B on the Mac uses ~5.2 GB.

## [2026-10-18] MJev patch embedding as its exact matrix-multiply form
**Decision:** At load, replace the vision tower's patch-embedding Conv3d (kernel = stride) with `PatchLinear` (same weights, F.linear over flattened patches); startup check vs the conv refuses to serve if they differ (> 0.05).
**Why:** E12d/E12e: that Conv3d took 20.6 of 23.8 s per request on T4 (bf16); the matrix form takes 4.8 ms with bf16-rounding-level differences.
**Rejected alternatives:** fp32 Conv3d only (15.9 ms but changes the dtype flow into bf16 blocks); fewer cameras (changes answers); waiting for an upstream fix.
**Risks / edge cases found:** Library-module swap at load time (documented, checked); upstream cause (slow bf16 Conv3d on sm_75) unreported.

## [2026-10-18] Serve MJev in fp16
**Decision:** `serve_mjev.py` default `--language-dtype fp16` (trunk + vision fp16, readout fp32), with `PatchLinear`.
**Why:** E12f: fp16 1.3 s vs bf16 3.5 s per 3-question request on T4; both 12/12 top-1 vs fp32, no NaN.
**Rejected alternatives:** bf16 (2.7× slower, T4 has no fast bf16); fp32 (does not fit with the trunk).
**Risks / edge cases found:** fp16 overflow risk on unseen inputs; validated on 4 scenes only → keep `check_precision.py` as a gate.

## [2026-10-19] Research-led plan for selector accuracy (Codex peer report)
**Decision:** Order: fix E13 (fixed option order, matched counterfactuals, destination-aware metric) → OneJev 1.6 0.8B/4B + base-Qwen answer-logit baseline → π0.5² language-head candidate scoring (+ fusion) → separate fact/switching questions → 0.8B LoRA pilot. Details: `docs/features/v2-selector-approaches.md` (removed 2026-10-22; in git history).
**Why:** Codex research + verification: E13's image ablation was confounded; OneJev 1.6 is released; hierarchical systems that work (Hi Robot, π0.5, RT-H) train the high-level interface; π0.5²'s head already maps images to this vocabulary.
**Rejected alternatives:** Replacing OneJev with a generative VLM at runtime; more prompt wording; shrinking options without recall measurement.
**Risks / edge cases found:** π0.5²-head scoring latency on T4 unmeasured; gains from fusing the executor's own prior change the paper's framing; trainer needs fp16 adaptation for T4.

## [2026-10-19] Decision model: OneJev 1.6 4B; drop 0.8B
**Decision:** Use OneJev 1.6 4B for arm D and all further selector tests; no more 0.8B runs.
**Why:** E13: 4B 40% vs 0.8B 22.5% on the same prompt; 1.6 released 2026-10-16; 4B fits the Mac (~5 GB Q8).
**Rejected alternatives:** 9B (does not fit the Mac budget or a T4 beside MJev); continuing 0.8B ablations.
**Risks / edge cases found:** 1.6 4B GGUF compatibility with local llama.cpp unverified (1.5 4B mradermacher build failed to load).


## [2026-10-20] Jev × Helix plan: test with frozen models before any training
**Decision:** No S1/S2 training for now; first test the plan's runtime, failure handling and Jev's value with frozen models (π0.5² in RoboCasa; scripted skills on SO-101 MuJoCo; OneJev 4B, MJev as is).
**Why:** User call; review found the protocol/failure gaps and Jev's usefulness are untested, and training is costly to undo.
**Rejected alternatives:** Starting S1/S2 training in parallel with the runtime (plan WP2–WP4 order).
**Risks / edge cases found:** π0.5² is Panda/RoboCasa, not SO-101 → model-behaviour results may not transfer; scripted SO-101 skills are a perfect executor, so they test the runtime, not S1.

## [2026-10-20] Adopt the Jev × Helix plan (amended) as v3; one roadmap file
**Decision:** v3 follows `docs/plan/jev_helix_implementation_plan.pdf` with the review's amendments; all phases/tests/gates tracked in `docs/plan/v3-roadmap.md`.
**Why:** Review (own audit + Codex + 3 research streams) found the responsibility split sound but the protocol, hold/transition controllers and completion ownership missing.
**Rejected alternatives:** BM25/transition candidate filter (replaced by skill registry × tracked objects); adaptive depth and merged OneJev+MJev trunk now (deferred until measured need).
**Risks / edge cases found:** Revoke/epoch/ACK gaps were blockers in the source plan; identity tracking method still to choose (MJev seeds a tracker).

## [2026-10-20] Simulator time modes; realtime physics thread replaces SimClock.paused
**Decision:** `MujocoRobot(time_mode=lockstep|paced|realtime)`; realtime = `RealtimeSimClock` stepping physics on its own thread at wall-clock rate; `SimClock.paused()` removed; `Rate` drops missed ticks.
**Why:** v3 T1.1: timing tests are meaningless if physics stops during inference; hardware never pauses.
**Rejected alternatives:** Keeping `paused()` (hides delay); stepping physics inside `read()` (couples physics to the caller); camera rendering on the physics thread (GL context on macOS, and it would stall physics).
**Risks / edge cases found:** Rendering is on the caller thread, so a slow render delays that caller (not physics); stepping slower than wall clock is reported via `lag_s`, not hidden; lockstep remains the default for deterministic tests.

## [2026-10-20] v1 spec (no training): one runtime, two tracks
**Decision:** `docs/plan/v3-v1-spec.md`: shared `rojev/runtime` (messages, owner, execution runtime, guard, S1 runner) used by an SO-101 realtime track (scripted mink skills, oracle target poses) and a RoboCasa lockstep track (frozen π0.5²); identity by MJev point + depth association; Jev v1 event-triggered with margin/dwell/budgets.
**Why:** Tests runtime/failure handling and Jev's value without training; scripted S1 isolates the runtime from S1 skill.
**Rejected alternatives:** Frame-rate visual tracker in v1 (new model); perception-driven skill targets in v1; WebSocket transport in v1.
**Risks / edge cases found:** Oracle poses make SO-101 S1 perfect (runtime results only); RoboCasa stays lockstep, so latency is measured on SO-101 only.

## [2026-10-20] Correction: v1 = full S1/S2 loop from open-source frozen components (RoboCasa)
**Decision:** v1 = plan WP0–WP2, route A, frozen: π0.5² (S1) + OneJev 1.6 4B + MJev + SAM 2 tracker + our runtime, real-time RoboCasa loop; replaces the earlier SO-101 scripted-skill/oracle-pose track.
**Why:** User definition of v1 + plan p.2 ("RoboCasa checkpoint today"), p.5 (route A = existing VLA), p.24 (WP1 S1 baseline, WP2 direct visual Jev); scripted skills with oracle poses are not an S1.
**Rejected alternatives:** SO-101 scripted mink skills as S1 (not open-source policy, oracle inputs); SmolVLA/GR00T zero-shot on our SO-101 scene (no checkpoint for it without training).
**Risks / edge cases found:** π0.5² overrides the denoise path without LeRobot's RTC hook → port needed; LeRobot async_inference supports pi05 not pi052 and SO robots only; SAM 2 fit on RoboCasa frames unverified.

## [2026-10-20] Outatime-style speculation: computation only (roadmap T10)
**Decision:** Adopt bounded speculation of decisions/chunks after T8; never execute speculative branches; no physical backtracking.
**Why:** Outatime (Lee et al., MSR TR 2014) masks ≤250 ms with ≤4 parallel branches + checkpoint/rollback, which needs a deterministic virtual world; a robot cannot undo actions.
**Rejected alternatives:** Executing predicted subtasks and backtracking on misprediction; speculating full responses to user changes (rare, open-ended).
**Risks / edge cases found:** Prepared chunks go stale (non-deterministic world); compute contention with the live S1 chunk; most switch delay may be Jev, not S1 (measure in T8).

## [2026-10-20] v1 step 1: robot-agnostic execution runtime + guard
**Decision:** `rojev/runtime` with `ActionSpec` as the only robot-specific input; policies keep `ActionChunk`, the runtime wraps it in `TimedChunk`; transitions `pause_hold` / `immediate` only.
**Why:** Any robot plugs in by describing its action vector (bounds, hold, switches, irreversible dims); no policy/wire change needed for command ids.
**Rejected alternatives:** Adding ids to `ActionChunk` (leaks runtime into policies); `finish_safe` transition (needs per-skill phase knowledge → heuristic); guessing hold values without a seed.
**Risks / edge cases found:** Blending acts on action values (deltas for PandaOmron, targets for joint robots) — check smoothness live; irreversible gating needs each robot spec to mark its switches correctly.

## [2026-10-20] v1 step 2: control loop + S1 runner; chunks placed at observation time
**Decision:** `ControlLoop` steps any backend on its own thread (robot clock t = k·dt, wall-paced, lag reported); `S1Runner` requests a chunk when the queue is under `refill_s` and places it at the observation's capture time; the π0.5² service stays unchanged.
**Why:** The robot never waits for a model; inference latency costs only the skipped prefix, never stale motion. Live RoboCasa check: real-time factor 1.00, max lag 3.5 ms, 95% of ticks from chunks.
**Rejected alternatives:** Placing chunks at arrival time (replays motion computed for an older state); adding command ids to the service (runner wraps instead).
**Risks / edge cases found:** RoboCasa step ≈ 40 ms of the 50 ms tick on the Mac; a policy slower than its whole chunk never moves the robot (correct, but needs a faster S1 or RTC).

## [2026-10-20] Review fixes: any robot plugs in; GL thread; robot I/O serialized
**Decision:** `RobotBackend` + `joint_target_spec` make any `rojev.robot.Robot` a runtime backend (robot clock paces the loop); `ControlLoop` uses a backend's own clock when it has one; MuJoCo cameras render on one GL thread; `Robot.read/write` serialized; leases end commands in any mode; chunk `dt` must equal the control period; old-command chunks cannot delay a takeover; `AcceptedGoal.instruction` conditions S1.
**Why:** Code review before the briefing: the SO-101 could not plug in; a control loop rendering from its own thread deadlocked (macOS CGL); a starved command outlived its lease; S1 would have seen the stale task text after a change of mind.
**Rejected alternatives:** Rendering only on the main thread (forces the loop onto it); resampling mismatched chunks inside the runtime (policy-specific); per-thread GL contexts (cannot be closed safely from another thread).
**Risks / edge cases found:** Threshold dimensions are never blended, so a continuous gripper modelled as a switch can jump (the robot's own velocity limit bounds it); one GL thread serializes all rendering of a robot.

## [2026-10-20] v1 step 3: state owner + S2 worker; baselines on the runtime
**Decision:** `StateOwner` (one thread, event queue) is the only writer of instruction revision, desired command and active record; `DecisionWorker` runs any decider (newest request wins) and stamps proposals; baselines `scripted` / `oracle` / `native` as deciders in `rojev/eval/deciders.py`; π0.5² service gains a planner mode (empty subtask) for `native`; `ExecutionAck` splits `rejected` (command) from `chunk_rejected` (one chunk).
**Why:** Plan p.10–12: proposal ≠ command ≠ active; one owner; stale results refused by metadata. Live RoboCasa with a change of mind: all three arms, 0 errors, real-time factor 1.00.
**Rejected alternatives:** Deciders writing commands directly (two writers); owner waiting on the model; one ACK kind for chunks and commands (a late chunk would have dropped a valid command); accepting proposals with preconditions unchecked.
**Risks / edge cases found:** Preconditions are refused until step 5 adds evidence; scripted ignores instruction changes by design; native planner path untested until π0.5² runs on a GPU.

## [2026-10-20] Package restructure: deep modules split by S1 / S2 / shared, layering enforced
**Decision:** `rojev/` = `protocol` (observation, commands, decisions) → `robot` (+ `action_spec`, `sim/` MuJoCo backend with `world`, `gl`) → `control` (execution, guard, loop, robot_backend) → `system1` (policy, wire, remote, runner) / `system2` (decider, worker, models/) → `owner` → `eval`; `observability`; v2 code quarantined in `legacy_v2`. Renames: `Executor`→`Policy`, `RemoteExecutor`→`RemotePolicy`, robot `Observation`→`RobotObservation`. `tests/test_architecture.py` enforces the import direction.
**Why:** User hard rule: no spaghetti; the `runtime/` catch-all and name clashes (`executor` vs `ExecutionRuntime`, two `Observation`s) would confuse the team as Jev, scenarios and the tracker land.
**Rejected alternatives:** Keeping `runtime/` and adding sub-folders later (cost grows with every phase); deleting v2 now (arm D is still the T8 comparison).
**Risks / edge cases found:** External notebooks/scripts importing old paths break (all in-repo consumers updated and run); review fix in the same change: the owner now falls back to the still-running command when a pending one is rejected or expires.

## [2026-10-20] Borrow execution-side mechanisms (CheckVLA, CA³C, PACE); review mode first
**Decision:** Build, in order: (1) review mode on instruction change; (2) commitment policy in `system1` (fixed → PACE phase boundaries → K-sample action consensus, CA³C-lite); (3) execution monitor (kinematic expected-vs-observed + conformal threshold → `Deviation` event to the owner, refill-now repair with the existing hard prefix); (4) progress keyframes in the owner. World-model versions (CheckVLA, CA³C) after v1.
**Why:** User goal: accuracy and naturalness. Sources verified (docs/research/2026-10-20-jev-role-and-novelty.md §10–11); all four are training-free in their v1 form.
**Rejected alternatives:** Training a VLA-Corrector-style critic now (needs our rollouts); CA³C with a world model now (latency unreported, small gain over a tuned fixed horizon).
**Risks / edge cases found:** Action-space disagreement is a proxy for consequence disagreement; kinematic monitor misses visual-only failures (slip without arm deviation) until the visual version exists.

## [2026-10-20] Spec revision: step 4 = review + commitment + execution monitor + keyframes; Jev v1 = interaction acts
**Decision:** `docs/plan/v3-v1-spec.md` rewritten: new step 4 (review mode, decelerating hold, commitment policy, kinematic execution monitor + `Deviation`, owner two lanes, progress keyframes); steps 5–8 build on it (faults → perception + Jev v1 → π0.5² service RTC/FP8/K samples → benchmark runs). Jev v1 = interaction acts {continue, revise, guarded-wait, ask, abort} + verifier checks; π0.5²'s own planner proposes subtask text; T8 = revision benchmark. Roadmap updated (T1.11–T1.14, T8).
**Why:** User: keep everything sequential so later steps build on one design; user's idea that Jev chooses the response to a changed instruction; research brief §5, §10–11.
**Rejected alternatives:** Appending the new work at the end (later steps would contradict it); keeping Jev as next-subtask selector (40% E13, open-ended set).
**Risks / edge cases found:** The paper framing changes from "Jev selects subtasks" to "Jev decides how to respond to change"; π0.5² planner mode is untested until GPU access.

## [2026-10-20] Step-4 features are plug-and-play; speculative prediction opt-in (`speculative_predict=False`)
**Decision:** Every step-4 feature sits behind an interface and one config field (spec §14): `speculative_predict` (default off), `commit_policy`, `execution_monitor` (off), `on_instruction_change` (`review` | `revoke`), `stop_s`, `keyframes`.
**Why:** User: flexibility, plug and play; off = documented default, so each feature is also a benchmark ablation (e.g. `revoke` = stop-and-go baseline).
**Rejected alternatives:** Hard-wiring the features; boolean checks scattered through modules instead of selected implementations.
**Risks / edge cases found:** Feature combinations multiply test cases → fault suite (step 5) runs the defaults plus each feature on alone.

## [2026-10-20] Step 4.1 built: review mode + decelerating hold
**Decision:** `ExecutionRuntime.review` (keep motion, freeze irreversible switches, cap lease) + `stop_s` ramp; `StateOwner(on_instruction_change="review"|"revoke", review_s)`; CONTINUE under a newer instruction = reaffirm (new command, same operation, new instruction, `immediate`); operation start time kept across reaffirmations.
**Why:** Naturalness without giving up safety; live: no pause after a change of mind (review) vs a 0.3 s stop (revoke).
**Rejected alternatives:** Renewing the old command on CONTINUE (S1 would keep the stale instruction); ramping KEEP dimensions (position robots already hold their last target).
**Risks / edge cases found:** Up to `review_s` of motion toward the old goal (reversible only); one test tolerance loosened (SO-101 real-time lag bound 50 → 100 ms under full-suite load).

## [2026-10-20] Step 4.2 built: commitment policy (fixed default; phase; consensus opt-in)
**Decision:** `system1/commit.py` + `S1Runner(commit=...)`; consensus = majority agreement with the committed sample, majority vote on irreversible switches, never-empty commitments; floor rule `min_commit · dt ≥ 2 × latency + dt`.
**Why:** The first design (truncate at any switch disagreement, even to zero; mean pairwise agreement) livelocked and starved in the closed-loop sim: sub-latency commitments are skipped by the runtime, and one outlier sample shrank every commitment.
**Rejected alternatives:** Deferring on empty commitments (holds on an unchanged observation → livelock); latency-adaptive floor (a heuristic; the rule is documented and tested instead); K-even tie rules.
**Risks / edge cases found:** Validated only with stand-ins and a simulated grasp; π0.5²'s real sample diversity unknown → T8a ablation before use in benchmarks; K samples cost K × inference until batched (step 7).

## [2026-10-21] Step 4.3 built: kinematic execution monitor with split-conformal threshold
**Decision:** `rojev/robot/effect.py` (`EffectModel` plug-in) + `rojev/system1/monitor.py` (window residual vs a linear map fitted on nominal runs; split conformal over time segments; calibration envelope); `Deviation` → owner decision + `S1Runner.refill_now`; advisory, never a stop; opt-in.
**Why:** CheckVLA-style execution check without training; the fitted map absorbs controller gains and lag, so plug-ins only name corresponding numbers.
**Rejected alternatives:** Hand-set controller gains (robot-specific, brittle); per-tick conformal (overlapping windows are not exchangeable); `eval/calibration.py` (eval may not import system1; the fit belongs to the monitor); stopping on a deviation (false alarms would stop the robot).
**Risks / edge cases found:** α is per segment (1 s at 0.05 ≈ 3 false alarms/min) → reported as alarms/min; stand-in calibration excites one axis → envelope rule; contacts in "nominal" runs inflate the threshold; spurious macOS Accelerate matmul RuntimeWarning during fit (results equal einsum).

## [2026-10-21] Commitment floor and request lead time derived from measured latency
**Decision:** `CommitPolicy.plan(candidates, spec, floor)`; `S1Runner` measures latency L (largest of the last 20 requests, observation age at submit) and passes floor ⌈2L/dt⌉ + 1, requests when < max(`refill_s`, L + dt) is queued; `min_commit` removed; runtime counts `starved_ticks`.
**Why:** The floor depends on latency (incl. K samples, network), which only the runner sees; a step constant was unsafe on RoboCasa (needed 13, default 10). Derivation: gap-free iff lead ≥ L and n·dt ≥ 2L.
**Rejected alternatives:** A configured `min_commit` (patch: wrong whenever latency changes); test-side starvation inference (the runtime owns what executes).
**Risks / edge cases found:** First request has no measurement (whole chunk, configured lead); a request slower than any recent one can still starve (counted); a chunk shorter than the floor starves unavoidably (unbatched K samples, 25-step chunk: measured).

## [2026-10-21] Monitor fixes found by stronger tests: split by run, lag-aware map, exercised directions, resolution
**Decision:** Calibrate on other runs than the fit (even runs fit, odd calibrate); regress on this and the previous window's commands; score only command directions exercised in calibration (no magnitude box); residual scale ≥ `EffectModel.resolution`; `TickRecord.before/after` so no tick is lost; recordings carry run conditions, calibration refuses runs below real-time factor 0.98 or with mixed policies.
**Why:** Held-out tests showed 0.17 false alarms per segment for α = 0.05 (segment split shares each run's motion), small pushes hidden by lag misfit, peak-speed blocks unscored by the min/max box, a dropped first tick, and calibration runs recorded under CPU contention.
**Rejected alternatives:** Loosening test bounds (done before, reverted: bounds now come from the conformal rank); `+1e-9` scale guards; a widened box.
**Risks / edge cases found:** Segments of one run are correlated → guarantee exact per run, approximate per segment (tested on held-out runs); saturation beyond the calibrated magnitude shows as alarms.

## [2026-10-21] Deterministic logic tests; wall-clock budgets as a separate `timing` suite
**Decision:** Runner and SO-101 tests run on simulated or lockstep time in one thread; pacing and real-time budgets (one control period) are `@pytest.mark.timing`, excluded by default, run alone with `pytest -m timing`.
**Why:** The SO-101 lag bound had been loosened 50 → 100 ms to pass under suite load (a patch); logic and timing are different claims.
**Rejected alternatives:** Keeping a loosened bound; skipping the timing tests.
**Risks / edge cases found:** Owner live tests stay multi-threaded (they test the threads; ordering assertions with second-scale margins).

## [2026-10-21] numpy matmul RuntimeWarning in the RoboCasa env: cause identified, not suppressed
**Decision:** Keep the warning visible; no `-W ignore`.
**Why:** RoboCasa pins `numpy==2.2.5` with `numba==0.61.2` (executor `pyproject.toml:30`); that build (Accelerate BLAS) raises spurious FP-exception warnings in matmul; the root env (numpy 2.5.4) does not; results equal `einsum` exactly.
**Rejected alternatives:** Silencing warnings; overriding RoboCasa's numpy pin (numba compatibility).
**Risks / edge cases found:** Real FP errors in that env would look the same; correctness of the fit is pinned by tests in the root env.

## [2026-10-21] Step 4.4 built: two decision lanes with priority; progress keyframes replace history
**Decision:** `StateOwner(planner, interaction=None)`: instruction / deviation triggers → interaction worker, the rest → planner; while an interaction request pends, planning requests are deferred and planning proposals refused; release by proposal, `DecisionFailure` (new; the worker posts it on decider errors) or `interaction_timeout_s`; on failure, timeout or refusal the trigger escalates to the planning lane. Proposals must answer a request the owner made. `Keyframe` (cmd, subtask, revision, decision's observation, evidence) recorded when a running step is moved past; `snapshot.keyframes` replaces `history`; always on.
**Why:** A routine next-step decision must never override the response to what the user just said; a silent decider must not block planning forever; progress must survive repairs and revisions with the observation it was judged on.
**Rejected alternatives:** Per-lane applied counters (priority gating keeps one global order valid); timeouts only (a failing decider is known at once); keyframes as a flag (they are the progress record); history kept beside keyframes (two records of one fact); keyframes on ABORT (the old history recorded aborted steps as moved past: fixed).
**Risks / edge cases found:** Keyframe evidence is the decider's judgement until postconditions (step 6); a reaffirmed step's keyframe carries the reaffirmed command id; deferred planning triggers are dropped (heartbeat resumes after release).

## [2026-10-21] Planning lane becomes event-driven in step 6; the 1 s heartbeat is a stand-in
**Decision:** Step-completion events (postcondition watchers, π0.5²'s planner switching subtask, phase boundaries) trigger planning decisions; the heartbeat remains only for liveness and lease renewal, slowed to just under the lease.
**Why:** Polling every second wastes S2 calls (most answer "continue"), notices a finished step up to 1 s late, and is costly with MJev (~1.3 s per check); no step-done signal exists until postconditions (step 6).
**Rejected alternatives:** Faster polling (more waste); removing the heartbeat (the lease is the dead-man's switch on S2).
**Risks / edge cases found:** A missed event falls back to the slow heartbeat; event sources need their own false-positive handling (postcondition flicker).

## [2026-10-21] Sim scaffolding lives in the executor; the core never bends to the simulator
**Decision:** Scripted S1, keyword interaction acts, revisable planner, fault injection, recording wrappers, mirror viewer, video and the scenario matrix live in `executors/robocasa-pi052/harness/` and `scenarios.py`, using only the core's interfaces; every stand-in is declared as a spoof.
**Why:** User rule; the core must behave the same on the real robot.
**Rejected alternatives:** Hooks in the core for recording (wrappers over Policy / CommitPolicy / Decider instead); a scripted S1 with privileged object attach (physics stays real).
**Risks / edge cases found:** Scripted grasps of a 15 cm flat steak fail physically (reach and contact are real); unreachable targets without base driving stay unfinished.

## [2026-10-21] Runtime projects its own blends onto the bounds; violations are upstream only
**Decision:** `ExecutionRuntime.tick` checks and clamps the incoming action (reported as `violations`), then blends / decelerates, then projects onto the bounds without reporting.
**Why:** The inertialized offset rides on later actions of a segment and pushed one to −1.006; the runtime reported its own transform as an upstream fault (10 of 13 live cases).
**Rejected alternatives:** Ignoring small clamps; tolerances in the invariant.
**Risks / edge cases found:** A projected blend is no longer a pure offset decay at the bound (correct: the robot cannot exceed it).

## [2026-10-21] Live viewer as a mirror process; run video encoded after the run
**Decision:** `harness/viewer.py` runs the MuJoCo passive viewer in its own `mjpython` process fed by shared memory (`.mjb` model + qpos / mocap); `harness/video.py` keeps raw frames during the run and encodes at the end.
**Why:** robosuite renders cameras through a GLFW window that needs the main thread, which `mjpython` takes; `mjpython` replaces dyld's default fallback path (system zlib for numba); encoding during the run stalled the loop 236 ms (GIL) and corrupted the S1 latency estimate.
**Rejected alternatives:** In-process viewer; OpenCV window; a video thread or queue.
**Risks / edge cases found:** Mirror shows physics state, not robosuite-side visual changes; video memory ~6 MB per second of run.

## [2026-10-21] Remote model transport: persistent connections; HTTP/3 (QUIC) as the primary path
**Decision:** `rojev/transport/`: `H3Transport` / `H3Server` (aioquic, CUBIC, self-signed certificate pinned as the only trust anchor, bearer token, reconnect once on `ConnectionLost`, never on `ServiceError`) and `HttpTransport` (HTTP/1.1 keep-alive, SSH-tunnel fallback); `RemotePolicy` picks by URL scheme; services answer through one handler on both (HTTP/1.1 threaded servers).
**Why:** Measured: a new connection per request cost 680 ms of network against a 216 ms ping; keep-alive 255 ms; QUIC adds no head-of-line blocking, cheaper resumption, address migration (robot Wi-Fi), and carries the next step (streamed frames, pushed chunks). The robot will reach a cloud GPU, so this is not sim-only.
**Rejected alternatives:** WebSocket over TCP (TCP head-of-line blocking; user asked for HTTP/3); 0-RTT early data (replayable /act); a public CA (pinning is stronger for one known server).
**Risks / edge cases found:** Needs UDP 4433 open (cloud firewall); UDP blocked on some robot networks → HTTP/1.1 fallback; aioquic is pure-Python (fine at ~10 req/s, ~150 KB); no protocol beats the ping: one request ≥ one RTT.

## [2026-10-21] GPU setup brings up S1 and S2 as components with one hook interface
**Decision:** `scripts/gpu_setup.sh HOST --executor E --services onejev,mjev` treats the executor and each S2 service as a component (`<kind>/<name>/gpu/hook.sh`); each runs as systemd unit `rojev-<name>` behind its own tunnel; one service JSON per box; weights pinned in `config/*_weights.txt` (`scripts/hf_pins.py`); NVIDIA utils installed by the built kernel module's version.
**Why:** One idempotent command with a read-only `--dry-run`; S2 models serve every executor; `nvidia-driver-570-server` is transitional to 580.
**Rejected alternatives:** nohup + pkill (matched its own ssh shell); git push to the box; copying vendor/ (11 GB); sending the HF token to the box; S2 setup inside the executor hook.
**Risks / edge cases found:** Fresh-box install, unit lifecycle, memory refusal and tunnels not yet exercised; the π0.5² hook predates HTTP/3 flags; `services/` is git-ignored locally (hooks and envs untracked); macOS openrsync quirks handled.

## [2026-10-21] Kinematic monitor cannot separate expected contact from a fault
**Decision:** Keep the single-threshold monitor; record contact-aware calibration (free-motion vs contact phases from a real signal: perception, force, or a world-model prediction) as an open step-6 item; no threshold tuning.
**Why:** Scripted S1 calibration runs press into contact while grasping (residual scale 2.9 cm), so a 2 s block during a grasp phase stays below threshold (max score 2.15 vs 2.54).
**Rejected alternatives:** Lowering the threshold or excluding contact ticks by hand (patches).
**Risks / edge cases found:** π0.5² makes intended contact too; until fixed, deviations are reliable only for faults far above contact-level residuals.

## [2026-10-21] π0.5² adapter maps switch outputs into the action space
**Decision:** `serve.py` `to_action_space`: PandaOmron switch dimensions (gripper, base mode) → low / high by their threshold, as RoboCasa's gym wrapper does; continuous dimensions untouched.
**Why:** First live π0.5² run: 486 of 600 ticks reported violations, all on the two switches (≤ 1.018); the runtime reported correctly, the adapter was not producing the robot's action space.
**Rejected alternatives:** Snapping switches in the core runtime (wrong for joint-target robots, whose gripper switch is a position); widening the bounds; ignoring the reports.
**Risks / edge cases found:** Continuous overshoots (e.g. 1.003) are still reported as upstream violations, as they should be.

## [2026-10-21] v1 = plan Route A with Jev limited to changes; Helix-type route B = v2
**Decision:** v1: π0.5² picks the next step and moves; Jev answers only changes (continue / revise / wait / ask / abort) and ambiguous checks; MJev checks end states and grounds objects. v2: Helix-type small fast S1 conditioned on a learned Jev latent, trained, promoted only if it beats v1.
**Why:** Jev as next-step selector measured 40% (E13) and added unwanted exchanges; the plan itself keeps Route A as the control and needs training for Route B (p. 5, 9); v1 is no-training.
**Rejected alternatives:** Jev selecting every next step (plan Figure 1 wording; weak in E13); Jev approving π0.5²'s proposals each step (same unwanted exchanges); Route B in v1 (needs training).
**Risks / edge cases found:** Deviates from Figure 1 ("Jev selects intent") for routine steps; the revision benchmark (T8) must show Jev's value on changes, or S2 is not justified.

## [2026-10-21] SAM 2 tracker deferred from the v1 critical path
**Decision:** v1 perception = MJev end-state checks and pointing on the current images; SAM 2 tracking comes with 5b perception faults and the contact-aware monitor.
**Why:** MJev answers concrete facts from the images without a tracker; π0.5² servos to the object itself; tracking matters for identity under occlusion / duplicates and for cheaper re-grounding.
**Rejected alternatives:** SAM 2 now (adds a GPU service before anything needs it).
**Risks / edge cases found:** Duplicate or occluded objects stay ambiguous until the tracker lands; the monitor stays contact-blind until then.

## [2026-10-21] Remote S1 default: HTTP/1.1 keep-alive + WebP-90 frames; HTTP/3 kept, aioquic not yet better
**Decision:** `RemotePolicy(..., codec="webp90")` over `HttpTransport` is the default remote path; `h3://` stays available.
**Why:** WAN (India → us-east, ping 216 ms, 60 interleaved requests, real π0.5²): keep-alive + WebP network p50 239 / p90 282 ms (round trip 559 / 597) vs keep-alive + raw 283 / 482, HTTP/3 + WebP 302 / 482, HTTP/3 + raw 324 / 902 (15/60 > 800 ms). WebP-90 actions are within π0.5²'s sampling noise (0.078 vs 0.070 lossless-to-lossless, 8 frames).
**Rejected alternatives:** HTTP/3 as default (aioquic inconsistent across runs, worse here); lossless frames (5.5× bytes, loss tail); JPEG-90 (40 KB, faster encode but larger).
**Risks / edge cases found:** Codec check on 8 frames only (recheck at scale in T8a); a native QUIC stack (quiche / msquic) untested; remaining lever = server-pushed chunks (one one-way trip) and a closer GPU region.

## [2026-10-22] Holds that answer a change persist until a later change (fixes D4)
**Decision:** ABORT / GUARDED_WAIT / ASK accepted in answer to a change (instruction, deviation, hold trigger) becomes the owner's standing `Hold`: no heartbeat, routine planning proposals refused, hold command kept leased by the owner, released only by a moving operation or COMPLETE answering a change; `snapshot.hold.interrupted` = the step to resume.
**Why:** `NativePlanner` (and any planner) answered the next heartbeat after a stop with SWITCH_TO, restarting the task the user stopped; the authority over holds belongs to the owner, not to each decider.
**Rejected alternatives:** Teaching each planner to respect holds (every decider would need it; one forgets → robot moves); keeping the interrupted step inside the interaction decider (state outside the owner's record).
**Risks / edge cases found:** An escalated change (interaction lane failed) answered by planning can still move the robot after "stop" (baseline behaviour, declared); in one-lane mode a newer routine proposal racing the stop is refused (tested).

## [2026-10-22] Safety holds end the command (`AckStatus.DROPPED`)
**Decision:** Guard holds for "owner lost" / "runtime error" drop the active and pending command with a `dropped` ACK; the owner treats it like rejected / expired (falls back, asks again); starvation holds keep the command.
**Why:** The hold kept the command "active" but refused its chunks; the owner renewed it forever and the robot stayed held (liveness bug found by the docs review).
**Rejected alternatives:** Clearing the hold on renew (lets a stale command move after the owner was gone); reusing REVOKED (the owner ignores owner-initiated revokes, so it would never re-decide).
**Risks / edge cases found:** A new command starts with `pause_hold` from the hold; recovery tested in unit tests only.

## [2026-10-22] Dialogue in the snapshot, utterances out of the owner (D2, D3, J6, J7)
**Decision:** Owner keeps the last `dialogue_turns` `Turn`s (user text; robot template + objects) in every snapshot; emits `Utterance` via `say`: `ACK` the moment an instruction arrives, and the template of every accepted templated operation (ASK, reports); the executor renders the words.
**Why:** Jev must read "on the plate" as the answer to its own question and must not repeat a question; the ≤ 200 ms acknowledgement must not wait for a model.
**Rejected alternatives:** Rendering text in the core (language / TTS are the app's); acknowledgements from Jev (model latency).
**Risks / edge cases found:** Robot turns carry template ids, so a decider sees its own wording only through `TEMPLATES`; the `ACK` is not a dialogue turn.

## [2026-10-22] Step-completion events: one generic end-state question, no per-skill parser
**Decision:** `StepEvent` (protocol) → owner → planning lane (trigger `step`) and keyframe evidence; `StepWatcher` posts it after `confirmations` consecutive P(done) ≥ `done_threshold` from a `Verifier`; `MJevVerifier` asks the arm-D question ("Robot step: '…'. Which is true right now?" done / possible / blocked) for any step text.
**Why:** The question was already validated on MJev (12/12 fp32 vs half-precision agreement); a per-skill postcondition parser would be a hand-written list per robot and task.
**Rejected alternatives:** Verb-template parser to postconditions (brittle, robot-specific); sim truth in the loop (evaluation only; kept as a declared harness spoof `TruthVerifier`).
**Risks / edge cases found:** Threshold and confirmations are declared tuned constants until T6 (MJev vs truth); S1-planner (subtask None) steps are not checked; π0.5²'s own subtask switch and phase boundaries as event sources are not built yet.

## [2026-10-22] Jev v1 interaction decider (J1–J8, D1, D5, D7)
**Decision:** `rojev/system2/jev`: legal acts generated in code per situation; one `Chooser` call (OneJev Choice); temperature `Calibration`; margin below threshold → ask (budget allowing) else wait; `max_asks` and no repeat without a user turn; deviation within `min_dwell_s` of a step start → CONTINUE; `hold` trigger → report once then keep holding (D7); REVISE through a `Reviser` (default `S1Replans` = SWITCH_TO(None), D1).
**Why:** Spec §9 and plan p. 4 ("generate complete, compatible candidates in code"); the decider is stateless so holds, dialogue and progress stay in the owner's record.
**Rejected alternatives:** A new `OpKind.REVISE` (the owner would need planner knowledge); asking OneJev for budgets or dwell (deterministic rules, no model needed).
**Risks / edge cases found:** Margin 0.2 and T = 1 are declared constants until a labelled set is fitted (D5: needs GPU + scenes); ASK has one generic template until perception supplies object ids; OneJev latency vs `interaction_timeout_s` unmeasured (D6).

## [2026-10-22] Deferred planning triggers are replayed after the interaction decision
**Decision:** While an interaction request is pending the latest planning trigger is kept and sent once that decision is applied; an escalation replaces it.
**Why:** Step events or rejected-command re-asks were dropped while a change was pending, recovered only by the heartbeat (now suppressed while holding).
**Rejected alternatives:** Queueing every deferred trigger (stale ones would be decided on old state).
**Risks / edge cases found:** Only the latest trigger survives; its snapshot is taken after the interaction decision (intended).

## [2026-10-22] Reaffirm is always `immediate`; CONTINUE logs whether it renewed
**Decision:** A reaffirm ignores the configured transition (spec §4); the `continue` event carries `renewed`.
**Why:** With `pause_hold` configured a reaffirm paused the robot, contradicting the spec; the log claimed renewals that did not happen.
**Rejected alternatives:** Changing the spec to follow the code.
**Risks / edge cases found:** `changes_answered` still counts an unrenewed continue as an answer (a decision was made).

## [2026-10-22] MJev client over `rojev.transport`; executors freeze the setup heap
**Decision:** `RemoteMJev` uses `open_transport` (kept connection; system2 → transport allowed); `realtime.py` runs `gc.collect(); gc.freeze()` before starting threads.
**Why:** MJev calls paid a new connection each (680 vs 255 ms measured for π0.5²); a full GC pass in the RoboCasa process took 45–96 ms (research brief 2026-10-22).
**Rejected alternatives:** urllib per request; disabling GC (leaks) or `gc.freeze` in the core (process policy belongs to the app).
**Risks / edge cases found:** Objects created after the freeze are still collected normally; whether full collections fire during runs is unmeasured.

## [2026-10-22] Replay-safe retries: client request ids, server replay cache in `rojev.transport`
**Decision:** Both transports send `X-Request-Id` per logical request and reuse it on their one retry; `rojev.transport.Endpoint` wraps any service handler with `ReplayCache` (one execution per id, copies wait for the running first, last 256 completed kept, 409 on id reuse with another body, exceptions → stored 500).
**Why:** A connection lost after the service ran `/act` made the retry run π0.5² (and its planner state) twice; only the server can know it already ran the request.
**Rejected alternatives:** No retry (a tunnel restart surfaces as an S1 error); retry only before sending (the client cannot know whether the server got it); a TTL cache.
**Risks / edge cases found:** A retry is lost from the cache only if > 256 other requests complete in between; MJev's service does not use `Endpoint` yet (its env lacks `rojev`; `/ask` is a pure function, so a replay is harmless).

## [2026-10-22] Admission before the body is a transport-server property
**Decision:** `Endpoint.admit(headers, length)` (token, declared length vs `max_body`) runs on the headers in `HttpServer` and `H3Server`; HTTP/1.1 discards a refused small body unbuffered and closes on a large one; HTTP/3 answers 413 and stops the stream past the limit; services keep only routing.
**Why:** `serve.py` buffered the whole body before checking the token; `H3Server` accumulated unbounded stream data on a public UDP port.
**Rejected alternatives:** Per-service checks (MJev already diverged); refusing before draining small bodies (client sees a broken pipe, not 401).
**Risks / edge cases found:** aioquic closes the whole connection on a content-length / data mismatch; chunked HTTP/1.1 bodies refused (411).

## [2026-10-22] Wire v2: K samples in one request, drawn in one forward batch
**Decision:** `wire.VERSION` 2: `samples` K in the request, K×n×d `actions` + per-sample `timings_ms` / `info` in the response, one `/act` endpoint; `RemotePolicy.act_samples` = one request; `serve.py` plans the subtask once and repeats the conditioned batch K times (`Draws`).
**Why:** K sequential round trips made L ≈ K × single latency, so the consensus floor exceeded the chunk.
**Rejected alternatives:** A separate `/act_samples` (two paths for one operation); sharing one prefix KV cache (LeRobot's `sample_actions` does not expose it); K adapter calls.
**Risks / edge cases found:** Prefix compute and memory scale with K; latency, memory, draw diversity and the real adapter accepting a K-row batch are unmeasured on the GPU (procedure in remote-models.md).

## [2026-10-22] ConsensusCommit switches: per-step strict majority, ties keep the value in effect (incl. the robot's)
**Decision:** Per step and switch the strict majority's value; on a tie the value in effect stays (previous step's, and at the first step the switch in the action last applied, passed as `plan(..., last)` from `ExecutionRuntime.last_action()`); only with nothing applied yet is a first-step tie reported `switch without a majority`.
**Why:** The largest identical group was a plurality (2 of 5 closing first decided the grasp); tied groups merged by nearest mean had no majority; a first-step tie needs the robot's switch, which only the runtime knows.
**Rejected alternatives:** Requiring odd K (three-way ties exist); stopping before a tied switch inside the floor (starves / livelocks); leaving the first-step tie as "any candidate".
**Risks / edge cases found:** `last` is the last applied action, not the last queued one; queued actions due before the chunk takes over are not seen (switches are frozen outside running chunks anyway).

## [2026-10-22] S1Runner: refills only for the supplied command; floor over chunk counted
**Decision:** `refill_now(cmd_id)` taken only when `cmd_id` is `runtime.target()`, else counted in `refill_ignored`; requests whose measured floor exceeds the chunk counted in `floor_over_chunk`.
**Why:** Deviations the owner logs as stale still forced refills; starvation from too-short chunks was silent.
**Rejected alternatives:** Filtering in `realtime.py` (every embedding repeats it); counting floor ≥ chunk (floor = chunk is still gap-free).
**Risks / edge cases found:** A refill taken for a command replaced before the next step is dropped and counted.

## [2026-10-22] One codec default in `wire`
**Decision:** `wire.DEFAULT_CODEC = "webp90"`, used by `encode_request` and `RemotePolicy`; a `codec` key in the service JSON still overrides.
**Why:** `RemotePolicy(codec="raw")` contradicted the 2026-10-21 decision.
**Rejected alternatives:** Codec only in the service JSON (two sources of truth).
**Risks / edge cases found:** Codec check rests on 8 frames (recheck in T8a).

## [2026-10-22] An unanswered change holds the robot; it never escalates to planning (reverses the 2026-10-21 escalation)
**Decision:** Interaction failure, timeout or refused proposal → GUARDED_WAIT `UNANSWERED` as a standing hold (interrupted step kept), said to the user; the interaction lane is asked again on the hold recheck and Jev decides the original change then; the planning lane never answers a change.
**Why:** Escalation let a planner (e.g. π0.5² re-prompted with "stop") answer an instruction it cannot read as an act, so an unread "stop" could move the robot.
**Rejected alternatives:** Escalating with planning limited to non-moving answers (a filter on the planner, not a rule about authority); waiting longer without holding (the robot keeps moving on an unread change).
**Risks / edge cases found:** A permanently failing or too-slow interaction lane keeps the robot held; `interaction_timeout_s` must be set above Jev's measured latency (D6, GPU); `--hold-recheck` must be set for the re-ask.

## [2026-10-22] π0.5²'s own subtask switch is a step event; phase boundaries are not
**Decision:** `S1Runner` posts `StepEvent(source="s1-planner", step=previous subtask)` when the policy's reported subtask changes under a command without a subtask; `StepEvent` gains `step`; when S1 plans its own steps the event itself is the keyframe; `realtime.py` heartbeat defaults to 0.8 × lease when step events cover progress (verifier on, or native π0.5²), else 1 s.
**Why:** Native mode had no progress record and no event source; π0.5²'s switch is its own completion judgement.
**Rejected alternatives:** Phase boundaries (motion slowing) as completion events (a slowdown is not completion; one planning request per chunk); parsing `ActionChunk.info` in the owner (S1 detail does not belong there).
**Risks / edge cases found:** A switch may mean "abandoned", not "done" (π0.5² does not say which); the planner's subtask text is free text, not a skill id.

## [2026-10-22] Jev calibration: T by NLL, margin by target accuracy, cases recorded from live runs
**Decision:** `fit_margin` = smallest margin whose decided cases reach `target_accuracy` (else 1.0: always abstain), `fit_calibration` = T + margin + coverage; `--record-interaction DIR` saves each Jev model decision (cameras, situation, legal options) with a `labels.jsonl` line to fill; `calibrate_jev.py` re-scores labelled cases with OneJev and writes the `Calibration` JSON.
**Why:** The margin was a declared constant; it decides when Jev acts alone, so it must come from measured accuracy, and cases must be the exact inputs Jev sees.
**Rejected alternatives:** Fitting the margin by NLL (NLL does not say when to abstain); synthetic cases without scenes (OneJev reads the images).
**Risks / edge cases found:** Labels are human judgements; few cases per act make the margin noisy; re-scoring needs the GPU.

## [2026-10-22] Harness invariants read the robot's switches from its ActionSpec; `dropped` ends a command
**Decision:** `switches_held(run, spec)` replaces the Panda-indexed gripper check; `no_stale_action` treats `dropped` like revoked / expired.
**Why:** The gripper index (6) was PandaOmron's; any robot plugs in through its `ActionSpec`.
**Rejected alternatives:** A per-robot invariant module.
**Risks / edge cases found:** Records must come from the same spec the run used (scenarios pass `PANDA_OMRON`).

## [2026-10-22] Precondition checks, contact-aware monitor and object-naming questions wait for perception
**Decision:** Not built now: proposals carrying preconditions stay refused; the monitor stays contact-blind; ASK keeps one generic template.
**Why:** Each needs object identity or contact evidence (tracker / force), which does not exist yet; building them now would be untestable code without a real signal.
**Rejected alternatives:** MJev-only precondition checks inside the owner (the owner never waits for a model); contact phases from hand-written rules.
**Risks / edge cases found:** `blocked` keeps failing; Jev cannot ask "which object?" by name.

## [2026-10-22] Revision-benchmark scorecard (T8)
**Decision:** `rojev/eval/scorecard.py` scores runs from the owner, tick, decision and summary records against authored injections (time, kind, expected act, goal text): answer / execution latency, act correctness, wrong-goal commitments, wasted motion (ticks), pause, ask precision, act ECE; aggregated per arm; `RecordingDecider` logs `scores` so ECE is computable.
**Why:** T8 needs one robot-agnostic definition for every arm; the owner's events with their `trigger` are the only authoritative record of what answered a change.
**Rejected alternatives:** Arm path length from tick actions (commanded deltas, robot-specific dims); classifying acts from decider reason text; counting heartbeat decisions as answers.
**Risks / edge cases found:** Subtask-less arms (native, Jev + `S1Replans`) get no goal metrics; `task_success` measures the starting task; a resume scores as revise; no live scored run yet.

## [2026-10-22] The review window is the interaction deadline (live bug: every change went unanswered)
**Decision:** With an interaction lane, a command under review keeps its lease until `interaction_timeout_s` (`StateOwner.review_window_s`); `review_s` applies only without one.
**Why:** GPU suite 2026-10-22: OneJev answered in 1.15 s, the 1 s review lease had already expired, the answer was refused as stale and the robot held "unanswered"; two deadlines for one wait cannot disagree if there is only one.
**Rejected alternatives:** Raising `review_s` by hand (drifts from the deadline again); accepting answers about an expired command (the robot may already be doing something else).
**Risks / edge cases found:** The deadline must stay above Jev's measured latency (1.1–1.5 s over the WAN; suite sets 1.5 × the slowest measured).

## [2026-10-22] An utterance never becomes the task text; only an accepted decision adopts new task text
**Decision:** `UserInstruction` joins the dialogue and bumps `instruction_rev`; the task text S1 is conditioned on (`snapshot.instruction`, `AcceptedGoal.instruction`) changes only when an accepted proposal carries `task` (`Verdict.task`): Jev's REVISE (the `Reviser` returns operation + task text), and baselines that re-prompt S1 (native, always-continue) say so explicitly; deciders read `snapshot.latest_utterance`.
**Why:** GPU suite 2026-10-22: "wait" and "go on" became π0.5²'s task text (in native mode its whole plan).
**Rejected alternatives:** Concatenating task and utterance (a heuristic merge); keeping the utterance as task (the bug).
**Risks / edge cases found:** A REVISE adopts the user's words verbatim ("actually, put the steak on the plate") and loses context from the original task; a generative rewrite would be better (not v1).

## [2026-10-22] An S1 request's deadline is one chunk duration
**Decision:** `S1Runner` passes `timeout_s` = the last chunk's horizon × dt to `Policy.act/act_samples`; `RemotePolicy` hands it to the transport, which bounds the whole call (retry included), closes the connection past it and raises `TimeoutError`; the runner counts it (`timeouts`) and re-requests at once on the newest observation; before the first chunk the transport's own timeout applies.
**Why:** A chunk is placed at its observation's time, so an answer older than its duration is entirely in the past; a 105 s network stall starved the robot for the rest of a run (and run shutdown waited for it).
**Rejected alternatives:** A fixed short timeout (wrong for slow policies or K samples); k × measured latency (abandons answers still useful); retrying the same request (stale observation).
**Risks / edge cases found:** The server may still run an abandoned request (its answer is discarded); a policy slower than its own chunk duration times out every time (that is the floor-over-chunk condition, already counted).

## [2026-10-22] π0.5²'s subtask switch is a claim; it counts as progress only when a check confirms it
**Decision:** `StepEvent.checked`; with a verifier the runner's switch goes to `StepWatcher.claim`, one end-state check of the step it left, posted as checked only if P(done) ≥ threshold (else `unconfirmed`); without one the owner gets the unchecked event (planning asked, no keyframe); native-mode keyframes only from checked events.
**Why:** GPU suite: π0.5²'s planner switched freezer ↔ fridge 18 times in 60 s, each switch recorded as progress.
**Rejected alternatives:** Debouncing switches by time (a heuristic, still not evidence); dropping the source (native mode would have no event at all).
**Risks / edge cases found:** Confirmation costs one MJev check (~0.46 s) per switch; MJev's positives are still unmeasured (T6).

## [2026-10-22] Jev accuracy: wording first, then calibration, then compute (adaptive depth / LEV) only if T9 says so
**Decision:** Order: (1) the model told what the user just said apart from the task (done: `task` / `user_just_said` in the situation) and act options worded as what the user wants (next, measured on held-out cases); (2) labelled cases → `calibrate_jev.py`; (3) T9 split of the remaining errors into evidence errors (the model lacked information) vs compute errors (a bigger model or more depth fixes them); adaptive depth / LEV only for the second kind; training (S2-B…) last.
**Why:** GPU suite: OneJev chose "continue" 8/8 for "put the steak on the plate" while picking (the continue option read as compatible), but "revise" p 0.87 in the gate run where the situation was described differently: a wording/inputs failure first, not shown to be a depth failure.
**Rejected alternatives:** LEV first (no evidence the errors are compute-bound); training first (no labelled set yet).
**Risks / edge cases found:** Prompt tuning on 14 cases overfits; wording changes need a held-out set.

## [2026-10-22] Any robot: one assembly layer (`rojev/app`), robots as executor plug-ins (spec, not built)
**Decision:** D-A1 `rojev/app` is the only assembly (top layer); D-A2 `Embodiment` = ActionSpec + EffectModel + cameras + start command, the backend built by the executor; D-A3 recording wrappers + Speaker move to `rojev/app`, invariants to `rojev/eval` (eval → robot); D-A4 the record format is the contract every run tool reads; D-A5 the SO-101 proof uses a declared joint-space stand-in S1. Specs: `docs/plan/specs/any-robot/`.
**Why:** The core is robot-agnostic by contract, but the assembly is one 500-line RoboCasa script, so a second robot means copying it.
**Rejected alternatives:** A base executor class to subclass (inheritance across repos/envs, fragile); keeping assembly per executor (the copy problem); a config-file-driven launcher (hides the wiring a builder must reason about).
**Risks / edge cases found:** The owner / runner / watcher closure order must be preserved exactly; record field names are a contract (scenarios, GPU suite, calibration, scorecard read them).

## [2026-10-22] Scenario matrix feeds the T8 scorecard
**Decision:** Each scenario declares its injected changes (`Scenario.changes`: kind, expected act, goal texts), paired in order with its `--say` / `--block-at` times; the matrix scores every case with changes under arm = `--interaction` else `--arm`, writes scorecard.md / .json beside report.json; `--score-only RUN_DIR` re-scores past matrices; kinds gain `pause` / `resume`.
**Why:** Expected acts are facts of each case's design, not inferable from text; one source for times keeps injections and arguments from drifting.
**Rejected alternatives:** Inferring kind / act from utterance text (heuristic); declaring times twice (drift); scoring past runs without checking what they injected.
**Risks / edge cases found:** A run whose recorded schedule differs from the current case is skipped, not mis-scored; arms pool faults and baselines sharing a decider; no phases declared yet (T8 needs phase-placed injections).

## [2026-10-22] Scorecard: an unanswered change is no decision; pauses judged by what the change asked; conditions are arms
**Decision:** The owner's safety hold after an `unanswered` event scores act `unanswered` (never correct) with an arm-level unanswered rate; pause is aggregated as `pause_when_moving` (expected continue / revise) and `hold_when_asked` (expected wait / ask / abort); scenario arms carry their conditions (`+revoke`, `+s2-slow`, `+s2-fail`).
**Why:** The first table scored S2 failures as correct waits (rules 1.00), pooled faults and the revoke baseline with the nominal system, and averaged right holds with wrong freezes.
**Rejected alternatives:** Excluding fault cases from scoring (hides the fallback rate); one pause number.
**Risks / edge cases found:** Answer latency has one control tick of resolution (robot clock); runs from before the 2026-10-22 hold rule (155022) score their escalated planning answers as wrong, which is what they did.

## [2026-10-22] MJev scope: the release signal of commit-and-verify (pending A/B), later the grounder for a tracker
**Decision:** v1: MJev's only loop job is the windowed step-done check that releases a committed π0.5² subtask, kept only if it beats OneJev on truth frames; grounding (points → SAM 2.1 Tiny) comes with the tracker; never a per-frame verdict that moves the robot or a metric estimator; v2: input to Jev's spatial / progress heads (one backbone).
**Why:** Today it changes no behaviour; π0.5²'s subtask head is memoryless (trained on task + image only), so a verified done signal is what a training-free commit-and-verify fix needs; the plan assigns completion to "Jev auxiliary heads or a validated monitor" and identity to a tracker.
**Rejected alternatives:** Dropping MJev before the A/B (loses the only candidate verifier with zero false "done" so far); keeping it as an always-on judge (literature: judges lean to "success", per-frame queries add false judgements).
**Risks / edge cases found:** Its positives are unmeasured; both checkers may exceed 5% false "done" (then the release signal must come from tracked predicates).

## [2026-10-22] Commit and verify: a self-planning S1 is held on its planned subtask until a check confirms it
**Decision:** `PlannedSubtask` (S1 runner → owner, `snapshot.planned`); `CommitVerifyPlanner` (`rojev/system2/planning.py`, `--arm commit`) commits S1 to its planned subtask (`SWITCH_TO` the text), releases on a checked step event about it (the owner records a checked keyframe), never re-commits a verified step (counted `reentries`), and releases an unverified commitment after `max_commit_s` (declared liveness bound, counted `expired`); `Keyframe.checked` separates verified progress from decider judgement.
**Why:** π0.5²'s subtask head is memoryless (trained on task + image only, recipe `subtask_mem.yaml`), so it flips when the image is ambiguous (18× in 60 s); it is steerable by subtask text, so pinning the step stops the flips reaching the robot without training.
**Rejected alternatives:** Feeding history into its subtask prompt (out of distribution: not trained); slowing its re-planning (`chunks_per_regen`; slower to react, not evidence); training its subtask head with memory now (v2).
**Risks / edge cases found:** Release depends on the checker's recall (unmeasured until the done-check A/B); steps with no visual cue (microwave "on") can only be released by the bound; re-entries of verified steps are counted, not prevented.

## [2026-10-22] Done-check A/B on simulator-labelled frames decides the checker
**Decision:** `done_check_ab.py build` sets kitchen states (object placed, door closed, appliance on), labels each checked step with the task's own completion test, drops states whose predicate fails (2 of 80) and never relabels; `score` asks MJev and `OneJevVerifier` the identical question; gpu_suite steps `done_check` and `commit_ab`.
**Why:** MJev's recall was never measured (0 positives in live runs); a checker must be chosen on balanced accuracy and false "done", the release signal's failure modes.
**Rejected alternatives:** Waiting for π0.5² to complete steps (it completed none); hand-labelled frames (truth is available from the simulator).
**Risks / edge cases found:** "pick up" steps are excluded (holding needs a real grasp); "turned on" has no visual cue (both checkers should fail it; reported per step type); teleported placements may look unlike placed-by-robot scenes.

## [2026-10-22] MJev out of the v1 loop: it fails the done-check gate on both question forms
**Decision:** MJev is not loaded or used by default; commit-and-verify keeps the bound as its only release until a measured completion signal exists; the next completion signal to build is tracked object state, and π0.5²'s planned subtask must be checked against the task before it is committed.
**Why:** On 222 simulator-labelled pairs MJev scored balanced accuracy 0.50 ("is this step done?") and 0.45 (yes / no postconditions), OneJev 0.52 / 0.52; live, commit cut subtask switches 19 → 2 but verified nothing and pinned a wrong first pick ("put steak in fridge").
**Rejected alternatives:** Keeping MJev for its zero false "done" (it never says "done" at all); lowering the threshold (no separation to threshold on).
**Risks / edge cases found:** The 256 × 256 multi-camera views may be the limit rather than the models (higher-resolution single-camera A/B pending); teleported placements may differ from robot-made ones; pointing (MJev's other use) is untested.

## [2026-10-22] OneJev sees each camera as its own image (one request), never views glued side by side
**Decision:** `onejev.camera_views` names `<image:1>…<image:k>` as separate views of the same moment; the interaction chooser and `OneJevVerifier` send the image list in one request instead of one concatenated 768 × 256 picture; `OneJevVerifier` asks every step as one question of that request (one call per frame).
**Why:** The concatenated picture gave the model no boundary between views (a container's edge in one view runs into the next); MJev already received separate images; qev takes several images per request, so it stays one pass.
**Rejected alternatives:** Drawing separator lines into a concatenated picture (still one image to the vision encoder, a pixel cue standing in for real image boundaries); one request per camera (three passes).
**Risks / edge cases found:** Three images cost more vision tokens than one at a third of the width each (latency to measure); earlier OneJev numbers (done-check, Jev runs) were taken with the glued view.

## [2026-10-22] π0.5²'s native planner memory stays off
**Decision:** `serve.py --planner-memory` remains an opt-in A/B flag; the default unit runs with memory off.
**Why:** A/B on MoveFridgeToFreezer (2 seeds × 60 s each way, `results/memory-ab/20261022`): subtask switches 16 / 10 off vs 21 / 11 on, no success either way, S1 p50 525 → 575 ms; the memory notes are false ("I placed it in the freezer" while the steak never got there), because the subtask head is not trained on memory and the note is written from the planned subtask, not observed state.
**Rejected alternatives:** Turning it on by default (no gain, slower, false notes); feeding its notes to S2 as progress (unverified claims, plan: never promote a prediction to "completed" because another model echoed it).
**Risks / edge cases found:** 2 seeds on one task; the memory head may help where subtasks are visually distinct.

## [2026-10-22] Inference delay: the runtime tells a policy what runs while it thinks; the policy decides how to use it
**Decision:** Core: `Continuation(actions, delay_steps)` from `ExecutionRuntime.queued_from(cmd, capture.t)` and ⌈L/dt⌉, passed by `S1Runner` only to a policy that declares `accepts_continuation` (wire v3 carries it); placement at observation time and skipping unchanged. Model-specific use stays in the executor: `serve.py --rtc H` feeds it to LeRobot's real-time chunking for π0.5² (normalized by the checkpoint's own normalizer).
**Why:** Real-time chunking needs the actions the robot will actually execute during inference; only the runtime knows them (commitment may keep part of a chunk; revokes, new commands and skips change the queue); `queued_from` costs 8.7 µs on the runner thread vs a 50 ms control period.
**Rejected alternatives:** An executor-side client that remembers its last chunk (wrong after partial commits, revokes or new commands; no observation time in `act`, so wall-clock guesses); RTC in the core (model-specific); passing the continuation to every policy (stand-ins would have to accept and ignore it).
**Risks / edge cases found:** RTC guidance runs under `no_grad` (autograd) instead of `inference_mode`: server latency to measure; the execution horizon (25 steps) is a declared choice; wire v2 clients are refused.

## [2026-10-22] Lockstep mode for the live harness (diagnostic)
**Decision:** `realtime.py --lockstep`: one thread alternates S1 supply and one control tick, so simulated time stops while S1 thinks; real time stays the default.
**Why:** π0.5² completed no task in real time; lockstep is how it was trained and evaluated, so it separates the policy's own skill from the cost of the delay; LeRobot's own runner needs the RoboCasa assets on the GPU box (absent; the simulator runs on the Mac).
**Rejected alternatives:** `drive.py` / lerobot-eval on the box (tens of GB of scene assets); slowing the simulator (still couples sim time to latency).
**Risks / edge cases found:** S2 threads run on the simulated clock and may see more ticks per decision than in real time.

## [2026-10-22] SAM 2.1 measured as a completion signal; truth rendered after the run
**Decision:** `services/sam2` (SAM 2.1 Hiera Tiny, transformers, pinned) with `/segment` and streaming `/track/*` (state on CPU, idle sessions closed, errors answered); measured offline: `--record-frames` saves cameras + simulator state, `sam_check.py track` renders truth in a fresh process. Not in the loop.
**Why:** Still frames: masks from correct points IoU 0.64 (0.82 for objects ≥ 100 px), MJev pointing hits 41%, 2D "placed inside" with perfect masks 0.68 balanced (freezer 90% false "done"), 44/74 placed objects unseen. Live tracking: IoU up to 0.86–0.96 when started on a visible object, but a bad first mask never recovers (freezer run IoU 0.00) and re-acquisition is 3/11; π0.5² placed nothing, so completion is unvalidated.
**Rejected alternatives:** Rendering truth inside the live process (offscreen renderer returns blank images there); GPU-resident session frames (7.5 GB, out of memory after 177 frames).
**Risks / edge cases found:** Needs grounding (no reliable pointer yet), a first-mask check, depth or "seen entering then occluded" for containment, and placements made by robot motion.

## [2026-10-22] π0.5² (pi052_robocasa) cannot execute our composite tasks; Xiaomi-Robotics-1 becomes the RoboCasa S1 candidate (EVIDENCE WITHDRAWN: see the render-context entry below)
**Decision:** Stop building on `pi052_robocasa` for RoboCasa composite tasks; evaluate Xiaomi-Robotics-1-RoboCasa365 (5B, Apache-2.0, 57.28% on target50) as S1 on the published target50 test cases (their seeds and horizons) in our own harness, lockstep first; briefs `docs/research/2026-10-22-s1-candidates.md`, `-xr1-integration.md`.
**Why:** π0.5² completed 0/14 episodes (real time, lockstep, full budget) and 0/4 even with an oracle planner (picks 2/4, places 0/4), matching RoboCasa365's π0.5 composite rate (7.1%); XR-1 is the strongest verified open checkpoint on this benchmark.
**Rejected alternatives:** An external planner over π0.5² (the oracle shows execution is the limit); their own evaluator (we test our runtime, not theirs); RTC (+150 ms, no measured gain).
**Risks / edge cases found:** Our two tasks are not in target50 (no published number); a 16-action chunk (0.8 s) is near our real-time latency budget; lockstep refill 0.05 executes ~14 of 16 actions per query (theirs 16).

## [2026-10-22] History-conditioned policies get their frames from the executor; the generic server moves to the core
**Decision:** `history.py` (executor): `FrameHistory` fed by the tick hook, `HistoryPolicy` sends `length` frames `interval` apart as `<cam>@k` images + stacked states on the existing wire; `rojev/system1/service.py` (`PolicyService`, `serve_policy`) is the server half of `RemotePolicy`, shared by the π0.5² and XR-1 services; `to_action_space` moved to `embodiment.py` (robot knowledge, applied on the robot side for XR-1).
**Why:** Frame history is a model requirement (handled outside the core, as the user asked for delay); the newest frame is matched by identity, so the window is exact even when the runner reads a frame before the hook files it; the server routing was identical for both models.
**Rejected alternatives:** `Observation.history` + wire v4 in the core (proposed by the integration agent; model-specific need in the core); a server-side history buffer (the server sees one frame per chunk, never t−2/t−4/t−6; breaks retry safety); upstream pickle socket server (remote code execution).
**Risks / edge cases found:** XR-1's environment pins transformers 4.57.1, so it installs only the core's S1-side dependencies (`dependency-metadata` override); flash-attn 2.8.3's wheel names a local version its metadata lacks, so `uv sync --frozen --check` never passes and `gpu_setup` stops at `env:robocasa-xr1` (service started by hand with the same launcher for this test; packaging fix open); SDPA is not supported by the model (flash-attn required, Ampere+ GPU).

## [2026-10-22] Camera images in live runs were corrupted by robosuite's render-context teardown; fixed in the RoboCasa executor
**Decision:** `executors/robocasa-pi052/render_context.py` installs a teardown that frees a robosuite render context in its own GL context (then restores the current one); `RoboCasaBackend` installs it before building the environment; regression test `tests/test_render_context.py`. Every live result from `realtime.py` before this fix is withdrawn until re-run: π0.5² memory A/B, delay A/B (lockstep / real time / RTC), full-budget lockstep, oracle, SAM live tracking, XR-1's first 1/6.
**Why:** robosuite frees a replaced simulator's MuJoCo context (`mjr_freeContext`) in whatever GL context is current, deleting the live simulator's framebuffer and textures (same ids); `realtime.py` calls `gc.collect()` before `gc.freeze()`, so from the first step the policy saw noise (neighbour-pixel difference 70–84 vs 5 for real frames; reproduced with a stand-in policy, no network, lockstep and real time). Still-frame sets (done-check, SAM check) never collect: 0 of 444 images corrupt, those results stand.
**Rejected alternatives:** Dropping `gc.collect()` (any later automatic collection triggers the same; the cause is the teardown); editing the vendored robosuite clone (lost on re-clone; the box re-clones at the pinned commit); recreating the live context after a collection (stale contexts appear on every reset).
**Risks / edge cases found:** The fix is GLFW/CGL-specific (macOS; EGL contexts have no `_context` and keep robosuite's behaviour); remove when robosuite frees in its own context; earlier "the checkpoint is the limit" and "delay is not the cause" conclusions are unverified until re-run.

## [2026-10-23] Lockstep stays the simulator evaluation mode for now
**Decision:** Sim evaluations run in lockstep (`--lockstep`), with real time reported alongside; hardware requires real-time passes first (`hardware-port.md`).
**Why:** S1 runs on a remote GPU (0.35–0.55 s vs an 0.8 s XR-1 chunk) and published benchmarks are scored in lockstep, so lockstep is the only fair model comparison until S1 runs next to the simulator / robot.
**Rejected alternatives:** Real time only (measures our network, not the model); lockstep only (hides the delay problem the robot will have).
**Risks / edge cases found:** Lockstep results overstate real-world performance; S2 threads run on the simulated clock and see more ticks per decision than in real time.

## [2026-10-23] Every run checks its camera frames; flash-attn for XR-1 comes from its sdist
**Decision:** `harness/frame_check.FrameCheck` samples frames every second in `realtime.py`; noise (neighbour difference > 25) or blank (std < 3) frames invalidate the run (summary, `frames_real` invariant, not scored). XR-1's env takes flash-attn 2.8.3 from PyPI, built with `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE` against the env's torch, so its setup.py fetches the matching prebuilt wheel under its own version; the hook's readiness imports `torch` then `flash_attn_2_cuda` (the kernels link against torch's libraries).
**Why:** The corrupted-frame bug was silent for a day; thresholds sit far outside 963 real frames (≤ 7.4, std ≥ 26.8). The prebuilt wheel used directly names `+cu12torch2.8…` while its metadata says 2.8.3, so `uv sync --frozen --check` never passed and gpu_setup could not provision XR-1.
**Rejected alternatives:** Checking every tick in the control loop (~5 ms per tick); a frozen-frame rule (a still scene is legitimately identical); compiling flash-attn (needs nvcc, ~30+ min); `--inexact` env checks (hides real drift).
**Risks / edge cases found:** If the prebuilt download fails, setup.py silently builds without kernels (caught by the readiness import); verified on the Mac lock only until the GPU box is back; on hardware the check moves to the robot layer with staleness and a guard hold.

## [2026-10-23] Lockstep also stops simulated time while S2 decides
**Decision:** In `--lockstep` the run loop does not tick while the planning or interaction lane is busy, so every model's computation takes zero simulated time.
**Why:** Smoke test with XR-1 + Jev: OneJev's ~1.1–1.5 s wall latency became 1.75–1.85 s of simulated time against a 1.0 s interaction deadline; both answers (wait p 0.88, then resume) arrived late, the hold turned "unanswered" and "go on" was revised into the task. With S2 paused too: wait at 3.05 s, resume at 6.05 s, success at 10.5 s.
**Rejected alternatives:** Longer interaction deadlines in lockstep (still depends on simulator speed); pausing only for S1 (S2 deadlines meaningless).
**Risks / edge cases found:** Lockstep hides S2 latency as it hides S1 latency; real-time runs remain the measure of responsiveness.

## [2026-10-23] First fitted Jev calibration, from live interactive decisions
**Decision:** T = 0.44 and abstain margin 0.10 (target accuracy 0.8), fitted on 30 labelled interaction decisions from the clean-frame interactive runs 2–4 and the run-1 replay (`results/jev-calibration/20261023/{labels.jsonl,calibration.json}`; labels = the typed intent; 2 ambiguous lines excluded); used with `realtime.py --jev-calibration`.
**Why:** At T = 1 / margin 0.2 Jev asked too often (top-1 accuracy 0.77, but many abstentions on clear intents); the fit sharpens (NLL 0.636 → 0.516) and keeps 87% of decisions at 81% accuracy. Target 0.9 would need margin 0.72 and abstain on half the turns.
**Rejected alternatives:** Target 0.9 (asks every other turn); fitting on the recorded `InteractionCases` (frames corrupted before 2026-10-22's fix).
**Risks / edge cases found:** 30 cases, one scene, labels by us; the remaining errors are semantic (a speed remark revised into a task, "close the cabinet door" not taken as a task, stop scored as wait), which calibration cannot fix: outcome-level abstention and REVISE over a capability list address them.

## [2026-10-23] v1 interactive testing stops; next build = capability signal + completion signal
**Decision:** No further v1 test rounds. Build the capability signal (what this S1 can do in this scene, as canonical task sentences; REVISE chooses among them) and the completion signal (trained progress / done estimate) before the next interactive or T8 run.
**Why:** Runs 2–5 (XR-1 + Jev, clean frames) fail on the same two gaps: trained tasks such as "close the cabinet" were never offered to Jev, remarks became tasks, and after success (36.75 s in run 5) the robot kept executing the finished task, was "resumed" into it and drove in circles.
**Rejected alternatives:** More wording / calibration tuning (run 5: calibration made wrong reads decisive); simulator success as the completion signal (sim-only spoof).
**Risks / edge cases found:** The completion signal needs training data and a GPU run; capability preconditions (e.g. a door must be open to close) must come from perception on a real robot.

## [2026-10-24] Jev revises to a trained task sentence, never the user's words
**Decision:** REVISE asks a second typed question (`task_choice.CanonicalTask`): one of the trained task sentences S1 can run here (the run's instruction + `--skills FILE`) or `none`; `none` or a low margin → ask, else wait. `S1Replans` (raw utterance as the task) removed.
**Why:** Runs 1–4: a remark and "make it fast pls" became XR-1's prompt, typos reached it, and "close the cabinet door" could not be chosen because no option named it.
**Rejected alternatives:** Better option wording (adds no missing option); an open list or a generated rewrite (v1 scope is trained tasks only; a rewrite needs a generative model, v2).
**Risks / edge cases found:** The list caps the repertoire (new compositions impossible until v2 adds a rewrite fallback); which listed task can run in the current state is not checked (critic's feasibility head); the task question has its own, uncalibrated, defaults.

## [2026-10-24] One interaction act per effect; no retry
**Decision:** Legal acts come from a situation table (moving · idle · held · deviation); retry removed; a deviation offers only continue or wait (hold until the user decides).
**Why:** XR-1 takes only a task sentence, so a retry is the same sentence (= continue) and only dilutes the Choice; deviation-revise used a stale utterance as the task.
**Rejected alternatives:** Separate intent classes for comment and speed request (same effect, splits probability).
**Risks / edge cases found:** A policy that can retry a step (a skill library) needs the act back, behind that policy's capability.

## [2026-10-24] A late interaction answer is applied as decided, not re-decided
**Decision:** At the interaction deadline the owner holds (`UNANSWERED`) and awaits the answer; when it arrives it is accepted if epoch, revision and supersession allow, the placeholder hold is not a new base, and CONTINUE re-issues the operation in force when the change was decided. No hold recheck while awaited; a newer interaction request ends the wait. The scorecard counts the first decider's answer, noting a late one.
**Why:** OneJev takes 1.1–1.5 s against a 1.0 s deadline; the answer was refused (its base was the placeholder) and the same change decided again, so in real time every change cost two decisions.
**Rejected alternatives:** Raising the deadline (the robot then keeps moving on an unread "stop" longer); deciding again on the recheck (doubles latency, can disagree with itself).
**Risks / edge cases found:** A decider that hangs without failing keeps the robot held (safe, needs a restart); the answer is applied to a ~1.5 s old situation (the robot was held meanwhile).

## [2026-10-24] Jev calibration ships only if it holds on held-out runs; the 2026-10-23 fit is withdrawn
**Decision:** One calibration per question (`act`, `task`); `calibrate_jev.py --question Q --fit DIR… --held-out DIR…` writes it only if the held-out decided accuracy reaches the target. The 2026-10-23 fit (T 0.441, margin 0.103) is withdrawn: it was fitted on 30 decisions from one scene, on the old act set, with no held-out set.
**Why:** Accuracy measured on the cases a threshold was fitted on is not evidence; the act set changed.
**Rejected alternatives:** Re-using the 2026-10-23 labels (options no longer match); a random case split (cases from one run are correlated: split by run / scene).
**Risks / edge cases found:** Needs new labelled interactive runs in at least two scenes before any fitted value is used; until then T = 1, margin 0.2 are declared.

## [2026-10-24] v1 test campaign: scripted, fuzz and live users, judged per line by the state it was heard in
**Decision:** `campaign.py` runs the v1 loop (XR-1 + Jev/OneJev, lockstep, ≤ 300 s wall per run) with scripted happy / faulty cases, fuzz users (fresh recorded seed per campaign) and live users (an agent appending to `--input-file`, watching `--live-view`); `harness/judge.py` scores each line against what the robot's state at that moment allowed: pass, unsafe (a wrong answer that moves the robot), safe miss, unanswered.
**Why:** Typing tests by hand covered a handful of cases; one answer can be right in one state and wrong in another ("go on" resumes a hold, continues a motion); unsafe errors matter more than over-cautious ones.
**Rejected alternatives:** Scorecard act classes only (one expected act per change, no state, no severity); random utterances without intents (nothing to judge against).
**Risks / edge cases found:** Lines are matched to owner events by text in order (a repeated line is matched to the next occurrence); tasks and seeds are ones XR-1 solved in its own evaluation, so S1 failures are under-sampled by design; revise targets are not checked for reachability in the scene.

## [2026-10-24] Jev hesitates between effects, not between acts that do the same thing
**Decision:** The act question's calibrated probabilities are summed per effect (move · hold · switch, `acts.effect`, situation-dependent: continue holds while held); abstention compares the top two effects; the act is the most probable one with the winning effect. Calibration fits and judges the act margin over the same effects (`calibrate_jev.py`; cases record effects).
**Why:** Campaign 2026-10-24: "stop" split abort 0.49 / wait 0.41 and the robot asked although both hold it; "ruko" went to revise 0.45 against hold 0.38. Replay of all 82 recorded lines: the unsafe "ruko" now asks, two stops now stop, one remark keeps holding; one vague line while already held now stops instead of waiting.
**Rejected alternatives:** A lower margin (lets weak label guesses through, e.g. "ruko"); merging labels into fewer options (loses what happens later: stop vs wait differ on resume and the wait budget).
**Risks / edge cases found:** Within a winning hold effect, stop can win over wait or ask (a vague line while held stopped the task); revise vs continue still abstains at small margins (clear new tasks still asked).

## [2026-10-24] OneJev over one kept connection; image quality unchanged
**Decision:** Every OneJev call site uses `onejev.KeptClient` (qev's client with one kept HTTP connection, reopened once when dropped); images stay JPEG q95.
**Why:** Of ~1.26 s per decision only 75–140 ms is the server; qev's client opened a connection per request, costing an extra wide-area round trip over the tunnel (~650 ms). Kept: median 604 ms, identical probabilities.
**Rejected alternatives:** A separate vision encoder (vision is ~20 ms per image of the total); smaller JPEG / WebP (changed the top option in 1 / 6 cases for ≤ 100 ms); patching the vendored qev (lost on re-clone).
**Risks / edge cases found:** `KeptClient` overrides qev's private `_post` / `_get` (pinned in uv.lock; re-check on a qev upgrade); requests on one client are serialised.
