# Experiments log (exact numbers; newest last)

Model under test unless stated: OneJev [onejev] 1.5 0.8B Q8_0 GGUF (`mradermacher/OneJev1.5-0.8B-GGUF` @57b376b8e75f) served by qev + llama.cpp 0.6.0 on a MacBook Air M5 16 GB. Single deterministic sample per query (no sampling). "n" = number of frames or episodes.

## E1 — OneJev uses the image (2026-10-16)
- Setup: Menagerie SO-101 `scene_box.xml`, free camera, 640×480; same request with the green box shown vs hidden (rgba α = 0).
- P(green box visible): **0.984 shown vs 0.140 hidden**. P(robot arm visible): 0.983 vs 0.986. P(approach_box): 0.934 vs 0.658.
- Identical repeat → identical probabilities. Latency 195–390 ms per request. Steady-state memory ≈ 1.3 GB (llama-server 1079 MB + qev 245 MB).
- Source: `docs/features/phase0-install.md`.

## E2 — Instruction text biases OneJev's answers (2026-10-16)
- Setup: SO-101 scripted trajectory, box hidden, 3 checkpoints, two instructions. n = 3 frames per instruction.

| Checkpoint | "Approach the visible green box": P(box) / P(approach) | "Decide the next action from what the image shows": P(box) / P(approach) |
|---|---|---|
| home | 0.138 / 0.569 | 0.003 / 0.309 |
| pickup | 0.064 / 0.228 | 0.001 / 0.058 |
| home_again | 0.118 / 0.502 | 0.002 / 0.289 |

- Box shown: P(box) 0.980–0.983, P(approach) 0.815–0.868.
- Finding: the instruction raises both the action answer and the *visibility* answer. Evidence questions must not share the instruction's state.

## E3 — Prompt / protocol variants for the false-premise case (2026-10-16)
- Setup: `scene_box.xml`, free camera, 640×480; 4 frames (home/pickup pose × box shown/hidden). Correct = `approach_box` iff the box is visible (segmentation ground truth). Phoenix project `critique`, session `critique-183951`. Script: `playground/archive/e3_prompt_variants.py`.

| Variant | Correct /4 | Hidden-box: P(approach) home / pickup | Shown: P(approach) home / pickup |
|---|---|---|---|
| V0 baseline ("Approach the visible green box") | 3 | **0.466** ✗ / 0.227 | 0.876 / 0.775 |
| V1 V0 + debias=3 (option-order shifts) | 3 | **0.490** ✗ / 0.254 | 0.887 / 0.786 |
| V2 neutral instruction | 4 | 0.252 / 0.057 | 0.610 / 0.440 |
| V3 goal instruction, no "visible" wording | 2 | **0.712** ✗ / **0.587** ✗ | 0.896 / 0.829 |
| V4 V3 + `report_missing` option | 3 | **0.545** ✗ / 0.342 | 0.896 / 0.817 |
| V5 V4 + preconditions in option text | 3 | **0.405** ✗ / 0.274 | 0.701 / 0.622 |
| V5 + debias=4 | 3 | **0.373** ✗ / 0.254 | 0.747 / 0.686 |
| **V6 two-step: visibility on neutral state → fed into decision** | **4** | 0.316 / 0.293 (picks report_missing 0.496 / 0.556) | 0.805 / 0.733 |

- V6 visibility (neutral state): 0.992 shown vs 0.009 / 0.005 hidden. With the instruction in the state it was 0.06–0.14 (E2).
- Option-order debiasing barely changes results (0.466 → 0.490) → order bias is not the cause.
- Small n (4 frames, one scene); evidence, not a benchmark.

## E4 — Viewpoint changes the answer (2026-10-17)
- Same question, SO-101 `scenes/pick_place.xml`, fixed **overhead** sensor camera, box hidden: P(box visible) 0.030 / 0.028 (home / pickup); `next` = observe 0.549 / 0.532, approach 0.224 / 0.257 → no false approach.
- The E2/E3 false-approach failure was specific to the free-camera view. Source: `docs/features/robot-interface.md`.

## E5 — OneJev readout mass (2026-10-16)
- `allowed_token_mass` (probability on the answer letters before renormalising): 3–8% on image questions; 11–32% on text-only questions with obvious answers.
- Explained by training: loss is "cross-entropy against a soft target plus a Brier term, both over the K answer slots" (`OneJev train/sft.py:368`). Expected by design, not a fault; probabilities are relative among options.

## E6 — Communication cost (2026-10-17)
- Loopback round trip vs decision (OneJev, image + 1 Choice): transport median 0.41 ms (p95 0.75) vs decision median 73 ms (p95 458) → ~0.6%.
- IPC for one observation (3 × 256×256 RGB = 590 KB), 300 trials: HTTP + JSON + base64 **2.96 ms**; Unix socket raw **0.44 ms**; shared memory + 1-byte notify **0.07 ms**. Script: `playground/archive/e6_ipc_bench.py`.
- OneJev median latency 145 ms headless vs 311 ms with the MuJoCo viewer open (GPU contention suspected, untested).

## E7 — SmolVLA executor (`lerobot/smolvla_robocasa`) in RoboCasa [shukor2025smolvla, smolvla_robocasa, nasiriany2024robocasa] (2026-10-17)
- **Three mismatches found in LeRobot's documented eval path**, all of which invalidated early runs:
  1. Camera mapping: wrist and right camera swapped vs training.
  2. Scene split `all` vs training `target`.
  3. Instruction: natural-language sentence vs the task names it was trained on.
- **Invalid runs** (kept for audit, not results): `g0a_smolvla_closefridge_20261017-140555`; `g0c_20261017-151400` (parallel envs crashed on macOS); `g0c_20261017-151517` and `g0c_20261017-152512` (cameras and/or split wrong); `g0c_20261017-153102` (sentence instruction). "ERROR" rows for OpenCabinet in those runs = runs I interrupted.
- **A/B instruction test** (cameras + split fixed, seeds 1000–1004, same seeds): task name **1/5** (seed 1002, 245 steps) vs sentence **0/5**. Runs: `rollout_CloseFridge_taskname_20261017-154211`, `rollout_CloseFridge_lang_20261017-154617`.
- **Corrected setup, 5 episodes per task** (seeds 1000–1004):

| Task | Success | Successful seed (steps) | Objects | Run |
|---|---|---|---|---|
| CloseFridge | 1/5 | 1002 (245) | lightwheel | `rollout_CloseFridge_taskname_20261017-154211` |
| OpenDrawer | 1/5 | 1004 (314) | lightwheel | `rollout_OpenDrawer_taskname_20261017-155446` |
| TurnOnSinkFaucet | 1/5 | 1004 (423) | lightwheel | `rollout_TurnOnSinkFaucet_taskname_20261017-155842` |
| TurnOnMicrowave | 1/5 | 1001 (420) | objaverse + lightwheel | `rollout_TurnOnMicrowave_taskname_20261017-160410` |

- TurnOnMicrowave first crashed at reset with lightwheel objects only (no valid object category): `rollout_TurnOnMicrowave_taskname_20261017-155817`.
- **Total 4/20 (20%)**; each 1/5 has 95% CI ≈ 4–62%. Too unreliable as the executor → switched to π0.5².
- Speed on M5 (MPS): ~27 steps/s; episode ≈ 58 s; peak RAM 3.39 GB.

## E8 — π0.5² subtask vocabulary [pi052_robocasa] (2026-10-17)
- Source: `pepijn223/robocasa_pretrain_human300_v4_annotated5` (32,043 episodes; 4,740 task strings), `language_persistent` rows of style `subtask`. Output: `data/vocab/pi052_robocasa_subtasks.json`.
- **21,326 distinct subtasks.** Episodes per subtask: ≥100: 212 · ≥20: 1,133 · 2–19: 7,415 · 1: 12,778.
- Top: open the refrigerator door 2,980 · open the cabinet door 2,726 · pick up the bowl 2,359 · close the refrigerator door 1,430.
- Paraphrases coexist: "open the fridge door" 596; 1,237 fridge-related variants.

## E9 — Candidate-builder coverage@K, exact strings (2026-10-17)
- Data: 32,043 ordered timelines (`data/vocab/pi052_robocasa_sequences.jsonl`; mean 5.48 subtasks/episode, median 5, max 43). Builder: `rojev/supervisor/candidates.py`. Script: `playground/candidates/coverage.py`. Run: `results/runs/e9_coverage_20261017-174343/`.
- Hit = true next subtask (normalised: lowercase, no punctuation/articles, fridge→refrigerator) is in the top-K shortlist.

| Split | Ceiling (target string exists in fitted vocab) | Variant | @1 | @5 | @10 | @20 | @50 |
|---|---|---|---|---|---|---|---|
| episodes held out (3,211 eps) | 92.2% | A transitions | .195 | .385 | .461 | .525 | .588 |
| | | B similar tasks | .163 | .462 | .584 | .682 | .767 |
| | | A+B | .277 | .525 | .633 | .724 | .799 |
| | | A+B+C | .284 | .539 | .630 | .699 | .766 |
| **tasks held out** (3,460 eps) | **74.8%** | A transitions | .078 | .188 | .237 | .286 | .335 |
| | | B similar tasks | .085 | .253 | .336 | .409 | .492 |
| | | A+B | .131 | .293 | .372 | .448 | .541 |
| | | A+B+C | .142 | .315 | .398 | .471 | .545 |

- **Finding:** for new tasks, 25.2% of needed subtasks never occur as exact strings in training. The most frequent: "place wine bottle next to wine glass", "pick up tea strainer", "place yogurt container on plate", "close electric kettle lid". These are new verb–object combinations of familiar templates. Exact-string retrieval cannot reach the 95% target → next: template + slot-filling builder (E9b).
- Runtime: both splits, 4 variants, ~33k steps in 47 s on the M5 (pure Python).

## E9b — Template + slot-filling candidate builder (2026-10-17; partial, not adopted)
- Builder: subtask → verb template + object phrases ("place {} next to {}", ("wine bottle", "wine glass")); verbs learned from subtask/instruction starts; template transitions + similar-task templates; fills from instruction, history, scene evidence; candidates weighted by learned P(phrase source | template, slot). Code: `playground/candidates/templates.py`, `coverage_templates.py`. Log: `results/logs/e9b.log`.
- Stopped after the held-out-tasks split, no-evidence condition (user: filter is an add-on; pipeline first). Episodes split and oracle-evidence rankings not measured.

| Held-out tasks, no scene evidence (21,171 steps) | @1 | @5 | @10 | @20 | @50 | unseen-string @10 |
|---|---|---|---|---|---|---|
| E9 exact A+B+C (reproduced exactly) | .142 | .315 | .398 | .471 | .545 | 0 |
| exact × grounding | .111 | .272 | .339 | .413 | .510 | 0 |
| templates × grounding | .041 | .193 | .270 | .357 | .448 | .146 |
| exact + templates, grounded | .159 | .317 | .405 | .497 | .613 | .108 |

- Ceilings (target reachable at all): exact strings 74.8%; templates 63.0%; either **89.4%**. With oracle scene objects + 2 distractor episodes: templates 90.1%, either **96.7%**.
- Finding: templates raise what is *reachable* (75% → 89–97%), but ranking stays weak (+0.7 pt @10, +6.8 pt @50). Grounding alone hurts (.398 → .339).
- Conclusion: keep the E9 builder; the shortlist only needs recall, OneJev handles wording/meaning. Revisit after the pipeline works.

## E10 — Live workflow bring-up, arm E, MicrowaveThawingFridge (2026-10-18)
- Setup: sim on the Mac (M5, 26.9 steps/s), π0.5² on Kaggle T4 (bf16) via Cloudflare tunnel; seed 1000 (same scene); max 2320 steps; arm E (authored sequence, ground-truth switching). Runs: `results/runs/workflow/MicrowaveThawingFridge_E_s1000_20261018-{134816,135317,135758}/`.
- Per chunk (median): π0.5² inference 2.40 s, network 0.21 s, request 121 KB; ≈ 4.5 min per 2320-step episode.

| Ep | Oracle version | First lift | Outcome | Stage at fault |
|---|---|---|---|---|
| 1 | grasp-only "pick up" check, forward-only | 299 (false: steak still on the plate) | step limit, 0/4 subtasks | ground-truth check (premature switch) |
| 2 | grasp + off plate/fridge, forward-only | 424; held at steps ~600–750, dropped in transit | step limit, 0/4 | executor (drop) + oracle did not go back |
| 3 | grasp + off plate/fridge, goes back on undone progress | 794 (brief); held again ~2100, dropped | step limit, 0/4 | executor (grasp reliability) |

- Same seed, different trajectories: π0.5²'s flow sampling is stochastic.
- Findings: (1) "grasped" ≠ "picked up" (contact fires while the object rests on its plate); (2) a forward-only oracle cannot recover drops; (3) with correct subtasks and switching, π0.5² failed to pick up and hold the steak reliably (~2000 of 2320 steps on "pick up").
- Invalid as arm comparison (n = 3, oracle changed between episodes); bring-up only.

## E11 — MJev-Spatial bf16 trunk vs fp32, arm-D template, T4 (2026-10-18)
- Setup: 4 RoboCasa pretrain scenes sampled with seed 0 (PrepareCocktailStation, ReheatMeal, BeverageOrganization, OpenElectricKettleLid), 3 cameras each, top-3 filter subtasks per instruction; question "Robot step: '<s>'. Which is true right now?" → done/possible/blocked. Reference fp32 on CPU; test bf16 trunk + fp32 readout on T4. Files: `results/kaggle/armd_20261018-interactive/`.
- **Top-1 agreement 12/12; max |Δp| 0.029.** GPU memory 9.0 GB.
- Latency per request (3 questions, 3 images, 1024 image tokens each): **≈ 24.5 s bf16 on T4**, ≈ 65 s fp32 on CPU.
- Zero-shot answers lean "possible" (0.65–0.94 in 11 of 12 judgements; ground truth not available for these frames).
- Consequence: precision OK; latency too high for per-chunk evidence (~8 questions) → speed levers next (image tokens, fp16, fewer images).

## E12 — MJev speed levers on T4 (2026-10-18)
- Same 4 scenes and fp32 reference as E11; `services/mjev/speed_check.py`; file `results/kaggle/armd_20261018-interactive/mjev_speed.json`.

| Variant | ms / request (3 questions) | top-1 vs fp32 | max \|Δp\| |
|---|---|---|---|
| bf16, 1024 image tokens, 3 cams | 24,167 | 12/12 | 0.027 |
| bf16, 512, 3 cams | 24,135 | 12/12 | 0.027 |
| fp16, 1024, 3 cams | 22,713 | 12/12 | 0.027 |
| fp16, 512, 3 cams | 21,713 | 12/12 | 0.027 |
| fp16, 512, 2 cams | 14,473 | 12/12 | 0.219 |

- Image-token cap irrelevant (256² frames are below both caps); fp16 ≈ −10%; 2 cams change answers (Δp 0.22) → keep 3.
- Hypothesis: transformers' PyTorch fallback for Gated DeltaNet (no `flash-linear-attention`, FLA).
- **E12b, FLA installed** (`mjev_speed_fla.json`): bf16 24,397 ms (12/12, Δp 0.023), fp16 21,821 ms (12/12, Δp 0.026) → **no speed-up; hypothesis rejected.** `causal_conv1d` still on its fallback. Next: stage profile of one request (`services/mjev/profile_mjev.py`).

## E12d — Where MJev's time goes (2026-10-18)
- One request (3 images, 3 questions), bf16, T4: total 23.8 s = vision + embeddings 20.7 s (87%) + language trunk 3.0 s + CPU preprocessing 0.02 s + readout 0.002 s.
- Inside vision: **patch embedding (one Conv3d, kernel = stride = 2×16×16) 20.6 s of 21.2 s**; 24 transformer blocks ≈ 0.8 s. Input 2304 × 1536 (3 images, 16×16 patch grid each), bf16, cuda, sdpa.
- Cause: a slow Conv3d path on T4, not the model size.
- **E12e fix candidates** (same 2304 × 1536 input; `mjev_patch_embed.json`): bf16 Conv3d 21,057 ms · fp32 Conv3d 15.9 ms · **bf16 linear 4.8 ms** · fp32 linear 2.8 ms; max |diff| vs current 0.0078–0.0156 (outputs mean 0.31; = bf16 rounding). Kernel = stride → the conv is exactly a matrix multiply over patches. Adopted: `PatchLinear` in `serve_mjev.py` (checked against the conv at load).
- **E12f end to end with PatchLinear** (`mjev_speed_patched.json`): bf16 **3,484 ms** (12/12, Δp 0.024) · fp16 **1,280 ms** (12/12, Δp 0.031, no NaN) per 3-question request — 7× and 19× faster than E12. Serving default → fp16.

## E13 — OneJev next-subtask decisions on π0.5² demo frames (2026-10-18)
- Benchmark: 40 frames from 15 held-out tasks (E9 split), 3 cameras each, true next subtask from the demo annotations; options always contain the truth; correct = same meaning (verb phrase + first object). `playground/onejev_bench/`; runs `results/runs/e13_onejev_bench_*`.
- Baselines: chance 4.8% (21 options) / 11.1% (9); filter top-1 (no images) 25%; truth in raw top-20 shortlist 67.5%.

| Variant | 0.8B | 4B |
|---|---|---|
| V0 arm-D prompt (composite image, task, 20 raw options) | 22.5% | 40.0% |
| T1 V0 + blank image | 17.5% | 42.5% |
| T2 V0 without task text | 12.5% | — |
| T3 ≤ 8 distinct meanings | 30.0% | 40.0% |
| T4 T3 + 3 separate images | 20.0% | — |

- Findings: 4B ≈ 2× 0.8B on the current prompt; **blank image does not lower accuracy for either model** → decisions come from task + history text, not the cameras; distinct options help 0.8B; separate images do not help; instruction copy rate low (0–17.5%).
- Caveat: n = 40 (≈ ±15 points); widen before paper claims. Labels are π0.5²'s noisy annotations.
- Consequence: prompt fixes alone cannot make OneJev read the scene → train OneJev on image-dependent decisions (same benchmark = data + yardstick).

## Planned (not yet run)
- π0.5² language following: no standalone diagnostic; per-subtask success vs the chosen subtask text is logged in the supervisor runs (arms A–E).
- Progress-value question for OneJev (RECAP-style): Score "how far from finishing <subtask>?" trained on π0.5² demo timestamps + our rollouts; compared with MJev zero-shot done/possible/blocked on held-out tasks.
