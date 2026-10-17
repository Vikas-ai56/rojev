# Related work (verified; keys → `references.bib`)

Every arXiv ID was checked against the arXiv API. "(fig)" = value read from a figure only. Only sources in `references.bib` (29, curated 2026-10-17).

## 0. Key findings: transfer learning in VLAs (read first)
- **What transfers (established):**
  - **Skills across robot bodies that were in training.** RT-X 75.8% vs 27.3% [collaboration2023openxembodiment]; Gemini Robotics 1.5 Motion Transfer [team2025geminirobotics15]; π0.7 zero-shot cross-embodiment shirt folding 80% [intelligence2026pi07].
  - **Objects and scenes**, for task types seen in training: RT-2 unseen objects 62% vs RT-1 32% [brohan2023rt2]; π0.5 new homes [intelligence2025pi05].
- **What does not transfer (no evidence found):**
  - New motor skills, zero-shot: Octo 5% [team2024octo]; RT-2 "does not acquire any ability to perform new motions"; π0.5 "not to accomplish new skills".
  - A robot absent from all training, zero-shot: listed as future work [team2025geminirobotics].
  - Domain jumps (household → warehouse/laundry) from the same data: no study; π\*0.6 uses task-specific data [intelligence2025pistar06].
- **π0.7 "laundry without being trained on it" [pi07_blog, intelligence2026pi07] = cross-embodiment transfer, not a new skill:**
  - Laundry folding *was* in training, on the static bimanual robot (blog: "We collected data of laundry folding with the static bimanual robot").
  - The UR5e had no laundry data but was in training for other tasks (paper: "Bag In Backpack", "Organize Tupperware").
  - Truly new tasks (per authors) are short: rice scooping, spinning a fan, wiping with a cloth ("despite no data being collected for these tasks"); unseen short-horizon tasks "in the 60-80% range". Authors concede it is "difficult to definitively determine which tasks are truly 'seen'" and possible "remixing".
  - Zero-shot air fryer: "performing part of the task after a few false starts, but not finishing it fully" (blog).
- **Consequence for this paper:** generalization must come from *recombining* skills the executor already has (§2), with object/scene shift handled by the executor and new orderings handled by the supervisor. Claim scoped accordingly (README "Explicitly not claimed").

## 1. How far learned VLAs generalize: full evidence (deep research, 2026-10-17)
**Position:** no VLA learns atomic skills and then generalizes across very different domains (kitchen → laundry → warehouse) from the same training data. High confidence.

| Axis | Status | Evidence |
|---|---|---|
| New objects | ✓ | RT-2 unseen 62% vs RT-1 32% [brohan2023rt2]; Octo novel objects 80% [team2024octo]; π0.5 OOD categories [intelligence2025pi05] |
| New scenes/homes | ✓, task types seen | π0.5 scored tasks "included in the training dataset (but evaluated in entirely new scenes)"; Octo novel env 40% |
| New instructions | partial | Gemini Robotics OOD instruction 0.65 (fig) [team2025geminirobotics]; VLAs often **ignore language** [fei2025liberoplus, zhou2025liberopro] |
| Skill transfer between trained robots | ✓ | see §0 |
| Robot absent from training, zero-shot | ✗ | see §0 |
| New motor skills, zero-shot | ✗ for VLAs | see §0 |
| Household → warehouse/laundry | ✗ | see §0 |

## 2. Composing trained subtasks (our route)
- **Hierarchy beats flat prompting on held-out compositions:**
  - RoboCombine HSE > full instruction for every model [zhang2026robocombine].
  - GemBench L4: 0.3 → 17.4% [garcia2024gembench].
  - Steerable Policies: 64 (subtask-only) vs 48 (flat); full steerable 84 [chen2026steerablevla].
  - CRAFT: flat π0.5 fine-tuning 0.7–8.4% on undemonstrated combinations [yang2026craft].
- **Absolute success is low; the low level is the bottleneck:**
  - RoboCombine sim ordered success 0.8–19.7%; oracle switching ≈ predicted switching.
  - GemBench with a ground-truth plan: 31.5%.
- **Limits (ranked):**
  1. Start-state gap: RoboCombine stage conversion 7.5% → 39.9% with PRTI.
  2. Low level ignores language: LIBERO-PRO, LIBERO-Plus [zhou2025liberopro, fei2025liberoplus]; CRAFT empty-instruction.
  3. Compounding errors: CALVIN 48.9% for one subtask → 0.08% for five [mees2021calvin].
  4. Scene/camera/initial-state shift.
  5. Vocabulary coverage; no new motions (§0).
  6. Selector latency: MemER, API VLMs at 10–15 s → "complete failures" [sridhar2025memer].

## 3. Hierarchical selectors and decision interfaces
- **Skill selection with feasibility:** SayCan (usefulness × affordance; 84%/74% plan/execute in mock kitchen) [ahn2022saycan].
- **VLM over VLA:**
  - Hi Robot: high level re-runs every 1 s or on user input [shi2025hirobot].
  - π0.5: internal subtask prediction [intelligence2025pi05]; π0.5² is its open RoboCasa variant [pi052_robocasa].
  - MemER: Qwen2.5-VL-7B at ~1 Hz over π0.5, keyframe memory [sridhar2025memer].
  - **Steerable Policies**: VLA trained on subtask/motion/pixel commands; an external VLM steers it [chen2026steerablevla]. Closest prior work; LeRobot ships a generative-JSON planner for it (our Arm C).
  - Gemini Robotics 1.5: orchestrator calls the VLA as a tool [team2025geminirobotics15].
- **Typed (System One) decision models:** OneJev [onejev]; MJev-Spatial: typed spatial choice/number/point, 65.64% avg over 12 benchmarks, **no calibration reported** [qin2026mjevspatial].
- **Our gap:** image-conditioned typed supervision of a learned VLA, with calibration and risk–coverage.
- **Decision reliability:** option-order bias and debiasing [zheng2023pride]; conformal prediction sets / ask for help [ren2023knowno].

## 4. Decision timing (sync vs async)
| System | High level | Low level | Mode | Switch / staleness |
|---|---|---|---|---|
| MemER [sridhar2025memer] | ~1 Hz | ~2 Hz, 8 of 15 actions | async | latest subtask |
| RoboCombine [zhang2026robocombine] | progress trigger | chunks | latched | switch only before next chunk query |
| π0.5 + RTC [black2025realtimechunking] | below action rate | 50 Hz, H=50 | async | freeze d actions, inpaint |
| SmolVLA async [shukor2025smolvla] | — | 30 fps, n=50 | async | sync 78.3% vs async 73.3% success |

**Gap:** no paper directly compares synchronous vs asynchronous subgoal replanning.

## 5. Evaluation methodology
- Suites: LIBERO-Plus/PRO (memorization; language ignored); CALVIN ABC→D (chains); GemBench levels; RoboCasa [nasiriany2024robocasa] (our simulator).
- To report:
  - ordered success, per-transition conversion, success@k;
  - language controls (blank, gibberish, swapped);
  - training-data overlap audit;
  - ≥ 380 episodes per arm for significance (RoboCombine scale).
