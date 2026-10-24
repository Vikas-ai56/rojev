# MJev-Spatial: keep, pair with a tracker, or drop? (research brief, 2026-10-22)

Findings and a recommendation, not a decision. Sources: this repo (file:line) and primary papers / repos (URLs).
Grades: weak · strong · strongest.

## Verdict
**As used today MJev earns nothing: drop it from the v1 loop.** It changes no robot behaviour, its one job ("is this
step done?") is unmeasured where it matters, and the literature says a separate specialised judge does not beat a
general VLM out of domain. **Its only defensible future is grounding (pointing at objects) for a tracker**, and that is
worth keeping only if it beats the alternatives on our scenes in one A/B (below). The π0.5² oscillation is not an
MJev-shaped problem: it needs subtask memory and a commit signal.

## What MJev is (repo, strong)
- Third-party (Mind Lab) spatial VQA / pointing model: Qwen3.5-4B, candidates scored from the last attention layer's
  Q/K, one forward pass; answer kinds choice, scalar, single point; no boxes or masks
  (`vendor/MJev-Spatial/README.md:15-44`, `schema.py:9`, arXiv 2610.22276). Trained on 12.3M generic spatial QA pairs,
  not on robot task progress or "is this step done". No calibration reported.
- Cost here: ~9 GB GPU memory, one more service, ~100 ms server / ~458 ms round trip per check.

## What it has done in this repo (repo, strong)
- E11: half precision agrees with fp32 12/12: precision, not accuracy; answers lean "possible".
- GPU suite 2026-10-22: 49 checks, truth "not done" on all 49, P(done) ≤ 0.15. Zero positives: **whether it ever
  recognises a done step is unmeasured** (T6 open).
- It changes no robot motion in any arm: no decider reads `snapshot.step_event`; native mode's planner always
  continues (`rojev/eval/deciders.py:84-96`). Arm D, the only path where its evidence fed a choice, never ran end to end.
- The oscillation (freezer ↔ fridge 18×; plate loop ~30× in the gate run) is **π0.5²'s own planner**: neither OneJev
  nor MJev was in the baseline loop (results/gpu-suite/20261022-163931/baseline/MoveFridgeToFreezer).

## What the literature says
| Claim | Evidence | Grade |
|---|---|---|
| Zero-shot VLM step-done judging is weak; fine-tuning helps in-distribution only | SuccessVQA: zero-shot 50%, fine-tuned 83% in-domain, 59–62% unseen (https://arxiv.org/html/2303.07280) | strong |
| Specialised failure/success judges fall below their own general base models out of domain; all judges lean to "success"; contact states ≤ 0.60 | FailBench, 13 detectors, 5/5 specialists below base (https://arxiv.org/abs/2609.03611, preprint); RoboFAC real-world 80.6 vs GPT-4o 97.2 (https://arxiv.org/html/2505.12224) | strong |
| Frequent per-frame VLM queries raise false judgements; tracked geometry monitors better | Code-as-Monitor vs DoReMi (https://arxiv.org/html/2412.04455) | strong |
| Spatial training clearly helps pointing, but general 4B models have closed much of the gap | Gemini Robotics-ER table (https://arxiv.org/html/2503.20020); Qwen3-VL-4B Where2Place 64.0 vs spatial fine-tune +4.3 avg with regressions (https://arxiv.org/html/2608.30935) | moderate |
| Subtask oscillation / repeated steps come from missing memory of progress; memory + a progress / commit signal fixes it | MemER on π0.5: 59 vs 32 of 60 (https://arxiv.org/html/2510.20328); τ0-VLA execution memory (https://arxiv.org/html/2608.16885); Critic in the Loop on π0.5 0/10 → 4/10 (https://arxiv.org/html/2603.05185) | strong (MemER) / weak–moderate |
| A VLA's own high level beats GPT-style planners | π0.5 (https://arxiv.org/html/2504.16054), Hi Robot (https://arxiv.org/html/2502.19417v2) | strong |

## Options
| Option | Wins | Loses |
|---|---|---|
| (a) MJev alone as step verifier (today) | never gave a false "done" (49/49) | no measured recall; changes nothing; 9 GB; judges of its kind lean to "success" and lose out of domain |
| (b) MJev grounding + SAM 2.1 Tiny tracker | pointing is where spatial training pays; tracked predicates (moved / in region / held) are the structural completion and precondition signal | tracker occlusion is the main error source; needs building; MJev's pointing on our scenes unmeasured; SAM 3.1 text grounding is a one-model alternative |
| (c) Drop; ask OneJev | one model less, ~9 GB freed | OneJev cannot point; its step-done accuracy is unmeasured too |
| (d) π0.5² alone | simplest | the configuration that oscillates, here and in the literature |

## Recommendation (low regret)
1. **Now:** stop loading MJev in the v1 loop (frees ~9 GB, removes a service); keep the code and client (reversible).
2. **Oscillation:** fix in the owner / S2 with subtask memory and a commit rule (a left step is not re-entered without
   evidence it was undone), driven by a progress signal: the tracked-object predicates of (b) when built, until then
   none (no heuristic stand-in).
3. **One A/B decides MJev's future**, on RoboCasa frames with simulator truth (positives made by setting object
   states, so no dependence on π0.5² succeeding):
   - step-done: balanced accuracy and false-"done" rate per step type, MJev vs OneJev, same typed protocol;
   - grounding: correct-object pointing rate (duplicates, relational references), MJev vs SAM 3.1 text grounding vs
     Qwen3-VL-4B.

## Flip triggers (measurable)
| Result | Decision |
|---|---|
| MJev step-done balanced accuracy ≥ OneJev + 10 pp and false "done" ≤ 5% | keep MJev as the verifier |
| MJev correct-object pointing ≥ best alternative + 10 pp on duplicates / relational prompts | keep MJev as the grounder in front of SAM 2.1 Tiny |
| Neither | drop MJev; ground with SAM 3.1 (or Qwen3-VL-4B points) + SAM 2.1 Tiny; verify with tracked predicates |

## Confidence
- High: today MJev changes no robot behaviour; the oscillation is π0.5²'s planner.
- Medium: a general VLM matches MJev on step-done (literature, not our data).
- Low–medium: MJev's grounding value on our scenes (no in-domain data).

## Addendum (same day): why π0.5²'s planner oscillates, and what that does to the verdict
- **π0.5²'s subtask head is memoryless by training and by inference.** Training recipe
  `vendor/lerobot-steerable/src/lerobot/configs/recipes/subtask_mem.yaml:8-12`: subtask target conditioned on the task
  text (+ images) only; the memory head is trained separately (`:18-29`) and never conditions the subtask. Inference
  prompt for a subtask = the task text only (`policies/pi052/inference/pi052_adapter.py:128-130`), regenerated every
  chunk (`chunks_per_regen=1`, `serve.py:78`). With an ambiguous image it flips (freezer ↔ fridge): the τ0-VLA / MemER
  failure mode. Feeding history into its subtask prompt would be out of distribution (not trained), so no
  training-free prompt fix is expected to hold.
- **Training-free fix: commit and verify.** π0.5² is steerable by subtask text, so the owner can pin the subtask its
  planner chose (a SWITCH_TO with that text) until a verifier says it is done or it stalls, then hand planning back.
  Its flips cannot reach the robot while a step is committed. Training fix (v2): condition its subtask head on
  completed subtasks / memory (MemER, τ0-VLA).
- **Effect on the verdict:** commit-and-verify needs a step-done signal, so a verifier would finally change behaviour.
  The verdict becomes: **keep MJev until one A/B picks the release signal** (MJev vs OneJev step-done on truth frames,
  then native vs native + commit on oscillation count and task success). Drop whichever loses.

## Scope of MJev (now → v2), checked against the Jev × Helix plan
The plan gives the capabilities, not the model: "Holding and completion — Jev auxiliary heads or validated monitor";
"Persistent object identity — tracker or explicit evidence association"; "Completion: evidence over a window with
explicit success conditions and separate uncertainty"; "Never promote a provisional prediction to 'completed' because
another model echoed it. Confirm through independently specified evidence"; "Do not build a geometry stack until the
decisions require one" (plan pp. 8, 10). Long term, S2 is one shared backbone with typed-choice **and**
spatial / progress readouts (`p_aux = spatial_and_progress_readouts(H)`, S2-B grounded supervision, `L_spatial`,
`L_progress`; plan pp. 6, 13), and the roadmap keeps a merged OneJev + MJev backbone "only if sequential MJev→OneJev
limits accuracy / latency" (`docs/plan/v3-roadmap.md:151`). So MJev is the frozen stand-in for Jev's future spatial and
progress readouts, not a permanent separate service.

| Horizon | MJev's job | In / out | Gate |
|---|---|---|---|
| v1 now | step-done check = the release signal of commit-and-verify (π0.5²'s committed subtask), windowed (2 confirmations), explicit per-step question | in, pending the A/B | MJev vs OneJev on truth frames (balanced accuracy, false "done" ≤ 5%); loser leaves the loop |
| v1 later | grounding: a point for the target object → SAM 2.1 Tiny track → held / in-region / visible predicates (preconditions, object-naming questions, contact-aware monitor) | out until the tracker is built | correct-object pointing vs SAM 3.1 text grounding / Qwen3-VL-4B (+10 pp to keep) |
| v1 never | per-frame verdicts that move the robot on their own; metric distances (its scalar is not calibrated, plan p. 8) | out | — |
| v2 | its training data and procedure inform Jev's spatial / progress heads (S2-B); a merged backbone replaces the separate service if latency or accuracy of the two-model path is the bottleneck | design input | T9 + measured two-model latency |

## Result of the A/B (GPU, Mumbai, 2026-10-22; `results/done-check/20261022`, `results/gpu-suite/20261022-mumbai`)
222 simulator-labelled (frame, step) pairs, 108 done; 2 tasks × 10 seeds; steps place / close / press.

| Checker | Question | Balanced acc. | Recall | False "done" | ms / check |
|---|---|---|---|---|---|
| MJev | "is this step done?" (done/possible/blocked) | 0.50 | 0.00 | 0.00 | 57 |
| OneJev | same | 0.52 | 0.06 | 0.01 | 350 |
| MJev | postcondition yes / no ("Is the steak inside the microwave?") | 0.45 | 0.22 | 0.32 | 55 |
| OneJev | same | 0.52 | 0.42 | 0.38 | 337 |

- Both are at chance on both question forms. "place" is inverted for both (P(yes) lower on done frames: MJev 0.17 vs
  0.36), consistent with not telling containers apart on 256 × 256 views; "close the door" carries a weak signal
  (OneJev 0.63 vs 0.47). "Microwave on" has no visible cue.
- Commit and verify, live (60 s runs): subtask switches 19 → 2 and 6 → 1–2, so pinning stops the oscillation; 0 steps
  verified, every release by the 25 s bound; and π0.5²'s first pick on MoveFridgeToFreezer ("put steak in **fridge**")
  was wrong and got pinned: commitment needs the planned subtask checked against the task.

**Verdict (gate applied):** MJev fails the keep gate on both question forms → out of the v1 loop. No VLM checker in
the stack detects completion on these views, so commit-and-verify releases only by its bound until a real signal
exists: tracked object state (SAM 2.1 Tiny + identity, plan "tracker or explicit evidence association") or a learned
progress head (v2, S2-C). One cheap check remains before concluding the views are the limit: the same A/B on
higher-resolution renders with the single relevant camera.

## π0.5² native planner memory (`enable_memory=True`), 2026-10-22 (`results/memory-ab/20261022`)
| Run | Subtask switches / 60 s | Success | S1 p50 |
|---|---|---|---|
| memory off, seed 1000 / 1001 | 16 / 10 | no / no | 532 / 518 ms |
| memory on, seed 1000 / 1001 | 21 / 11 | no / no | 592 / 557 ms |

- With memory on it still flips "put steak in freezer" ↔ "put steak in fridge" every chunk or two.
- Its notes claim progress that did not happen ("I picked up the steak from the fridge and placed it in the freezer").
  The memory text is generated, not observed, so it cannot serve as a completion signal either.
- Confirms the addendum: the subtask head is not trained on memory, so memory does not change its choice.
