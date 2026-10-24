# Related work (keys → `references.bib`)

Rewritten 2026-10-24 for the S1/S2 design: S1 = XR-1 (frozen), S2 = Jev v1 (OneJev typed choices) as the interaction
layer. Numbers come from the cited paper's own page (arXiv API abstract) or the repo research briefs
(`docs/research/2026-10-2*`). "Vendor" = authors' own claim, not replicated.

## (a) S1/S2 and hierarchical VLAs: the interface and who says "done"
| System | S2 → S1 interface | Completion signal |
|---|---|---|
| Helix [figure2025helix] | VLM at 7–9 Hz → one continuous latent vector → 80M S1 at 200 Hz; trained end-to-end; closed | synthetic "percentage task completion" action, so it "predict[s] its own termination condition" |
| π0.5 [intelligence2025pi05] | same model predicts a subtask in text, then actions | implicit (next subtask) |
| Hi Robot [shi2025hirobot] | separate VLM emits language commands to π0 [black2024pi0]; re-runs every 1 s or on user input | implicit |
| Gemini Robotics 1.5 [team2025geminirobotics15] | VLA interleaves actions with natural-language reasoning; ER 1.5 orchestrates and calls the VLA as a tool | ER 1.5 does "progress estimation" (vendor) |
| GR00T N1 [nvidia2025gr00tn1] | VLM (System 2) + diffusion transformer (System 1), "tightly coupled and jointly trained end-to-end" | none stated |
| MolmoAct2 [fang2026molmoact2] | flow-matching action expert grafted on the VLM via per-layer KV-cache conditioning; adaptive-depth "Think" variant | none stated |
| XR-1 [xiaomi2026xr1] (our S1) | single whole-task VLA; pretraining language = auto-labelled "scene state transitions"; post-trained on imperative instructions | none; whether it follows subtask text is untested |
- Steering a VLA from outside: Steerable Policies train the VLA on subtask / motion / pixel commands; subtask-only 64 vs
  flat 48, full steering 84 [chen2026steerablevla]. π0.5: GPT-4 picking from a label list scored worst, its trained high
  level best [intelligence2025pi05].
- Systematic study of hierarchical VLA agents [hu2026hivla]: a principled hierarchy beats flat control and a naive
  hierarchy; false "done" is the dominant failure, false "not done" costs little.
- VLA-OS [gao2025vlaos]: visually grounded plans beat language plans; language-only plans fall below flat.
- Take: interfaces range from latent (Helix, GR00T, trained jointly) to text (Hi Robot, π0.5, Gemini) to none (XR-1).
  We keep text: no clean text-vs-latent ablation at equal S1 exists, and latent needs joint S1 training.

## (b) Compositional generalization: what a frozen S1 can and cannot do
- **Scope.** No new motions zero-shot: RT-2 "does not acquire any ability to perform new motions" [brohan2023rt2]; Octo
  novel skill 5 % [team2024octo]. π0.7's "new tasks" are recombinations: short unseen tasks 60–80 %, long ones (air
  fryer) only with step-by-step coaching [intelligence2026pi07, pi07_blog]. Our target: new compositions of trained skills.
- **RoboCasa365 Composite-Unseen** [nasiriany2026robocasa365] (built on RoboCasa [nasiriany2024robocasa]; human
  pretrain data, multi-task): π0 0.7 %, π0.5 1.2 %, GR00T N1.5 4.4 %; N1.5 + target data 42.1 %. MimicGen data
  *lowered* it (32.3 → 22.7 at 10 % target).
- **XR-1: 32.1 % Composite-Unseen zero-shot** (official pretrain set only; Table 3), every other model ≤ 14.9 %
  [xiaomi2026xr1]; overall 57.4 % (card: 57.28 %). Vendor numbers; our parity runs covered no Composite-Unseen task.
- **Hierarchy helps, absolute success stays low:** RoboCombine hierarchy > full instruction for every model, sim
  ordered success 0.8–19.7 %, oracle switching ≈ predicted [zhang2026robocombine].
- **Skill alignment:** CRAFT finds a vision shortcut; π0.5 on held-out skill combinations 8.4 → 84.2 %, naive segment
  chaining 0.3–25.7 % [yang2026craft].
- **Seams:** Restore, Don't Retry: two-seam chain success 0.885 → 0.076 from off-support end states (displaced
  drawers, secondary objects); best-of-K, a stronger policy, a video world model and RL all fail; detect–restore–resume
  0.076 → 0.265 [wagh2026restore]. CALVIN: 48.9 % for one subtask → 0.08 % for five [mees2021calvin].
- **Language is often ignored** by the low level [zhou2025liberopro, fei2025liberoplus]: subtask steering of XR-1 must
  be tested (v2 gate G1), not assumed.
- **Wrappers around frozen VLAs gain from verification and recovery, not new skills:** CheckVLA on RoboCasa365 36.1 vs
  27.6 % for periodic replanning; timely recall 77.9 vs 48.6 % (observation-only) at 5 % false alarms; sim only
  [liu2026checkvla]. Harness VLA: frozen VLA as a retryable contact primitive beside analytic primitives, +25.4 pts on
  RoboCasa365 and +38.6 on LIBERO-Pro over the strongest baselines [zhang2026harnessvla].

## (c) Completion, progress and value signals
- π\*0.6 / RECAP [intelligence2025pistar06, pistar06_blog]: value = negative steps to completion, advantage = change in
  value; >2× throughput, ~½ failure rate on the hardest tasks.
- Robometer [liang2026robometer]: frame-level progress loss + trajectory-comparison preference loss, RBM-1M with
  failures; 4B, Apache-2.0: the v2 critic base.
- RoboReward [lee2026roboreward]: 4B/8B; negatives by counterfactual relabelling and temporal clipping; no off-the-shelf
  VLM excels across tasks.
- VLAC [zhai2025vlac]: pairwise frames + goal → progress delta and done; trained to reject irrelevant prompts and detect
  stagnation; ~30 → ~90 % success in 200 real episodes. CC-BY-NC-SA.
- SARM [chen2025sarm]: stage + progress labels from subtask annotations, not frame index (T-shirt 83 / 67 % vs BC 8 / 0 %):
  our progress-label recipe.
- SAFE [gu2025safe]: failure scalar from VLA internal features, generalises to unseen tasks, conformal timing.
- Restore's progress monitor: 0.903 AUROC in sim, 0.625 on a real Franka [wagh2026restore].
- **Ours:** zero-shot done checks are at chance on our views: MJev and OneJev balanced accuracy 0.45–0.52 on 222
  labelled frames [qin2026mjevspatial, onejev]. Hence v2 trains a critic (done / progress / feasibility).

## (d) Real-time execution of chunked policies
- RTC [black2025realtimechunking]: freeze the committed actions, inpaint the rest. Training-time RTC
  [black2025trainingrtc]: simulate delay in training and condition on the prefix; no inference overhead, better at high delay.
- Xiaomi-Robotics-0 [cai2026xr0]: post-training for asynchronous execution + timestep alignment of consecutive chunks;
  the reference recipe, since the XR-1 checkpoint has no prefix input.
- SmolVLA async inference [shukor2025smolvla]: sync 78.3 vs async 73.3 % success.
- VLA-Rolling [he2026vlarolling]: the discarded tail seeds the next chunk; cost down in 16/16 model–benchmark pairs (up
  to 1.48×), success up in 14; needs a trained confidence head. Little gain for XR-1 (5 Euler steps, all 16 actions run).
- MemER [sridhar2025memer]: API VLMs at 10–15 s per decision → "complete failures"; S2 must answer within a chunk.
- Ours: an XR-1 chunk is 16 actions = 0.8 s at 20 Hz; our WAN latency needs ~24, so evaluation runs lockstep.

## (e) Interaction and asking for help
- SayCan [ahn2022saycan]: usefulness × feasibility (84 / 74 % plan / execute); v2 scores REVISE as P_Jev × Q(s, ℓ).
- KnowNo [ren2023knowno]: conformal prediction sets decide when to ask. PriDe [zheng2023pride]: option-order bias in
  multiple-choice scoring, relevant to typed choice.
- Mid-task change is not new: Hi Robot takes interjections [shi2025hirobot]; SwitchVLA switches task mid-execution
  (forward / rollback / advance) [li2025switchvla], without calibration or ask / abort. π0.7 coaching: a high level
  fine-tuned on human step-by-step transcripts [intelligence2026pi07].
- Typed decision models: OneJev [onejev] (our S2); MJev-Spatial, no calibration reported [qin2026mjevspatial].

## (f) World models
- V-JEPA 2-AC [assran2025vjepa2]: zero-shot pick-and-place 65–80 % to goal images, ~16 s per action. RoboJEPA
  [zholus2026robojepa] loses to π0.5 on pick-and-place (27 vs 53 %). HWM [zhang2026hwm]: hierarchical latent planning
  70 % vs 0 % single-level. Not a controller at our rates; at most a feature of the done / feasibility critic.

## (g) Where we stand
- **New (engineering):** a runtime that keeps a frozen VLA moving while a typed S2 picks one act on a user change
  (continue / resume / revise / wait / ask / abort), with instruction authority (revision ids, leases, stale-goal
  suppression) and gating of irreversible switches.
- **New (decision):** hesitation by *effect* (move / hold / switch), not by act, with offline temperature + margin
  calibration that must pass a held-out accuracy target before use (fit on real OneJev answers not yet validated).
- **New (interface):** REVISE chooses among S1's trained task sentences (or `none` → ask), never free text S1 was not
  trained on.
- **New (evaluation):** scored per injected user line (right / unsafe / safe miss) on top of task success; reaction
  latency and wrong-goal commitments are the benchmark metrics.
- **Not new:** hierarchy over a VLA [shi2025hirobot, chen2026steerablevla, hu2026hivla], mid-task switching
  [li2025switchvla], calibrated asking [ren2023knowno], execution verification [liu2026checkvla, gu2025safe],
  committed-prefix chunking [black2025realtimechunking], completion heads [figure2025helix, liang2026robometer, zhai2025vlac].
- **Not yet shown:** a working completion signal in our stack; gains on Composite-Unseen; composites beyond S1's own planning.
