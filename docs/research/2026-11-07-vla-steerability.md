# VLA steerability: does S1 follow the step text?

Date: 2026-11-07. Question: S2 steers S1 by step text; do open VLAs follow text that the scene does not already imply,
and what makes them? Numbers from the papers' HTML text unless marked *(summary, unverified)*.

## Verdict
- No open VLA is known to be steerable by text out of the box; all measured ones read the task mostly from the scene.
- The fix is data, not architecture: same-scene demos with different instructions, step labels, mid-task switches.
- Our GR00T probe is the gate to keep: run it on every S1 candidate and after every fine-tune.

## Our measurement (GR00T N1.7 base, zero-shot DROID)
- Contradicting or empty text moves actions 1.5–2× seed noise; paraphrases move them less than noise.
- Details: `docs/model-bringup/groot.md`, results `results/gpu-session/20261107-183203/`.

## LangGap (arXiv 2603.00592) [hou2026langgap]
- Benchmark: 40 original LIBERO tasks + 59 new tasks in the same scenes, so the scene no longer reveals the task (20 episodes each).
- π0.5: 93.8% on the original tasks vs 21.4% on the extended ones; change target 0.0% (0/260), spatial description 11.0%,
  change object 29.3%, drawer action 31.7%.
- Others (original / extended): π0 48.3 / 8.6, π0-FAST 47.5 / 2.7, SmolVLA 38.0 / 6.4.
- Fix: same-scene multi-task data, no architecture change (LoRA on π0.5, one RTX 4090).
  - 1 extended task: 3.75 → 90.0%.
  - 5 extended tasks: 0 → 28.0%; adding the 40 original tasks drops it to 4.0%.
  - 40 original + 16 extended: extended 26.2 → 27.5%, original 93.8 → 85.5%.
- Authors' reading: the gain dilutes as tasks scale; the core problem stays open.
- Code / benchmark: announced, not released when read.

## ReSteer (arXiv 2603.17300) [chen2026resteer]
- Steerability: success after the instruction is switched mid-execution from in-distribution states (LIBERO-Goal, 10 rollouts per state × prompt).
- Scores: OpenVLA-OFT 0.252, MolmoAct (v1) 0.295, π0.5 0.403; all three score above 90% on LIBERO-Goal without a switch.
- Proxy: conditional mutual information I(action; language | state) tracks steerability (r = 0.74 across models) *(summary, unverified)*.
- Method:
  1. Find low-steerability states.
  2. Generate stage-aware motion segments between tasks (SteerGen).
  3. Fine-tune on the policy's own successful switches (SRBC).
- Gains: +10–11% steering success in simulation (abstract says 11, intro 10); 2.2× real-world vs teleop-only fine-tuning.
- Real-world per-model figures (π0.5 7%, fine-tuned 33%, ReSteer 73%) *(summary, unverified; tables do not obviously match)*.
- Code / data: promised, not found.

## Other models
- LIBERO-Plus [fei2025liberoplus]: OpenVLA-OFT's object-suite success barely changes with the instruction removed.
- MolmoAct2 [fang2026molmoact2]:
  - Open weights; SO-100/101 checkpoint; 56.7% zero-shot SO-101 pick-and-place vs π0 45.3% *(summary)*.
  - No instruction-following or steerability test published.
  - Unique instructions in training raised 22% → 46% by VLM relabelling.
- π0.7 [intelligence2026pi07]:
  - Reports large language-following gains over π0.5 / π0.6; the numbers are in figures only.
  - Recipe: subtask text + subgoal images + metadata, each randomly dropped in training.
  - Weight release not stated.

## Implications for the SO-101
- Demos: one scene, several tasks and targets ("pick red" vs "pick blue", stack vs place); a label per step; some mid-task switches.
- Gate: the probe's moved-from-original under a contradicting step must clearly exceed seed noise.
- Next GPU session: the same probe on MolmoAct2-SO100_101, then on the first fine-tune.
- Flip trigger: a candidate that passes the gate zero-shot changes the data plan from "required" to "helpful".
