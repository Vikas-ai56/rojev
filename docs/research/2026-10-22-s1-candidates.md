# S1 candidates (frozen VLA behind the remote S1 interface)

Date: 2026-10-22. Question: which open VLA should S1 be, per target (RoboCasa v1, LIBERO, real SO-101, real Franka)?
Every number below was re-fetched today from the cited primary page, unless it is marked derived or not verified.

## Verdict

| Target | Pick | Confidence | Why |
|---|---|---|---|
| RoboCasa v1 | **Xiaomi-Robotics-1-RoboCasa365** (XR-1), replacing `lerobot/pi052_robocasa` | Medium | Best published open RoboCasa365 result: 57.28% on target50 (derived per split: 81.8 / 58.0 / 29.0), against π0.5's 39.6 / 7.1 / 1.2. No model has a published number on our two tasks. |
| LIBERO | **MolmoAct2-LIBERO-LeRobot**, or keep π0.5 (`pi05_libero`) | Medium | The benchmark is saturated: every candidate scores 96.9–98.7. Choose on integration cost. MolmoAct2's checkpoint is native to LeRobot and Apache-2.0. |
| Real SO-101 | **MolmoAct2-SO100_101** (self-evaluation required) | Low | It is the only candidate checkpoint built for SO-100/101. No success rates are published. |
| Real Franka | **MolmoAct2** (base or -DROID) | Medium-low | 87.1% vs π0.5's 45.2% zero-shot. This is Ai2's own comparison and nobody has replicated it. |

- In our measurements the failure sits in S1's motor skill, not in planning or timing (see the "Our measurements" section). Changing S1 is the right lever. Tuning S2 or the timing will not help.
- Before trusting a swap, move the v1 evaluation onto target50 tasks. MicrowaveThawingFridge and MoveFridgeToFreezer belong to pretrain200/300, not to target50 (vendored `robocasa/utils/dataset_registry.py`, commit a9e152e).

## Comparison

Evidence grades: **A** = primary page, numbers re-checked today. **B** = vendor's own claim with nothing independent. **C** = derived by us from primary data. **D** = not verified.

| Model | Open? | Licence | Design | Reported result (protocol) | Grade | Source |
|---|---|---|---|---|---|---|
| π0 / π0.5 / GR00T N1.5 (RoboCasa365 paper) | Checkpoints released | — | Baseline VLAs | Table 1, human pretrain data, multi-task (atomic / comp-seen / comp-unseen / avg): π0 36.3/5.2/0.7/15.0; π0.5 39.6/7.1/1.2/16.9; GR00T N1.5 43.0/9.6/4.4/20.0. 30 trials per task. | A | [paper](https://arxiv.org/html/2603.04356v1) |
| GR00T N1.5 + target data | Yes | — | VLM + DiT | Table 2, pretrain + target 100%: 68.5/40.6/42.1/51.1 | A (HTML table flattened; columns matched to Table 4) | [paper](https://arxiv.org/html/2603.04356v1) |
| `lerobot/pi052_robocasa` (S1 until 2026-10-22) | Yes | Apache-2.0 | π0.5 port. 3 cameras at 256², 16-D state, 12-D action | Trained on `pepijn223/robocasa_pretrain_human300_v4_annotated5` (32,043 episodes). No evaluation numbers on the card. | A (card) / D (performance) | [HF](https://huggingface.co/lerobot/pi052_robocasa) |
| **Xiaomi-Robotics-1-RoboCasa365** | Yes, 5B BF16 | Apache-2.0 | Qwen3-VL + DiT (mixture-of-transformers). Observation history 4 at interval 2. 16 actions per query. | Split `pretrain`, target50, 50 episodes per task, seed 7: 1432/2500 = 57.28%. Derived by us per split from that `summary.json`: 81.8 / 58.0 / 29.0; the paper's Table 3 (leaderboard run) reports 80.2 / 57.1 / 32.1; the two runs differ, quote the paper's for comparisons. | A (total) / C (per split) | [HF](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365), [summary.json](https://github.com/XiaomiRobotics/Xiaomi-Robotics-1/blob/main/eval_robocasa365/summary.json) |
| Xiaomi-Robotics-0 | Yes, 4.7B | Apache-2.0 (repo) | Training-time real-time chunking: committed-action prefix plus a Λ-shaped attention mask | LIBERO 98.8/100/98.8/97.2, avg 98.7 (its table lists π0.5 at 96.9). 80 ms inference on an RTX 4090; this is inference only, not end to end. | A / B | [paper](https://arxiv.org/html/2602.12684v1), [repo](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0) |
| π0.5 `pi05_libero` | Yes | openpi | Flow-matching VLA | LIBERO 98.8/98.2/98.0/92.4, avg 96.85 | A | [openpi](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) |
| **MolmoAct2** | Yes: weights, data, code | Apache-2.0 (repo §7; LIBERO-LeRobot card). The base and SO100 cards have no licence field. | Molmo2-ER VLM plus a flow-matching action expert. Adaptive-depth "Think" variant. | Real Franka zero-shot, 15 trials per task: 87.1% vs π0.5 45.2%. LIBERO 97.2 (Think 98.1). LeRobot port: 98.25. About 180 ms base / 790 ms adaptive per call on one H100. | B (Franka) / A (LIBERO) | [blog](https://allenai.org/blog/molmoact2), [repo](https://github.com/allenai/molmoact2), [LIBERO-LeRobot](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot) |
| MolmoAct2-SO100_101 | Yes | Not on card (repo: Apache-2.0) | 2 RGB cameras plus raw 6-value state | No success rates. BF16 "under 16GB in our tests". | A (spec) / D (performance) | [HF](https://huggingface.co/allenai/MolmoAct2-SO100_101) |
| Molmo2-ER (backbone) | Yes, standalone checkpoint | Apache-2.0 | 4B VLM (the card's metadata says 5B). Qwen3-4B + SigLIP2. | Average 63.8 over 13 embodied-reasoning benchmarks, including Point-Bench. No per-benchmark pointing score on the card. | A (average) / D (pointing) | [HF](https://huggingface.co/allenai/Molmo2-ER) |
| GR00T N1.7 | Yes, 3B | Code Apache-2.0. Weights: the card says NVIDIA Open Model License, while the repo README says "Apache 2.0" (conflict). | Cosmos-Reason2-2B VLM + 16-layer flow DiT, horizon 40 | LIBERO 97.65/97.5/98.45/94.35 (≈97.0 average, our mean). Original 24-task RoboCasa: 70.8% (N1.6: 66.22%), episode count not stated. **No RoboCasa365 number.** E2E 85.8 ms eager / 27.9 ms TensorRT on H100. | A (with caveats, see the not-verified list) | [repo](https://github.com/NVIDIA/Isaac-GR00T), [LIBERO](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md), [RoboCasa](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa/README.md), [card](https://huggingface.co/nvidia/GR00T-N1.7-3B) |
| Helix (Figure) | No | — | S2: 7B VLM at 7–9 Hz. S1: 80M policy at 200 Hz. | ~500 h teleoperation data. No release stated. | A (design) | [Figure](https://www.figure.ai/news/helix) |
| Gemini Robotics 2 / ER 2 | No open weights listed | — | VLA plus ER model | ER 2 is a preview in AI Studio. The VLA is waitlist only ("100+ trusted testers"). | A | [DeepMind](https://deepmind.google/models/gemini-robotics/) |

## Per-candidate notes

**XR-1-RoboCasa365 (the RoboCasa v1 pick)**
- Use: the S1 service for RoboCasa v1, behind the existing remote S1 protocol.
- Its strongest split is composite-unseen: 29.0 derived, against 1.2 for π0.5 and 42.1 for GR00T N1.5 trained with target data. Composite-seen: 58.0 derived, against 40.6 for GR00T N1.5 with target data.
- The reference evaluator has the same shape as our runtime. `scripts/deploy.sh` starts `deploy/server.py`, which runs `pickle.loads` on socket bytes (line 57). That is remote code execution if the server is reachable. Its default host is `localhost`. Do not expose that server. Wrap the model in our own transport. The runtime server in `xr1/mibot/server/runtime/server.py` uses `np.load(allow_pickle=False)`, which is the safer pattern (repo commit 0dd7aef).
- The history input (4 frames at interval 2) and the 16-action chunk belong in the model's executor-side client adapter, not in `rojev/` core. This follows the "core never bends to sim" rule.
- Latency must be profiled on our GPU before any run in real time. The model is 5B, the vendor publishes no latency, and its evaluator used 8 servers on 8 GPUs. At 20 Hz control, a 16-action chunk lasts 0.8 s.
- Not known: whether it follows subtask text (S1 steering) or only whole-task RoboCasa365 instructions.

**MolmoAct2 (Franka, SO-101, LIBERO)**
- Use: the real Franka (base or -DROID), SO-101 (-SO100_101), and LIBERO (-LIBERO-LeRobot) pick.
- Constraint: the Franka margin is Ai2's own evaluation (5 tasks, 15 trials each). Replicate it on our own tasks before relying on it.
- Latency: 180 ms per call on an H100, or 790 ms with Think. Use the base model for S1. Think is too slow for a streaming S1.
- The released LeRobot integration means it needs the least adapter work of all candidates.

**Xiaomi-Robotics-0**
- Use: reference design for real-time chunking (prefix conditioning trained in). It has no RoboCasa checkpoint.
- Constraint: 80 ms covers policy inference only. The paper's π0.5 LIBERO row is openpi's published number (96.85), not a re-run.

**GR00T N1.6 / N1.7**
- Use: a fallback for real arms (it has SO100 and DROID examples) and for the original 24-task RoboCasa.
- Constraints: there is no RoboCasa365 checkpoint or number, and the weights licence conflicts with the repo README. The N1.6 card links a PDF named "NVIDIA-OneWay-Noncommercial-License"; we did not review it.

**Molmo2-ER**
- A standalone Apache-2.0 checkpoint that serves through vLLM and SGLang. It is an S2 pointing/grounding candidate, not an S1. The card gives no per-benchmark pointing score.

**Helix, Gemini Robotics**
- Design references only (a two-rate S1/S2 split). We cannot deploy either.

## Our measurements (`lerobot/pi052_robocasa`)

Setup: 20 Hz control, `latency_s` 0.3, remote S1. Source files: `results/delay-ab/*/*/summary.json`.

| Run set | Episodes | Setup | task_success |
|---|---|---|---|
| `results/delay-ab/20261022` | 6 | lockstep, realtime, rtc × 2 tasks; seed 1000; 60 s | 0/6. Realtime and rtc `s1_latency_est_s` 0.6–0.75 s. |
| `results/delay-ab/20261022-lockstep-full` | 4 | lockstep (no latency), native planner; seeds 1000/1001; 116 s / 76 s | 0/4 |
| `results/delay-ab/20261022-oracle` | 4 | lockstep, oracle planner fed from ground truth | 0/4 |

- In the oracle runs, ground truth advanced past "pick" in 2/4 episodes, both with seed 1000:
  - MicrowaveThawingFridge: at 16.3 s. It fell back to "pick" at 35.6 s and then oscillated between pick and place.
  - MoveFridgeToFreezer: at 18.15 s.
- No oracle episode completed a place step.
- The model fails with zero latency and with a perfect planner. The bottleneck is S1 competence on these tasks.
- This matches the paper's π0.5 composite results (7.1 / 1.2). Caveat: `pi052_robocasa` is a LeRobot port, not the paper's checkpoint.
- Sample sizes are 2 seeds × 2 tasks per set. These runs show the model fails; they say nothing about rates.

## Flip triggers

- XR-1 on a 5-task target50 smoke test, using our adapter, lands far below its published per-task rates (`summary.json`). Our adapter or observation mapping is wrong. Fix that before judging.
- XR-1 ignores subtask text (oracle steering with subtask text does no better than the whole-task instruction). The S1-steering assumption fails for this model: run it whole-task, or pick another S1.
- XR-1's end-to-end latency on our GPU comes close to or exceeds a 0.8 s chunk. Real time would then need asynchronous prefix execution, or RoboCasa v1 stays lockstep.
- The RoboCasa365 leaderboard lists an open checkpoint with higher per-split scores, or NVIDIA ships a RoboCasa365 N1.7 checkpoint with numbers. Re-rank.
- An independent Franka replication puts MolmoAct2's lead over π0.5 at under about 10 points. Prefer whichever has the cheaper adapter.
- MolmoAct2-SO100_101 fails our SO-101 self-evaluation. Fall back to the GR00T N1.7 SO100 example or to a fine-tune.

## Not verified

- XR-1: latency and VRAM, subtask-text conditioning, and the identity of the "2nd best 46.6%". The repo README says 57.4% while the card and `summary.json` say 57.28%. XR-1 runs 50 trials per task; the paper's protocol is 30, so the comparison is approximate. Both evaluate in pretrain scenes (vendored `vendor/robocasa/docs/benchmarking/multitask_learning.md`, commit a9e152e).
- The XR-1 per-split numbers are our own derivation. Split membership comes from `TARGET_TASKS` in the vendored registry; there is no vendor breakdown.
- The vendored RoboCasa docs table gives π0 as 34.6/6.1/1.1/14.8, against the paper's 36.3/5.2/0.7/15.0. π0.5 and GR00T N1.5 agree.
- MolmoAct2: the 87.1 vs 45.2 Franka result has no independent replication, and the π0.5 configuration it used is unstated. There are no SO-101 success rates. The base and SO100 cards have no licence field.
- Molmo2-ER: no per-benchmark pointing numbers. The parameter count reads 4B in the text and 5B in the metadata. The paper calls it "MolmoER" ([arXiv 2605.02881](https://arxiv.org/abs/2605.02881)).
- GR00T N1.7 LIBERO: the percentages disagree with the fractions (195/200 is 97.5, not 97.65; 189/200 is 94.5, not 94.35). The 24-task RoboCasa episode counts are unstated. The weights licence conflicts between card and README.
- Helix: the page does not say the weights are closed. We infer it because there is no release or download. Gemini Robotics: no API or licence terms for the VLA.
- `lerobot/pi052_robocasa`: no published evaluation, and the base model is not stated in the part of the card we read.
