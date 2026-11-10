# Citations index

- Full BibTeX: `docs/paper/references.bib`. arXiv metadata pulled from the arXiv API (`export.arxiv.org/api/query?id_list=`); every id re-checked against its title on 2026-10-24; nothing typed from memory.
- Add every new source here and in the .bib in the same change.
- Curation: load-bearing research papers + SmolVLA / π0-family papers, blogs and checkpoints + our two models (OneJev, MJev). No codebases, tool docs or one-off side references.
- 2026-10-24: S1/S2 rewrite of `related-work.md`; removed `collaboration2023openxembodiment`, `team2025geminirobotics`, `garcia2024gembench` (no longer cited in `docs/`).

| Key | Source | Title | Used for |
|---|---|---|---|
| `ahn2022saycan` | arXiv:2204.01691 | Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (SayCan) | usefulness × feasibility; REVISE scoring |
| `assran2025vjepa2` | arXiv:2506.09985 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | world-model planning too slow for control |
| `black2024pi0` | arXiv:2410.24164 | $π_0$: A Vision-Language-Action Flow Model for General Robot Control | π0 family; Hi Robot's low level |
| `black2025realtimechunking` | arXiv:2506.07339 | Real-Time Execution of Action Chunking Flow Policies (RTC) | committed-prefix inpainting |
| `black2025trainingrtc` | arXiv:2512.05964 | Training-Time Action Conditioning for Efficient Real-Time Chunking | prefix conditioning learned in training |
| `brohan2023rt2` | arXiv:2307.15818 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | no new motions zero-shot |
| `cai2026xr0` | arXiv:2602.12684 | Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution | async recipe for XR-1 |
| `chen2025sarm` | arXiv:2509.25358 | SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation | stage-aware progress labels |
| `chen2026resteer` | arXiv:2603.17300 | ReSteer: Quantifying and Refining the Steerability of Multitask Robot Policies | steerability measured; mid-task switch data |
| `chen2026steerablevla` | arXiv:2602.13193 | Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control | external VLM steering a VLA |
| `fang2026molmoact2` | arXiv:2605.02881 | MolmoAct2: Action Reasoning Models for Real-world Deployment | open S2/S1 VLA; real-arm S1 candidate |
| `fei2025liberoplus` | arXiv:2510.13626 | LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models | low level ignores language |
| `figure2025helix` | figure.ai/news/helix | Helix: A Vision-Language-Action Model for Generalist Humanoid Control | latent S2→S1; completion action |
| `gao2025vlaos` | arXiv:2506.17561 | VLA-OS: Structuring and Dissecting Planning Representations and Paradigms in Vision-Language-Action Models | language vs grounded plans |
| `gu2025safe` | arXiv:2506.09937 | SAFE: Multitask Failure Detection for Vision-Language-Action Models | failure head on VLA features |
| `he2026vlarolling` | arXiv:2610.25043 | VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse | discarded tail as next start |
| `hou2026langgap` | arXiv:2603.00592 | LangGap: Diagnosing and Closing the Language Gap in Vision-Language-Action Models | low level ignores language; same-scene multi-task data |
| `hu2026hivla` | arXiv:2606.10267 | What Matters in Orchestrating Robot Policies: A Systematic Study of Hierarchical VLA Agents | false "done" dominates |
| `intelligence2025pi05` | arXiv:2504.16054 | $π_{0.5}$: a Vision-Language-Action Model with Open-World Generalization | text subtask then actions |
| `intelligence2025pistar06` | arXiv:2511.14759 | $π^{*}_{0.6}$: a VLA That Learns From Experience | RECAP value / advantage |
| `intelligence2026pi07` | arXiv:2604.15483 | $π_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities | new tasks = recombination; coaching |
| `khazatsky2024droid` | arXiv:2403.12945 | DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset | GR00T zero-shot reference data (E32) |
| `lee2026roboreward` | arXiv:2601.00675 | RoboReward: General-Purpose Vision-Language Reward Models for Robotics | VLM reward / done baseline |
| `li2025switchvla` | arXiv:2506.03574 | SwitchVLA: Execution-Aware Task Switching for Vision-Language-Action Models | mid-execution task switching |
| `liang2026robometer` | arXiv:2603.02115 | Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons | v2 critic base |
| `liu2026checkvla` | arXiv:2607.26789 | CheckVLA: Execution-Time Verification with Action-Conditioned World Model for Long-Horizon Mobile Manipulation | verification over a frozen VLA |
| `mees2021calvin` | arXiv:2112.03227 | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks | chaining collapse |
| `nasiriany2024robocasa` | arXiv:2406.02523 | RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots | simulator |
| `nasiriany2026robocasa365` | arXiv:2603.04356 | RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots | Composite-Unseen baselines |
| `nvidia2025gr00tn1` | arXiv:2503.14734 | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | jointly trained S2/S1; SO-101 S1 candidate (E32) |
| `onejev` | github.com/OmniJev/OneJev | OneJev (qev) | our S2 typed-choice model |
| `pi052_robocasa` | huggingface.co/lerobot/pi052_robocasa | lerobot/pi052_robocasa | earlier S1 (π0.5²) |
| `pi07_blog` | pi.website/blog/pi07 | π0.7 blog post | laundry = cross-embodiment; air-fryer caveat |
| `pistar06_blog` | pi.website/blog/pistar06 | π\*0.6 blog (RECAP) | value = negative steps to completion |
| `qin2026mjevspatial` | arXiv:2610.22276 | MJev-Spatial: System One Spatial Intelligence via Native Attention Matching | our spatial model; no calibration |
| `ren2023knowno` | arXiv:2307.01928 | Robots That Ask For Help: Uncertainty Alignment for Large Language Model Planners (KnowNo) | conformal sets, ask for help |
| `shi2025hirobot` | arXiv:2502.19417 | Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models | VLM over π0, interjections |
| `shukor2025smolvla` | arXiv:2506.01844 | SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics | async inference |
| `smolvla_robocasa` | huggingface.co/lerobot/smolvla_robocasa | lerobot/smolvla_robocasa | earliest executor |
| `sridhar2025memer` | arXiv:2510.20328 | MemER: Scaling Up Memory for Robot Control via Experience Retrieval | S2 latency limit |
| `team2024octo` | arXiv:2405.12213 | Octo: An Open-Source Generalist Robot Policy | novel skill 5 % |
| `team2025geminirobotics15` | arXiv:2510.03342 | Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer | orchestrator calls VLA as a tool |
| `wagh2026restore` | neurips.cc/virtual/2026/162913 | Restore, Don't Retry: Diagnosing and Recovering from Observation-Space Shift at Long-Horizon Skill Seams | seam failures |
| `xiaomi2026xr1` | arXiv:2607.15330 | Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Trajectories | our S1; 32.1 % Composite-Unseen |
| `yang2026craft` | arXiv:2610.00524 | Same Scene, Different Task: Skill Alignment for Compositional Generalization in VLAs | skill alignment, vision shortcut |
| `zhai2025vlac` | arXiv:2509.15937 | A Vision-Language-Action-Critic Model for Robotic Real-World Reinforcement Learning (VLAC) | progress delta + done |
| `zhang2026harnessvla` | arXiv:2607.08448 | Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents | frozen VLA as retryable primitive |
| `zhang2026hwm` | arXiv:2604.03208 | Hierarchical Planning with Latent World Models | hierarchical latent planning |
| `zhang2026robocombine` | arXiv:2610.15425 | Exploring the Limits of Combinatorial Generalization in Embodied Manipulation (RoboCombine) | hierarchy vs flat; start-state gap |
| `zheng2023pride` | arXiv:2309.03882 | Large Language Models Are Not Robust Multiple Choice Selectors (PriDe) | option-order bias |
| `zholus2026robojepa` | arXiv:2610.10515 | RoboJEPA: Scaling Robotic Latent World Models | JEPA control vs π0.5 |
| `zhou2025liberopro` | arXiv:2510.03827 | LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization | memorization vs robustness |
