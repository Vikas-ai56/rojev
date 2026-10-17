# Citations index

- Full BibTeX: `docs/paper/references.bib`. arXiv metadata pulled from the arXiv API; nothing typed from memory.
- Add every new source here and in the .bib in the same change.
- Curation (2026-10-17): research papers the argument rests on + SmolVLA / π0-family papers, blogs and checkpoints + our two models. No codebases, tool docs or one-off side references.

| Key | Source | Title | Used for |
|---|---|---|---|
| `ahn2022saycan` | arXiv:2204.01691 | SayCan | usefulness x feasibility |
| `black2024pi0` | arXiv:2410.24164 | $π_0$: A Vision-Language-Action Flow Model for General Robot Control | new tasks need fine-tuning |
| `black2025realtimechunking` | arXiv:2506.07339 | RealTimeChunking | async chunk transitions (RTC) |
| `brohan2023rt2` | arXiv:2307.15818 | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic  | 'no new motions' limitation |
| `chen2026steerablevla` | arXiv:2602.13193 | Steerable Vision-Language-Action Policies for Embodied Reasoning and H | external-VLM planner steering a VLA (LeRobot planner follows it) — closest prior work |
| `collaboration2023openxembodiment` | arXiv:2310.08864 | Open X-Embodiment: Robotic Learning Datasets and RT-X Models | cross-robot skill transfer within training robots |
| `fei2025liberoplus` | arXiv:2510.13626 | LIBERO-Plus | instruction ignoring, perturbations |
| `garcia2024gembench` | arXiv:2410.01345 | GemBench | generalization levels |
| `intelligence2025pi05` | arXiv:2504.16054 | pi05 | hierarchical subtask + action inference |
| `intelligence2025pistar06` | arXiv:2511.14759 | $π^{*}_{0.6}$: a VLA That Learns From Experience | per-task RL improvement |
| `intelligence2026pi07` | arXiv:2604.15483 | $π_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emerge | unseen tasks 60-80%, cross-embodiment folding; 'remixing' caveat |
| `mees2021calvin` | arXiv:2112.03227 | CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long- | long-horizon chaining collapse |
| `nasiriany2024robocasa` | arXiv:2406.02523 | RoboCasa | simulator/benchmark |
| `onejev` | https://github.com/OmniJev/OneJev | OneJev (qev) | typed decision model + server |
| `pi052_robocasa` | https://huggingface.co/lerobot/pi052_robocasa | lerobot/pi052_robocasa | executor (π0.5²; no paper) |
| `pi07_blog` | https://www.pi.website/blog/pi07 | π0.7 blog post | laundry = cross-embodiment transfer; zero-shot air fryer caveat |
| `qin2026mjevspatial` | arXiv:2610.22276 | MJev-Spatial | spatial-evidence model (decision: MJev before OneJev) |
| `ren2023knowno` | arXiv:2307.01928 | KnowNo | conformal sets, ask for help |
| `shi2025hirobot` | arXiv:2502.19417 | HiRobot | hierarchical VLM over pi0, 1 s cadence |
| `shukor2025smolvla` | arXiv:2506.01844 | SmolVLA | executor #1; async inference |
| `smolvla_robocasa` | https://huggingface.co/lerobot/smolvla_robocasa | lerobot/smolvla_robocasa | executor #1 |
| `sridhar2025memer` | arXiv:2510.20328 | MemER | async hierarchy, keyframe memory, latency limits |
| `team2024octo` | arXiv:2405.12213 | Octo: An Open-Source Generalist Robot Policy | novel skill 5% zero-shot |
| `team2025geminirobotics` | arXiv:2503.20020 | Gemini Robotics: Bringing AI into the Physical World | ALOHA generalization axes; zero-shot embodiment = future work |
| `team2025geminirobotics15` | arXiv:2510.03342 | GeminiRobotics15 | orchestrator calls VLA as a tool; points as output |
| `yang2026craft` | arXiv:2610.00524 | Same Scene, Different Task: Skill Alignment for Compositional Generali | undemonstrated combinations 0.7-8.4% for pi0.5; vision shortcut |
| `zhang2026robocombine` | arXiv:2610.15425 | RoboCombine | handoff latch, start-state gap, ordered/transition metrics; code reused (MIT) |
| `zheng2023pride` | arXiv:2309.03882 | PriDe | option-order bias + debiasing |
| `zhou2025liberopro` | arXiv:2510.03827 | LIBERO-PRO | memorization vs robustness |
