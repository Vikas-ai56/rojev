# Discarded action tails as the next chunk's prefix (VLA-Rolling)

- **Paper:** VLA-Rolling: Fewer Denoising Steps, Better Control via Adaptive Action Reuse. Sicheng He, Yi Pan, ... Jianwei Yin (13 authors), OmniAI Group, ZJU ACES Lab, Zhejiang University. arXiv 2610.25043 (v1 18 Oct 2026, v3 21 Oct 2026). https://arxiv.org/abs/2610.25043, project page https://chenbao123ww.github.io/VLA-Rolling/
- **How found:** not in earlier transcripts (10d7aa5b), repo docs, `docs/paper/` citations or memory. Found by web search. Its abstract says "action tails are not merely discarded predictions".
- **Confidence it is the user's paper:** medium (~65%). Near misses: Action Upcycling (2609.34911) runs the tail itself instead of re-planning, and ChainVLA (2608.02326) trains a 1.2B policy that conditions on the unexecuted tail.

## Idea
- **What is discarded:** a flow-matching VLA predicts H actions, runs K and throws away the tail of L = H−K actions when it re-plans.
- **The claim:** the tail is a partly denoised guess at the next window. In about 98% of adjacent steps the observation and state barely change. Starting from the tail, 4 Euler steps (π0.5) or 6 (SmolVLA) match a 10-step chunk started from noise.
- **How the tail is reused:** it fills the first L slots of the next chunk as c·tail + (1−c)·noise, and the last K slots start from pure noise. Denoising then runs for N = clip(⌈(1−c_min)·N_full⌉, N_min, N_full) steps. The tail is a soft starting state, not a hard prefix.
- **Training:** the VLA stays frozen. A ~30M confidence Transformer predicts c for each action and is trained on demonstrations (4k steps). The method is therefore not fully training-free.

## Results (numbers as read in the paper)
- **Setup:** H=50, K=10 (X-VLA: H=30). Sync receding-horizon execution, 4× RTX 4090.
- **Success and cost per chunk:**

  | Model, benchmark | Success (%) | Cost (ms) |
  |---|---|---|
  | π0.5, LIBERO | 96.38 → 97.83 | 168.09 → 129.60 (−22.9%) |
  | SmolVLA, LIBERO-Plus | 20.74 → 35.82 | 130.90 → 120.97 |
  | MolmoAct2, LIBERO | 95.50 → 95.70 | 291.34 → 196.61 (−32.5%) |

- **Across all 16 model–benchmark pairs:** cost fell in all 16, with up to 1.48× speedup. Success rose in 14; X-VLA fell slightly on LIBERO-Plus and RoboTwin.
- **Ablation:** reusing the tail with no confidence estimate lowered π0.5 LIBERO success by 2.13 pp (table reading).
- **Real robot (AgileX PiPER, π0.5, 20 trials):** success 70% → 75%, cost 227.28 → 176.50 ms.

## Fit for our system
- **What we already do:** our runtime places each chunk at observation time, skips actions already in the past and can pass the queued actions as a `Continuation` (actions + `delay_steps`) to a policy that accepts it (`rojev/system1/policy.py`). Those queued actions are exactly the tail this paper reuses. The π0.5² service already uses them as a hard prefix through LeRobot RTC; VLA-Rolling is a cheaper, soft alternative.
- **XR-1 has no prefix input:** XR-1 (RoboCasa365 checkpoint, `accepts_continuation=False`) needs no new input, because VLA-Rolling only changes the starting state and the step count. Our service would still need a sampler hook (initial state plus start time) instead of noise drawn from the global RNG. The VLA weights stay untouched, but the confidence head must be trained on RoboCasa365 demos. Without it, the ablation suggests success drops.
- **Little to gain today:** XR-1 already uses only 5 Euler steps and in parity runs executes all 16 actions (no tail), so the saving is small. It only matters if we execute K < 16 or run async.
- **Lockstep:** sim evaluation is unaffected; the method only changes compute per call.
- **Tests:**
  - XR-1 lockstep with K=8: tail-start versus noise-start at N = 2, 3, 5. Measure success, chunk-to-chunk jump and ms per call.
  - A fixed c (no learned head) as the cheap baseline.

## Not verified
- How a per-slot mix maps onto one flow start time: Eq. 9 and Eq. 12 are not reconciled in the text.
- Ablation numbers: the text and Table 2 disagree.
- No async method or async numbers are reported (only the horizon bounds in App. E.2).
- Code release status.
- Whether it works with 5-step samplers such as XR-1.
