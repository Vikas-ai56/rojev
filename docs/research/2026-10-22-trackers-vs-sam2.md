# Trackers: anything lighter, faster and as accurate as SAM 2? (research brief, 2026-10-22, by Codex, read-only + web)

## Verdict

No released, permissively licensed model is simultaneously lighter, faster, and at least as accurate as SAM 2.1 Tiny across the relevant VOS benchmarks.

For rojev, use **SAM 2.1 Hiera Tiny on the A100** as the accuracy-first baseline, with one tracker state per camera and a robot-level identity layer above it. Benchmark **EdgeTAM** as the resource fallback: it is substantially lighter and faster, including an official CoreML path, but gives up measurable VOS accuracy.

Public VOS scores do not test cross-camera identity, grasp-induced occlusion, identical-object swaps, or attachment to a moving gripper. Those requirements need system-level state, regardless of tracker.

Scores below are `MOSE / DAVIS17 / LVOS / SA-V test`; `—` means not reported. Results are not fully protocol-normalized.

## Comparison

| Model | Params | Reported speed | Reported VOS accuracy | Prompt and occlusion behavior | Licence | Mac |
|---|---:|---|---|---|---|---|
| **[SAM 2.1 Tiny](https://github.com/facebookresearch/sam2)** | 38.9M | 91.2 FPS, compiled A100 | 71.8 / — / 77.3 / 76.5 | Point, box, mask; streaming memory, object-presence/occlusion handling | Apache-2.0 | Preliminary MPS; Meta warns of slower/degraded behavior |
| **[SAM 2.1 Small](https://github.com/facebookresearch/sam2)** | 46M | 84.8 FPS, compiled A100 | 73.5 / — / 78.3 / 76.6 | Same; modest accuracy gain over Tiny | Apache-2.0 | Preliminary MPS |
| **[EdgeTAM](https://github.com/facebookresearch/EdgeTAM)** | ≈15.7M executable components ([Qualcomm breakdown](https://aihub.qualcomm.com/compute/models/edgetam)) | 150.9 FPS compiled A100; 15.7 FPS iPhone 15 Pro Max | 70.0 / 87.7 / — / 71.7 | Point, box, mask; SAM-like memory | Apache-2.0 | **Yes: official CoreML/macOS export** |
| **[EfficientTAM-S](https://openaccess.thecvf.com/content/ICCV2025/html/Xiong_Efficient_Track_Anything_ICCV_2025_paper.html)** | 34M | Roughly 85–100 FPS A100 across released tables; mobile variant 28 FPS iPhone 15 Pro Max | 71.4 / 89.2 / 73.4 / 74.5 | Point, box, mask; compressed encoder and memory | Apache-2.0 | **Yes: official MPS path** |
| **[SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)** | 848M | Absolute single-object FPS **UNVERIFIED**; Meta reports a relative ~7× multi-object gain at 128 objects on H100 | **79.6 / 92.7 / 89.2 / 85.1** | **Text**, exemplar, point, box, mask; detects, segments and tracks; 16-object multiplexing | Custom SAM licence, not Apache/MIT | No official Mac runtime |
| **[Efficient-SAM2](https://arxiv.org/abs/2602.08224)** | Essentially host SAM 2.1 plus small cache components | Up to 1.68× SAM 2.1-L | Roughly −1 SA-V J&F versus host | Same prompts/memory; inference acceleration | Apache/BSD-derived code | **UNVERIFIED** |
| **[MobileSAM2](https://arxiv.org/abs/2607.12297)** | 5.8M / 10.4M / 23.7M | 13.3 / 10.8 / 8.2 FPS on RTX 4060 | 23.7M: 65.8 / 72.1 / 69.0 / 67.8 | SAM 2-style prompts/memory | Public implementation/licence **UNVERIFIED** | **UNVERIFIED** |
| **[SAM2Long](https://github.com/Mark12Ding/SAM2Long)** | Same host SAM 2 weights | Slower; exact deployable speed **UNVERIFIED** | Up to about +5 J&F on long-video tests ([paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Ding_SAM2Long_Enhancing_SAM_2_for_Long_Video_Segmentation_with_a_ICCV_2025_paper.pdf)) | Branching memory tree improves long occlusion/reappearance | CC BY-NC 4.0 portions; not permissive | **UNVERIFIED** |
| **[DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM)** | Same host T/S/B+/L weights plus memory logic | L ≈11.8 FPS A100 versus 13.2 baseline | L: LVOSv2 85.5; VOST 55.4 ([paper](https://arxiv.org/abs/2509.13864)) | Stronger distractor rejection and reacquisition | Apache-2.0 | **UNVERIFIED** |
| **[SAMURAI](https://github.com/yangchris11/samurai)** | Same host SAM 2.1 weights | Comparable/slower; primary FPS **UNVERIFIED** | Requested VOS sets **UNVERIFIED**; paper emphasizes VOT benchmarks | Motion-aware memory and Kalman prediction; useful under fast motion/occlusion | Apache-2.0 | **UNVERIFIED** |
| **[SAM-MT](https://github.com/FudanCVL/SAM-MT)** | Based on SAM 2.1 B+; total **UNVERIFIED** | 36+ FPS for 10 targets, RTX A6000 | Protocol-specific scores; not directly comparable | Clicks; sparse identity memory, overlap prevention, many targets | **No licence file found** | **UNVERIFIED** |
| **[Cutie](https://github.com/hkchengrex/Cutie)** | 35M | 65 FPS A100 in EdgeTAM comparison | 69.9 / 87.9 / — / 62.7 | Requires initial mask; object-level memory | MIT | **UNVERIFIED** |
| **[XMem](https://github.com/hkchengrex/XMem)** | ≈62M | ≈61 FPS A100 in comparative tables | 59.6 / 86.0 / — / 62.3 | Initial mask; excellent long-term memory design but older accuracy | MIT | **UNVERIFIED** |
| **[DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)** | ≈69M | 25–65 FPS depending protocol | 66.0 / 87.0 / — / 56.2 | Can ingest periodic detector masks and re-detect objects | CC BY-NC-SA 4.0 | **UNVERIFIED** |
| MobileSAM / EfficientSAM / EfficientViT-SAM | 9.8M / 25.3M / 34.8M | 278 / 183 / 762 images/s, A100 TensorRT FP16 ([comparison](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Zhang_EfficientViT-SAM_Accelerated_Segment_Anything_Model_Without_Performance_Loss_CVPRW_2024_paper.pdf)) | No VOS score | Image-only; must add Cutie/XMem, so no native occlusion memory | Apache-2.0 | ONNX possible; complete tracking path **UNVERIFIED** |
| **[CoTracker3](https://github.com/facebookresearch/co-tracker)** | **UNVERIFIED** | Hardware-comparable result **UNVERIFIED** | Not a mask/VOS model | Tracks points and visibility through occlusion; useful as supporting motion evidence only | CC BY-NC 4.0 | **UNVERIFIED** |

`Track-Anything` is primarily SAM plus XMem orchestration, not a newer tracker. `Grounded-SAM-2` is likewise a grounding-plus-SAM pipeline; its own repository says continuous ID tracking is still unstable ([official repository](https://github.com/IDEA-Research/Grounded-SAM-2)).

The reported FPS values are mostly single-stream, model-default-resolution measurements. They do not establish three concurrent 20 Hz sessions while the A100 is serving rojev’s other models. SAM 2.1 Tiny’s 91.2 FPS gives only about 1.5× theoretical throughput over the aggregate 60 camera-frames/s requirement.

## Recommendation

Use this arrangement:

```text
MJev point/text grounding at ~1 Hz
        ↓
SAM 2.1 Tiny tracker state per camera
        ↓
canonical robot object record
(track_id, semantic name, per-view masks, 3D pose/confidence,
 visibility, last-seen time, gripper-attachment state)
```

Why SAM 2.1 Tiny:

- It remains the best permissive accuracy/compute compromise.
- Its public SA-V, MOSE and LVOS results beat EdgeTAM and EfficientTAM despite their efficiency advantages.
- Point prompts directly match MJev-Spatial’s existing output, avoiding another resident detector.
- It has explicit object-presence handling and a maintained multi-object video predictor.
- The 38.9M weight count is small relative to the existing models, though activation and multi-session memory must still be measured.

For occlusion while carrying, the canonical track should remain alive from robot state: once a grasp is confirmed, associate the object pose with the end-effector transform, discount or freeze contaminated visual-memory updates, and re-prompt after reappearance. Cross-camera handoff should use calibrated depth/3D position plus appearance, not SAM’s internal object ID. No compared model provides persistent identity across three independent camera streams.

**Fallback:** EdgeTAM. Use it if SAM 2.1 Tiny misses the latency or co-residency limit and EdgeTAM passes a robot-domain non-inferiority test. It is the strongest Mac option because CoreML export is official, but the published 15.7 FPS iPhone result does not prove one-camera 20 Hz, much less aggregate 60 Hz, on the M5.

If Tiny meets latency but loses objects after difficult occlusions, try SAM 2.1 Small first. DAM4SAM/SAM2Long are useful research challengers for distractors and long occlusion, but neither is lighter or faster, and SAM2Long’s licence is unsuitable for the stated preference.

## Text prompt → mask → track

A verified single model now exists: **SAM 3/SAM 3.1 performs text-conditioned detection, segmentation and video tracking**. SAM 3 contains a shared encoder, text-conditioned detector and SAM-style tracker, totaling 848M parameters ([official model description](https://github.com/facebookresearch/sam3); [Meta overview](https://ai.meta.com/research/sam3)).

It is not the better production choice for rojev yet:

- It is over 20× SAM 2.1 Tiny’s parameter count.
- Official absolute single-object latency and incremental VRAM under co-residency are **UNVERIFIED**.
- Its licence is custom rather than Apache/MIT/BSD.
- Concept prompts are short noun phrases; they do not establish that arbitrary relational references such as “the cup left of the plate” outperform MJev-Spatial.
- SAM 3.1 is much more accurate on generic VOS, so it deserves a benchmark as a 1 Hz grounding/recovery model, not an assumed 20 Hz tracker.

Other text routes do not improve the overall proposition:

- Grounding DINO 1.5/DINO-X releases are API-oriented rather than permissively downloadable self-hosted weights ([official API repository](https://github.com/IDEA-Research/Grounding-DINO-1.5-API)).
- [Florence-2 base](https://huggingface.co/microsoft/Florence-2-base) is MIT and 232M, but is an image grounding model and still needs a tracker.
- OWLv2 is also an image detector, not a persistent mask tracker.
- Grounded-SAM-2 adds orchestration and another model but does not solve stable identity.

Therefore, keep **MJev → point → SAM 2.1 Tiny** unless SAM 3.1 wins a rojev-specific text-grounding and resource benchmark.

## What would flip the choice

Use these as measurable acceptance gates:

| Observation under the real co-resident workload | Decision |
|---|---|
| SAM 2.1 Tiny sustains ≥60 aggregate FPS, ≤30 ms per-camera p95, ≥2 GB peak GPU headroom, and causes ≤5% latency regression to existing models | Keep SAM 2.1 Tiny |
| EdgeTAM is no more than 1.0 J&F below Tiny on rojev data, has no more identity switches or false merges, and materially improves latency/headroom | Switch to EdgeTAM |
| EdgeTAM CoreML sustains ≥60 aggregate FPS for 30 minutes, ≤30 ms per-camera p95, with the same non-inferiority result | Move tracking to the Mac |
| Small or a distractor-aware method cuts duplicate/swap and post-occlusion failures by ≥50% while retaining the latency target | Prefer that accuracy variant |
| SAM 3.1 beats MJev→SAM by ≥2 percentage points on correct-object selection, including relational/duplicate prompts, while meeting latency, licence and GPU-headroom gates | Replace the two-model text path |
| Typical scenes contain ≥5 simultaneously tracked objects and SAM-MT obtains a usable commercial licence while meeting latency | Re-evaluate SAM-MT |

The decisive evaluation set should include clear views, partial and complete gripper occlusion, 0.25/0.5/1/2-second disappearances, carry motion, leave/re-enter, cross-camera handoff, identical duplicates and deliberate swaps. Measure mask J&F, identity switches per 1,000 visible frames, post-occlusion reacquisition, false merge/swap rate, p95/p99 latency, incremental VRAM, and interference with the VLA/VLM services.

