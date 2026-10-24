# Xiaomi-Robotics-1 (RoboCasa365) as a remote S1: integration plan (research brief, 2026-10-22)

Findings and a plan, not a decision. Read-only analysis; nothing was run.
Sources, cited `file:line`:
- `XR1:` = github.com/XiaomiRobotics/Xiaomi-Robotics-1 @ `0dd7aef` (2026-09-09).
- `HF:` = huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365 @ `3a6d029` (2026-08-25). Code and config files were read; the weights were not downloaded.
- Ours = this repo at the working tree.

## Verdict
- **The fit is good.** The action is the same 12-dim PandaOmron vector our LeRobot env takes, at 20 Hz. Action
  normalization is the identity. The cameras, the 256² frames and the vertical flip match.
- **Needed:**
  - a new executor service on our wire (no pickle);
  - a pure-function 16→14 state conversion;
  - the exact prompt;
  - **observation history (4 frames, 2 ticks apart) sent by the robot side**. This is a core change: the wire,
    the runner and the loop.
- **Single GPU is fine:** one model per GPU, about 10.1 GB of BF16 weights. It needs Ampere or newer for
  flash-attn 2, so Kaggle's T4 / P100 are out.
- **Parity runs need `--lockstep` and `--refill ≤ dt`.** A 16-step chunk (0.8 s) cannot cover our WAN latency in
  real-time mode: the floor ⌈2L/dt⌉+1 is about 24 at L = 0.57 s.

## a. Model inputs and outputs per query
| Item | Exact value | Source |
|---|---|---|
| Cameras | `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand` (prompt order Left, Right, Wrist), uint8 H×W×3 | `XR1:eval_robocasa365/entry.py:33-37,134-159` |
| Resolution | 256×256 (RoboCasa gym wrapper default) | `vendor/robocasa/robocasa/wrappers/gym_wrapper.py:46` |
| Crop | Center crop to `int(256·0.95)=243`, then bilinear resize **back to 256×256**; no other resize (`do_resize=False`) | `entry.py:92-104,136,176`; `HF:preprocessor_config.json` `do_resize:false`, patch 16, merge 2, temporal patch 2, mean/std 0.5 |
| Frames per camera | 4, passed as a "video" (list of PIL images) | `entry.py:81-89,135-148` |
| Which frames | Indices `max(0, n−1−(3−i)·2)`, i=0..3 → ticks **t−6, t−4, t−2, t**. The deque holds 7 observations (reset obs + one per `env.step`). Early in an episode, indices clamp to the oldest frame (the reset obs). | `entry.py:81-89,227-234,263-265` |
| State | 14-dim per frame, same 4 indices: `ee_pos_rel(3) + axis_angle(ee_quat_rel xyzw)(3) + gripper_qpos(2) + base_pos(3) + axis_angle(base_quat)(3)`. The quaternion is sign-canonicalized (w ≥ 0). | `entry.py:40-71` |
| State tensor | `(1, 4, 60)` float32, zero-padded after dim 14. **Not normalized**: the processor only casts it to bf16, and the model MLP-projects the raw values. | `entry.py:167-169`; `HF:processing_mibot.py:291-296`; `HF:modeling_mibot.py:1850`; `HF:config.json` `state_dim 60, state_length 4` |
| Language | `annotation.human.task_description` (ep_meta `lang`) inside a fixed template: user turn = `"Left camera: "<video>"\nRight camera: "<video>"\nWrist camera: "<video>"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"`, assistant turn = `"<cot></cot>"`, `apply_chat_template(tokenize=True, return_dict=True, return_tensors="pt", do_resize=False, state=…, robot_type=…)` | `entry.py:139-179,225`; `HF:chat_template.jinja` |
| Robot type | `robot_type="robocasa365"` (processor → `action_mask`); `task_id="robocasa365"` added to the request dict (server reads it, model ignores it via `**kwargs`) | `entry.py:30,177-181`; `XR1:deploy/server.py:58`; `HF:processing_mibot.py:118-130` |
| Other model args | `action_mask` (1,16,60) bf16 from `std > 1e-5`; `num_steps=5` (Euler flow matching); noise from the **global torch RNG** seeded at import: `MIBOT_SERVER_SEED` or `7 + first CUDA_VISIBLE_DEVICES index` | `HF:modeling_mibot.py:29-41,1821-1874` |
| Output | `actions` (1, **16**, 60) → `decode_action` = `x·std + mean` with mean 0 and std 1 on dims 0–11, std 0 elsewhere → client keeps `[0, :, :12]` | `HF:modeling_mibot.py:1866-1874`; `HF:processing_mibot.py:132-150`; `HF:preprocessor_config.json` `action_config.robocasa365` (16×60, checked: all means 0, std ∈ {0,1}); `entry.py:184` |
| Executed | **All 16** (`--replan-steps 16`), open loop, then a new query | `entry.py:244-257`; `XR1:scripts/launch_robocasa365.sh:44` |
| Action space | `ee_pos(3) ee_rot(3) gripper_close(1) base_motion(4) control_mode(1)`, the input of `robocasa.utils.env_utils.convert_action`. Identical to LeRobot's flat 12 and to `PANDA_OMRON`. **No mapping or unnormalization is needed.** The gripper and base mode are regressed values thresholded at 0.5 by the gym wrapper in both paths. | `vendor/robocasa/robocasa/utils/env_utils.py:142-154`; `vendor/lerobot-steerable/src/lerobot/envs/robocasa.py:101-113`; `executors/robocasa-pi052/embodiment.py:21-27`; `gym_wrapper.py:121-140` |
| Rate | 20 Hz: one action per `env.step` (robocasa `control_freq=20`) | `vendor/robocasa/robocasa/environments/kitchen/kitchen.py:381`; `embodiment.py:19` |

- Our `agent_pos` (16) → their state (14) is a pure function of one frame. Both are read from the same raw keys,
  so the quaternion convention is identical by construction:
  - `[0:3]`;
  - `aa([3:7])`;
  - `[14:16]`;
  - `[7:10]`;
  - `aa([10:14])`.
  
  Sources: `vendor/lerobot-steerable/src/lerobot/envs/robocasa.py:247-256` vs `entry.py:58-71`.
- `to_action_space` (`executors/robocasa-pi052/serve.py:57-66`) maps the switches with the same 0.5 rule as the
  gym wrapper, so it is reusable unchanged.

## b. Their env vs our `RoboCasaBackend`
| Aspect | XR1 client | Ours (`executors/robocasa-pi052/realtime.py:85-88`) | Changes results? |
|---|---|---|---|
| Wrapper | `gym.make("robocasa/<task>", split, seed=7)` → `RoboCasaGymEnv` (`entry.py:215`; `gym_wrapper.py:400-424`) | LeRobot `RoboCasaEnv` → same `RoboCasaGymEnv` (`robocasa.py:224-231`) | No: same images (flipped `[::-1]`, `gym_wrapper.py:271-273`), same `convert_action` / `unmap_action` |
| Split | `pretrain` (`launch_robocasa365.sh:41`) | `pretrain` | No |
| Object registries | Kitchen default `("objaverse","lightwheel")` (`kitchen.py:397-400`) | `("objaverse","lightwheel")` passed explicitly | No |
| Controller | `load_composite_controller_config(controller=None, robot="PandaOmron")` (`env_utils.py:81-84`) | Same function | No |
| Cameras / size | 3 cams, 256² (`gym_wrapper.py:36-47`) | 3 cams, 256² (`robocasa.py:44-48,132-133`; `workflow.py:49`); dict order differs | No, if the server selects cameras **by name** |
| Constructor seed | `seed=7` → robosuite rng at construction | None | Probably no: `reset(seed)` replaces `env.rng` (`gym_wrapper.py:342-346`); not verified for all RNG users |
| Episode seed | `7 + task_index·50 + episode` (`entry.py:222-224`) | `--seed` (default 1000) + `episode_index` 0 (`robocasa.py:274`) | Yes for scene identity; pass their seed to reproduce an episode |
| Task set | `target50` = atomic_seen + composite_seen + composite_unseen (`dataset_registry.py:2953-2955`), 50 trials each | Our 3 tasks (`tasks.py:92-96`) are pretrain composites, **not in target50** | XR1 has no published number on our tasks |
| Horizon | `get_task_horizon(task)` steps (450–4350; `dataset_registry_utils.py:240-253`; `summary.json`) | `--seconds·20` (robosuite horizon +1000, `ignore_done=True`) | Set `--seconds = horizon/20` for parity |
| Stop | Break on first `info["success"]` | `terminate_on_success=False`, success latched | No for success rate |
| Timing | Lockstep: sim waits for the model | Real-time by default; `--lockstep` available (`realtime.py:211-213,454-456`) | Yes: see (c) |
| Replan | Exactly every 16 steps | Runner requests when queue ≤ `max(refill_s, L+dt)` (`rojev/system1/runner.py:127-130,162`). Lockstep L = 0. Default `--refill 1.0` > 0.8 s chunk → **a request every tick**. With `--refill ≤ 0.05`, a request at 2 actions left → about 14 of 16 executed. | Yes: 16 needs lead 0 (runner change; see below) |
| Bounds | Raw actions to `env.step` (robosuite clips per controller) | Guard / runtime reports continuous overshoot (`docs/features/remote-models.md:46`) | Check what the guard does with \|a\| > 1 (not verified) |

## c. Wire, security, and where history lives
- **Their protocol:** a 4-byte big-endian length followed by a **pickle** in both directions (`XR1:deploy/server.py:46-64`,
  `deploy/client.py:40-55`).
  - `pickle.loads` on network bytes is arbitrary code execution for anyone who can reach the port. There is no
    auth or size limit (`server.py:49-51` trusts any 32-bit length). It serves one connection at a time
    (`listen(1)`). The client also unpickles replies, so a hostile server owns the client.
  - Their own post-training runtime already uses `np.load(allow_pickle=False)` (`XR1:xr1/mibot/server/runtime/server.py:46`).
  - `trust_remote_code=True` runs `HF:modeling_mibot.py` / `processing_mibot.py`. Pin revision `3a6d029`. Those
    files import only typing, numpy, torch, math, os, hashlib (unused) and transformers. The only side effect is
    `torch.manual_seed` at import.
- **Our replacement:** a new `executors/robocasa-xr1/serve.py` built like `executors/robocasa-pi052/serve.py:170-241`:
  - `Service.handle` serves `POST /act`, `POST /reset` and `GET /health`, under `Endpoint(handle, $ROJEV_TOKEN, MAX_BODY)`.
  - `HttpServer` on 127.0.0.1 behind the SSH tunnel; H3 optional.
  - `wire.decode_request` / `encode_response` (npz, `allow_pickle=False`, `rojev/system1/wire.py:71-100`).
  - Model and processor in-process, so there is no socket and no pickle. `XR1Policy.act_samples` does the following
    on the box:
    1. Frames → per camera, center crop 0.95 + resize to 256 → PIL.
    2. State → 14-dim conversion → (1,4,60).
    3. Instruction = `subtask or obs.task`.
    4. `processor.apply_chat_template(...)` exactly as `entry.py:171-181`.
    5. Tensors to device with the dtype rule of `server.py:59`.
    6. `model(**inputs)` → `decode_action(·,"robocasa365")[0,:,:12]` → `to_action_space` → `ActionChunk` (16×12)
       with timings `preprocess` / `infer` / `postprocess`.
  - `accepts_continuation = False`: the HF forward has no prefix input (`modeling_mibot.py:1821-1826`). The xr1
    post-training model's async prefix (`XR1:xr1/mibot/models/VLA/XR1.py:342-394`) is a different code path.
  - No own planner, so `info` carries no `subtask`. Arms `native` and `commit` (which need S1's planner) are not
    available.
  - K samples: start with K sequential forwards (the default `Policy.act_samples`, `rojev/system1/policy.py:50-56`).
    Batching K needs the VLM inputs repeated K times; measure before doing it.
- **History: the robot side sends it (recommended).**
  - **Server-side buffer: rejected.**
    - The server sees one observation per request, about every 14–16 ticks (`runner.py:166`, `loop.py:62-64`
      keep only the latest), so it never has ticks t−2, t−4, t−6.
    - Streaming every tick (≈ 20 × 26 KB/s) would make the service stateful: per-session ordering, `/reset` at
      every episode, two clients corrupting one buffer.
    - An abandoned (timed-out) request would still have mutated the buffer.
  - **Client-sent history: stateless server.**
    - A retried request id is answered from the replay cache (`docs/features/remote-models.md:27`). Re-running
      it would only draw new noise, which is still a valid chunk.
    - It is consistent with "chunk placed at observation time": every history frame is at or before `capture.t`,
      and action 0 is due at `capture.t` (`runner.py:201-202`), as XR1 was trained (the first action follows the
      newest frame).
  - **Structural shape (core change; needs a decision-log entry):**
    1. `Observation` gains `history: tuple[Observation, ...] = ()`: earlier frames, oldest first.
       `rojev/protocol/observation.py:8-14`.
    2. `Policy` declares `history: tuple[int, int] | None` (frames, interval ticks), like `accepts_continuation`.
       `rojev/system1/policy.py:39-40`.
    3. `ControlLoop` keeps a bounded ring of recent `Capture`s and returns ids `{id−6, id−4, id−2}`, clamped to the
       oldest kept. This matches XR1's `max(0,…)` from the episode's first observation. `rojev/control/loop.py:52,107-118`.
    4. `S1Runner` attaches the history when the policy declares one. `runner.py:166-172`.
    5. Wire v4 (`VERSION` 3→4): `hist/<i>/{img|webp}/<cam>` + `hist_state` (n×d). The π0.5² service redeploys
       with the new wire (`wire.py:29,104-106`).
    - Cost: 4× the frames, about 104 KB per request at WebP-90.
  - **Codec:** WebP-90 parity was measured on π0.5² only (`wire.py:11-15`). Use `raw` for the XR1 parity run, then
    A/B WebP.
- **Latency:** with a 16-step chunk, gap-free real-time supply needs ⌈2L/dt⌉+1 ≤ 16, so **L ≤ 0.375 s** end to end
  (`runner.py:132-137`).
  - WAN π0.5² was L ≈ 0.57 s (`remote-models.md:54,60`). Every request would hit `floor_over_chunk`, and the robot
    would starve between chunks.
  - Real-time needs a GPU close to the simulator. Parity and benchmark runs use `--lockstep`.
  - Exact 16-step replanning needs the runner to request at lead 0 in lockstep. Today the lead is
    `max(refill_s, L+dt)` ≥ dt (`runner.py:127-130`). Making the lead floor 0 is a runner-policy change (decide it;
    do not patch it per policy).

## d. Single GPU
- **Nothing needs multiple GPUs per model.**
  - `server.py:20`: `AutoModel.from_pretrained(..., attn_implementation="flash_attention_2", dtype=bfloat16).cuda()`
    loads one model on one device.
  - `scripts/deploy.sh:39-55` starts N independent servers, `CUDA_VISIBLE_DEVICES = i % NUM_GPUS`, one per port.
    The 8 servers are for 8 parallel episodes (`eval_robocasa365/README.md:44-65`).
  - There is no `device_map` or tensor parallelism in `HF:modeling_mibot.py`.
- **Memory:** weights 4.99 + 5.00 + 0.12 GB = **10.1 GB** BF16 (HF tree listing). Activations: about 3 × 128 video
  tokens plus text, through the DiT's 36 layers × 5 steps (an estimate). A 24 GB card (A10G, L4) should fit; A100 is
  comfortable.
- **Arch:** flash-attn 2.8.3 + bf16 needs **sm80+** (A100, A10G, L4, RTX 30/40). Kaggle T4 / P100 cannot run it.
  - `attn_implementation="sdpa"` might work (VLM `_supports_sdpa=True`, `modeling_mibot.py:560`; the DiT always uses
    SDPA, `:1664`).
  - But the top class declares only `_supports_flash_attn` (`:1749`), and transformers may refuse sdpa (not verified).
- **RNG:** the seed comes from the first visible device index or `MIBOT_SERVER_SEED`. Set `MIBOT_SERVER_SEED`
  explicitly for repeatable runs.

## e. Minimal ordered plan
| # | Add / change | Reuse unchanged |
|---|---|---|
| 1 | Decision-log entries: (i) XR1 as a second S1 in its own executor; (ii) history sent by the robot side, wire v4; (iii) the runner's lockstep lead | — |
| 2 | `executors/robocasa-xr1/pyproject.toml` + `uv.lock`:<br>• `requires-python ==3.12.*`<br>• torch 2.8.0 / torchvision 0.23.0 from an explicit `pytorch-cu128` index<br>• `transformers==4.57.1`<br>• `flash-attn==2.8.3` from the prebuilt wheel URL (`XR1:docs/DEPLOYMENT.md:27`, cp312 linux x86_64)<br>• `pillow`, `numpy`<br>• `rojev` editable<br>• `[tool.uv] environments = linux x86_64`, `package = false` | `rojev.transport`, `rojev.system1.wire` |
| 3 | `executors/robocasa-xr1/xr1_inputs.py`: `state14(agent_pos)`, `center_crop`, `build_messages`. A port of `entry.py:40-71,92-104,134-159`, no network. | — |
| 4 | Core history (section c, steps 1–5) + tests: loop ring and clamping, runner attach, wire v4 round trip. `tests/test_architecture.py` must stay green. | `Policy`, `RemotePolicy` (sends whatever is in the Observation) |
| 5 | `executors/robocasa-xr1/serve.py`: `XR1Policy` + `Service` + `main` (`--model` local snapshot at revision `3a6d029`, `--port`, `--idle-min`, `--seed`). Routing copies the π0.5² service's shape; `/reset` is a no-op (stateless). | `Endpoint`, `HttpServer`, `H3Server`, `wire`, `to_action_space`, `PANDA_OMRON` |
| 6 | Move `to_action_space` next to `PANDA_OMRON` (the robot plug-in owns its switch levels) where both executors can import it, e.g. a small shared `executors/robocasa-common` package. Do not copy it. | — |
| 7 | `executors/robocasa-xr1/tests/`: stand-in model test like `executors/robocasa-pi052/tests/test_serve.py`. **Parity test on the box:** recorded frames + states → our `XR1Policy` inputs (`input_ids`, `pixel_values_videos`, `video_grid_thw`, `state`) bit-equal to `entry.EvalClient` inputs; actions equal under the same seed. | — |
| 8 | `executors/robocasa-xr1/gpu/hook.sh` (key `xr1`, own port, `gpu_mem_mib` from a measurement, weights pre-downloaded on the box) | `scripts/gpu_setup.sh` interface (`executors/robocasa-pi052/gpu/hook.sh:10-27`) |
| 9 | `realtime.py`: `--policy remote --s1 xr1` picks `service_addresses(...)["xr1"]` (`workflow.py:65-74`, `realtime.py:266-268`). Parity run: `--lockstep --refill 0.05 --seconds horizon/20 --seed 7+i·50+e` on one target50 task (e.g. `CloseFridge`, 0.94). Their per-episode outcomes are in `XR1:eval_robocasa365/summary.json`. | `RoboCasaBackend`, `ControlLoop`, `S1Runner`, `ExecutionRuntime` |
| 10 | Docs in the same change: `docs/CODEBASE.md`, `docs/features/remote-models.md` (XR1 row, wire v4), `docs/features/system1-map.md` | — |

## Not verified
- What transformers 4.57.1 `apply_chat_template` does with a list of 4 PIL frames:
  - whether `do_sample_frames: true`, `fps: 2` (`HF:video_preprocessor_config.json`) drops or keeps frames;
  - the timestamp text it writes (`processing_mibot.py:246-253` defaults fps to 24 when metadata is missing).
  - Running the processor exactly as `entry.py` does makes this moot; step 7's parity test checks it.
- Whether the `task_id` kwarg reaching `self.vlm(**kwargs)` is ignored (it is in their server path, so pass it too).
- Inference latency per query, peak GPU memory, and whether sdpa loads without flash-attn.
- That `reset(seed)` alone reproduces their scenes despite the different constructor seed (`seed=7` vs None), and
  which robocasa commit their reference run used. `summary.json` names an internal `rng_stream` checkpoint path.
- XR1's training data: whether our three pretrain composite tasks were seen in post-training.
- Behaviour of our guard on continuous |a| > 1 vs robosuite's clipping.
- `rojev`'s own dependencies (`qev`, `mujoco`, `arize-phoenix-otel`, `aioquic`) co-resolving with torch 2.8 /
  transformers 4.57.1 in one uv env.
- The Python version: the HF card says 3.11 (`HF:README.md:26`), `DEPLOYMENT.md:17` says 3.12. The plan uses 3.12.
- Whether XR1 actions overshoot ±1 on the continuous dimensions.
