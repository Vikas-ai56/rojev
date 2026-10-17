# v2 G0: frozen executor (RoboCasa + SmolVLA)

Plan: `docs/onejev_learned_policy_plan_v2.md` §5 G0.

## Done
- **0.1 Sources pinned:** `config/lerobot.commit` (0.6.2), `config/robocasa.commit` (1.0.1), `config/robosuite.commit` (1.5.2). All Apache/MIT-style `LICENSE` files are present (read before copying any code).
- **Executor env:** `executors/robocasa/` is its own uv project (`uv sync` there). It's separate because the pins conflict with the `rojev` env: robosuite needs `mujoco<3.10`, RoboCasa pins `mujoco==3.3.1` / `numpy==2.2.5`, LeRobot pins `transformers<5.6`. RoboCasa's `lerobot==0.3.3` pin is removed through `[[tool.uv.dependency-metadata]]`, the uv form of LeRobot's documented `--no-deps` route.
- **Resolved:** mujoco 3.3.1, numpy 2.2.5, torch 2.11.0 (**MPS available**), transformers 5.5.4, tianshou 0.4.10, gymnasium 1.4.0. `SmolVLAPolicy` and `lerobot.envs.robocasa` import cleanly. `setup_macros` has been run.

## Facts for the experiment
- The checkpoint is trained on `robocasa_target_human_unified` (50 tasks, 25,307 episodes, 20 fps) for 20k steps at batch 32. **No success rates are published.** LeRobot's CI runs only a 1-episode smoke test of 10 atomic tasks: CloseFridge, OpenCabinet, OpenDrawer, TurnOnMicrowave, TurnOffStove, CloseToasterOvenDoor, SlideDishwasherRack, TurnOnSinkFaucet, NavigateKitchen, TurnOnElectricKettle.
- **Inputs:** 3 cameras at 256×256 (left / wrist / right → `camera1/2/3` via `rename_map`) and a 16-number robot state. **Output:** 12-number action in [-1, 1] (base 4, control mode 1, end-effector position 3, rotation 3, gripper 1). The control-mode entry means an all-zero action is not a safe hold.
- `chunk_size = n_action_steps = 50`: 2.5 s of open-loop actions per chunk.

## Pending approval: downloads (~6 GB, ~9–10 GB on disk; 320 GB free)
| Item | Size |
|---|---|
| Kitchen assets: tex 0.54 + tex_generative 1.18 + fixtures_lw 0.53 + objs_lw 0.79 | 3.05 GB zipped |
| `lerobot/smolvla_robocasa` checkpoint | 0.91 GB |
| `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` (pulled at load, because `load_vlm_weights: true`) | 2.03 GB |
| Not needed: objaverse 2.16 + aigen 5.77 GB | — |

## Next
0.2 render a RoboCasa scene on macOS (MUJOCO_GL=cgl) · 0.3 one CloseFridge rollout on MPS (shapes, success, memory, wall time) · 0.4 the 10 CI tasks → capability table.
