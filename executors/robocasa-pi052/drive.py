"""Drive π0.5² (`lerobot/pi052_robocasa`) in RoboCasa with a chosen subtask source; record success and subtasks.

Built from LeRobot's own language runtime (`lerobot.runtime`, the code behind `lerobot-rollout --sim`): same env
factory, policy/processor loading and PI052 adapter. Additions: env success is recorded per step, each subtask change
is logged, and the subtask source is a parameter:
  own      π0.5² generates the subtask from the instruction (its own planner; arm A)
  direct   the instruction is fed to the action expert as the subtask, unchanged (no planner)
The supervisor (OneJev) plugs in at the same point as `direct`: `runtime.state.set_context("subtask", ...)`.

One tick = one action dispatched = one env step (ctrl_hz=0: no wall-clock catch-up); the language head runs only
between action chunks (synchronous, timing variant T1).
Writes <out>/{run_config.json,episode.json} and an annotated mp4.

  MUJOCO_GL=egl uv run python drive.py --env-task CloseFridge --instruction "close the fridge" --planner own
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.runtime.adapter import GenerationConfig
from lerobot.runtime.language_runtime import LanguageConditionedRuntime
from lerobot.runtime.registry import get_language_adapter_factory
from lerobot.runtime.sim_robocasa import RoboCasaSimBackend, create_sim_env

POLICY = "lerobot/pi052_robocasa"
PALIGEMMA = "google/paligemma-3b-pt-224"  # gated tokenizer; --tokenizer-dir serves it from a local copy


def load_policy(device: str, tokenizer: str):
    """`lerobot.runtime.cli._load_policy_and_preprocessor` plus the tokenizer location.

    PaliGemma is named in two places: the checkpoint's preprocessor steps (overridden here) and the policy config's
    `tokenizer_name`, which the PI052 runtime reads for subtask generation and prompts.
    """
    cfg = PreTrainedConfig.from_pretrained(POLICY)
    cfg.pretrained_path, cfg.device = POLICY, device
    cfg.compile_model = cfg.gradient_checkpointing = False  # same as the runtime: variable prompts, inference only
    cfg.tokenizer_name = tokenizer
    policy = get_policy_class(cfg.type).from_pretrained(POLICY, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=POLICY, preprocessor_overrides={
        "device_processor": {"device": device},
        "pi052_text_tokenizer": {"tokenizer_name": tokenizer},
        "action_tokenizer_processor": {"paligemma_tokenizer_name": tokenizer}})
    return policy, pre, post


class RecordingBackend(RoboCasaSimBackend):
    """RoboCasaSimBackend that keeps env 0's success flag instead of discarding `info`."""

    success = False
    steps = 0

    def action_executor(self, action) -> None:
        if self.postprocessor is not None:
            action = self.postprocessor(action)
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).detach().to("cpu").numpy() if action.ndim > 1 else action.detach().cpu().numpy()
        row = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, _r, _term, _trunc, info = self.env.step(np.tile(row, (self.env.num_envs, 1)))
        self._last_obs = obs
        self._capture_frame()
        self.steps += 1
        self.success = self.success or bool(np.asarray(info.get("is_success", [False]))[0])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-task", default="CloseFridge", help="RoboCasa task class (scene)")
    p.add_argument("--instruction", default="close the fridge")
    p.add_argument("--planner", choices=["own", "direct"], default="own")
    p.add_argument("--split", default="pretrain")
    p.add_argument("--obj-registries", default="objaverse,lightwheel")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--after-success", type=int, default=20, help="steps kept in the video after success")
    p.add_argument("--chunks-per-gen", type=int, default=1, help="own planner: regenerate subtask every N chunks")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-dir", default=PALIGEMMA, help="local copy of the PaliGemma tokenizer (default: Hub)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_config.json").write_text(json.dumps({**vars(a), "out": str(a.out), "policy": POLICY}, indent=2))

    t0 = time.perf_counter()
    env, obs = create_sim_env(task=a.env_task, split=a.split, seed=a.seed,
                              obj_registries=[r for r in a.obj_registries.split(",") if r])  # before CUDA init
    t_env = time.perf_counter() - t0
    policy, pre, post = load_policy(a.device, a.tokenizer_dir)
    t_load = time.perf_counter() - t0 - t_env
    gen = GenerationConfig(enable_memory=False, enable_subtask=a.planner == "own", chunks_per_regen=a.chunks_per_gen)
    backend = RecordingBackend(env=env, last_obs=obs, task=a.env_task, seed=a.seed, device=a.device,
                               preprocessor=pre, postprocessor=post, record=True, output_dir=str(a.out))
    runtime = LanguageConditionedRuntime(policy_adapter=get_language_adapter_factory(policy.config.type)(policy, gen),
                                         observation_provider=backend.observation_provider,
                                         action_executor=backend.action_executor, event_collector=None,
                                         ctrl_hz=0)  # >0 pops round(lag*hz) queued actions and runs only the last
                                                     # (real-robot catch-up); in sim that skips actions
    backend.bind_runtime(runtime)
    runtime.set_task(a.instruction)
    runtime.state.set_context("subtask", a.instruction if a.planner == "direct" else None)
    runtime.state["mode"] = "action"

    subtasks, last, success_step, t_run = [], None, None, time.perf_counter()
    try:
        while backend.steps < a.max_steps:
            with torch.inference_mode():
                runtime.step_once()
            cur = runtime.state.language_context.get("subtask")
            if cur != last:
                subtasks.append({"step": backend.steps, "subtask": cur})
                print(f"[step {backend.steps}] subtask: {cur!r}", flush=True)
                last = cur
            if backend.success and success_step is None:
                success_step = backend.steps
                print(f"[step {backend.steps}] SUCCESS", flush=True)
            if success_step is not None and backend.steps >= success_step + a.after_success:
                break
    finally:
        runtime.stop()
        backend.disconnect()
    run_s = time.perf_counter() - t_run
    episode = {"env_task": a.env_task, "instruction": a.instruction, "planner": a.planner, "seed": a.seed,
               "success": success_step is not None, "success_step": success_step, "steps": backend.steps,
               "subtasks": subtasks, "env_setup_s": round(t_env, 1), "policy_load_s": round(t_load, 1),
               "run_s": round(run_s, 1), "steps_per_s": round(backend.steps / run_s, 2) if run_s else None,
               "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None}
    (a.out / "episode.json").write_text(json.dumps(episode, indent=2))
    print(json.dumps(episode, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
