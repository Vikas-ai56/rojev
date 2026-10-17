"""Frozen-policy rollouts in RoboCasa with an explicit instruction string.

Mirrors `lerobot_eval.rollout` step for step (same env, policy, pre/post processors and env processors). The only
difference: the instruction is a parameter. `lerobot-eval` sends RoboCasa's natural-language `ep_meta["lang"]`
("Close the fridge doors."), but `smolvla_robocasa` was trained on task names ("CloseFridge"). By default the
instruction is the task name and must be in the checkpoint's training vocabulary; `--instruction lang` reproduces
lerobot-eval for A/B comparison.

  uv run python rollout.py --task CloseFridge --episodes 5
  uv run python rollout.py --task CloseFridge --episodes 5 --instruction lang   # what lerobot-eval sends
"""
import argparse
import json
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

import checkpoint

REPO = Path(__file__).resolve().parents[2]
VIDEO_CAMERA = "robot0_agentview_left"


def build(policy_id: str, task: str, split: str, device: str, obj_registries: list[str]):
    policy_cfg = PreTrainedConfig.from_pretrained(policy_id)
    policy_cfg.pretrained_path = Path(policy_id) if Path(policy_id).exists() else policy_id
    policy_cfg.device = device
    if hasattr(policy_cfg, "load_vlm_weights"):
        policy_cfg.load_vlm_weights = False  # final trained weights only (decision 2026-10-17)
    env_cfg = RoboCasaEnv(task=task, split=split, obj_registries=obj_registries)
    envs = make_env(env_cfg, n_envs=1)
    env = next(iter(next(iter(envs.values())).values()))
    rename = checkpoint.rename_map(policy_id)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg, rename_map=rename)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=policy_id,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)},
                                "rename_observations_processor": {"rename_map": rename}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy_cfg)
    return env, policy, pre, post, env_pre, env_post, rename


def run_episode(env, policy, pre, post, env_pre, env_post, instruction: str | None, seed: int, video: Path | None):
    """One episode. `instruction=None` sends the env's natural-language description, exactly like lerobot-eval."""
    policy.reset(); pre.reset(); post.reset()
    obs, info = env.reset(seed=[seed])
    lang = list(env.call("task_description"))[0]
    sent = lang if instruction is None else instruction
    max_steps = env.call("_max_episode_steps")[0]
    frames, success, step, t0 = [], False, 0, time.perf_counter()
    while step < max_steps:
        if video is not None:
            frames.append(np.asarray(obs["pixels"][VIDEO_CAMERA][0]).copy())
        o = preprocess_observation(obs)
        o["task"] = [sent]
        o = pre(env_pre(o))
        with torch.inference_mode():
            action = policy.select_action(o)
        action = env_post({ACTION: post(action)})[ACTION]
        obs, reward, terminated, truncated, info = env.step(action.to("cpu").numpy())
        step += 1
        success = success or bool(np.asarray(info.get("is_success", [False]))[0])
        if bool(terminated[0]) or bool(truncated[0]) or success:
            break
    if video is not None and frames:
        iio.imwrite(video, np.stack(frames), fps=20, codec="libx264")
    return {"seed": seed, "success": success, "steps": step, "wall_s": round(time.perf_counter() - t0, 1),
            "instruction_sent": sent, "env_lang": lang}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="lerobot/smolvla_robocasa")
    p.add_argument("--task", default="CloseFridge")
    p.add_argument("--instruction", default=None,
                   help="text sent to the policy; default = task name; 'lang' = env natural language (lerobot-eval)")
    p.add_argument("--allow-unseen-instruction", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--split", default="target")
    p.add_argument("--obj-registries", default="objaverse,lightwheel",
                   help="object packs to sample from; the target split needs objaverse categories")
    p.add_argument("--device", default="mps")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    use_lang = args.instruction == "lang"
    instruction = None if use_lang else (args.instruction or args.task)
    vocab = checkpoint.training_instructions(args.policy)
    if instruction is not None and instruction not in vocab and not args.allow_unseen_instruction:
        why = "unknown (no train_config.json)" if not vocab else f"{len(vocab)} strings"
        raise SystemExit(f"instruction {instruction!r} not in {args.policy}'s training vocabulary ({why}); "
                         f"pass --allow-unseen-instruction to run anyway")

    tag = "lang" if use_lang else "taskname"
    out = Path(args.out or REPO / f"results/runs/rollout_{args.task}_{tag}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    env, policy, pre, post, env_pre, env_post, rename = build(args.policy, args.task, args.split, args.device,
                                                              args.obj_registries.split(","))
    (out / "run_config.json").write_text(json.dumps(vars(args) | {"rename_map": rename,
                                                                  "instruction_in_training_vocab": instruction in vocab
                                                                  if instruction else None}, indent=2))
    rows = []
    for ep in range(args.episodes):
        r = run_episode(env, policy, pre, post, env_pre, env_post, instruction, args.seed + ep,
                        out / f"episode_{ep}.mp4")
        rows.append(r)
        (out / "episodes.json").write_text(json.dumps(rows, indent=2))
        print(f"ep {ep} seed {r['seed']}: {'SUCCESS' if r['success'] else 'fail'} in {r['steps']} steps ({r['wall_s']}s) "
              f"| sent {r['instruction_sent']!r} | env lang {r['env_lang']!r}", flush=True)
    n = sum(r["success"] for r in rows)
    summary = {"task": args.task, "instruction_mode": tag, "n_success": n, "n_episodes": len(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.task} [{tag}]: {n}/{len(rows)} -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
