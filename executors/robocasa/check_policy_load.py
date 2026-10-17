"""G0-A pre-check: load a LeRobot policy from its final trained checkpoint only, with strict key matching.

`load_vlm_weights` is switched off so no base-VLM weights are downloaded and then overwritten. `strict=True` makes any
missing or unexpected tensor an error instead of a log line, so nothing can stay randomly initialised.

  uv run python check_policy_load.py --policy lerobot/smolvla_robocasa --device mps
"""
import argparse
import resource
import time

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="lerobot/smolvla_robocasa")
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.policy)
    cfg.device = args.device
    if hasattr(cfg, "load_vlm_weights"):
        cfg.load_vlm_weights = False
    t = time.perf_counter()
    policy = get_policy_class(cfg.type).from_pretrained(args.policy, config=cfg, strict=True)
    params = sum(t.numel() for t in policy.parameters())
    dtypes = sorted({str(t.dtype) for t in policy.parameters()})
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    print(f"loaded {args.policy} ({cfg.type}) strictly in {time.perf_counter() - t:.1f}s: {params / 1e6:.1f}M params, "
          f"dtypes {dtypes}, device {next(policy.parameters()).device}, peak RSS {peak_gb:.2f} GB")
    print(f"chunk_size {cfg.chunk_size}, n_action_steps {cfg.n_action_steps}, "
          f"inputs {list(cfg.input_features)}, output {dict((k, v.shape) for k, v in cfg.output_features.items())}")
    if torch.backends.mps.is_available():
        print(f"MPS allocated {torch.mps.current_allocated_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
