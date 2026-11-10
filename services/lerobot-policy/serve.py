"""The LeRobot policy service: one model config (`config/s1/<name>.yaml`) served as a remote S1 behind our wire.

`model_config.load_model_config` reads the config, `adapter.load` builds the `LeRobotPolicy` (pinned checkpoint and
overrides from the local Hugging Face cache), and `rojev.system1.PolicyService` + `serve_policy` serve it exactly as
every other S1 service: our wire and `Endpoint` (token, body limit, replay cache), 127.0.0.1 only, stopping after
`--idle-min` minutes without a request. /health reports the config, the checkpoint at its revision, the overrides,
the camera map, the state key, the action width and the device. `accepts_continuation` stays False: the generic
path calls `predict_action_chunk` without LeRobot's real-time chunking inputs.

  ROJEV_TOKEN=… uv run python serve.py --config config/s1/groot-libero-goal.yaml --port 8007
"""
import argparse
import os
from pathlib import Path

from adapter import load
from model_config import ROOT, ModelConfig, load_model_config
from rojev.system1 import PolicyService, serve_policy
from rojev.transport import Endpoint

MAX_BODY = 32 * 1024 * 1024


def about(config: ModelConfig, policy_type: str) -> dict:
    return {"policy": f"LeRobot {policy_type}", "config": config.name, "checkpoint": config.checkpoint.repo,
            "revision": config.checkpoint.revision, "overrides": {k: str(v) for k, v in config.overrides.items()},
            "cameras": config.cameras, "state": config.state, "action_dim": config.action_dim,
            "device": config.device, "compiled": list(config.compile_modules), "compile_mode": config.compile_mode}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="model config, repo-relative (config/s1/<name>.yaml)")
    p.add_argument("--port", type=int, default=8007)
    p.add_argument("--idle-min", type=float, default=30)
    a = p.parse_args()
    config = load_model_config(ROOT / a.config)
    policy = load(config)
    warm_up_ms = policy.warm_up(config.warmup)
    print(f"[lerobot-policy] warm-up {config.warmup} request(s): {warm_up_ms} ms", flush=True)
    service = PolicyService(policy, {**about(config, policy.policy_type), "warm_up_ms": warm_up_ms},
                            name="lerobot-policy")
    serve_policy(Endpoint(service.handle, os.environ["ROJEV_TOKEN"], MAX_BODY), service, a.port, a.idle_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
