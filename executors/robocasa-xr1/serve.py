"""Xiaomi-Robotics-1 (RoboCasa365 checkpoint) as a remote System 1 policy: `rojev.system1.PolicyService` on the box.

Loads the pinned checkpoint as its own deployment server does (transformers `AutoModel` / `AutoProcessor` with the
checkpoint's code, flash-attention 2, bfloat16), but behind our wire (npz, no pickle) and `Endpoint` (token, body
limit, replay cache) instead of its pickle socket, which would run any bytes sent to it. Per /act request:
history-encoded observation (`xr1_inputs.unpack`) → the checkpoint's prompt (`xr1_inputs.messages`; instruction =
the subtask if given, else the task) → processor → model → `decode_action` → the first `ACTION_DIM` values of each
of its 16 steps, as the model produced them (the robot side maps them into its action space). Stateless: the robot
side sends the history, so a retried request answers the same.

  ROJEV_TOKEN=… uv run python serve.py --model <snapshot dir> --port 8004
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy, PolicyService, serve_policy
from rojev.transport import Endpoint
from xr1_inputs import ACTION_DIM, ROBOT_TYPE, messages, unpack

MAX_BODY = 32 * 1024 * 1024
MODEL = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"


class XR1Policy(Policy):
    def __init__(self, model: Path, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(model, trust_remote_code=True, attn_implementation="flash_attention_2",
                                               dtype=torch.bfloat16).to(device).to(torch.bfloat16).eval()

    def _sync(self) -> float:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter()

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        t0 = self._sync()
        frames, state = unpack(obs.images, obs.state)
        inputs = dict(self.processor.apply_chat_template(messages(frames, subtask or obs.task), tokenize=True,
                                                         return_dict=True, return_tensors="pt", do_resize=False,
                                                         state=state, robot_type=ROBOT_TYPE))
        batch = {k: (v.to(self.device, torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point()
                     else v.to(self.device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        batch["task_id"] = ROBOT_TYPE
        t1 = self._sync()
        with torch.inference_mode():
            out = self.model(**batch)
        t2 = self._sync()
        actions = self.processor.decode_action(out.actions.float().cpu(), robot_type=ROBOT_TYPE)
        actions = np.asarray(actions, dtype=np.float32)[0, :, :ACTION_DIM]
        t3 = self._sync()
        return ActionChunk(actions, {"preprocess": round((t1 - t0) * 1000, 1), "infer": round((t2 - t1) * 1000, 1),
                                     "postprocess": round((t3 - t2) * 1000, 1)})

    def reset(self) -> None:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, required=True, help="local snapshot of the pinned checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8004)
    p.add_argument("--idle-min", type=float, default=30)
    a = p.parse_args()
    policy = XR1Policy(a.model, a.device)
    if a.device.startswith("cuda"):
        print(f"[xr1] GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    service = PolicyService(policy, {"policy": MODEL, "device": a.device}, name="xr1")
    serve_policy(Endpoint(service.handle, os.environ["ROJEV_TOKEN"], MAX_BODY), service, a.port, a.idle_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
