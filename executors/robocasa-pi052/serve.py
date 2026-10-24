"""π0.5² as a remote System 1 policy: HTTP service for `rojev.system1.RemotePolicy` (runs on the GPU box, e.g. Kaggle).

Per /act request it does exactly what LeRobot's language runtime does for one chunk (`lerobot.runtime`):
raw env observation → `preprocess_observation` → checkpoint preprocessor → PI052 adapter `select_action` with the
given subtask (prompt "{subtask}, State: …") → postprocessor applied per action. No sim, no assets on this side.
An empty subtask runs π0.5²'s own planner instead: it regenerates the subtask from the task every request, keeps that
state across requests until the task text changes or /reset, and returns the chosen subtask in each chunk's `info`
(the native baseline).

Real-time chunking (`--rtc`): a request may carry a `Continuation` (the actions the robot runs from the observation's
time while this request is in flight, and how many run before the chunk lands). With --rtc the prefix is mapped into
the policy's action space by the checkpoint's own normalizer and passed to LeRobot's RTC guidance (`inference_delay`,
`prev_chunk_left_over`, `execution_horizon`; Black et al. 2025), so the new chunk is inpainted to agree with it
instead of jumping to a plan made without it. RTC's guidance needs autograd, so with a continuation the denoising
runs under `no_grad` instead of `inference_mode`. Without --rtc a continuation is ignored (`info["rtc"]` says which).

K samples (`samples` in the wire request): the subtask is generated once, then the conditioned batch is repeated K
times and π0.5²'s `sample_actions` draws K independent noises in one forward batch (`Draws`). The prefix (images +
prompt) is computed for each of the K rows: LeRobot's `sample_actions` builds the prefix KV cache and runs the
denoising loop in one method, so sharing one prefix across K draws would mean re-implementing it here.

  ROJEV_TOKEN=… uv run python serve.py --tokenizer-dir /path/to/paligemma --port 8000 \
      [--h3-port 4433 --cert cert.pem --key key.pem]
Endpoints (all need `Authorization: Bearer $ROJEV_TOKEN`): GET /health · POST /reset · POST /act (rojev.system1.wire),
over HTTP/1.1 on loopback (SSH tunnel) and, with --h3-port, over HTTP/3 (QUIC) on all interfaces with a pinned
certificate; both serve one `rojev.transport.Endpoint` around `rojev.system1.PolicyService.handle` (token and MAX_BODY checked before a
body is buffered; a retried request id is answered from the replay cache, never run twice).
Exits after --idle-min minutes without requests (frees the GPU quota).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs import RTCAttentionSchedule
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.processor import TransitionKey
from lerobot.processor.converters import create_transition
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import GenerationConfig
from lerobot.runtime.registry import get_language_adapter_factory

from drive import PALIGEMMA, POLICY, load_policy
from embodiment import to_action_space
from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Continuation, Policy, PolicyService, serve_policy
from rojev.transport import Endpoint

MAX_BODY = 32 * 1024 * 1024


class Draws:
    """The policy as the PI052 adapter sees it when K chunks are wanted for one observation: `predict_action_chunk`
    repeats every tensor of the batch the adapter conditioned (images, state, prompt tokens and masks) `k` times along
    the batch dimension, so `sample_actions` draws k independent noises against one conditioning in one forward batch
    and returns k chunks (k × n × d)."""

    def __init__(self, policy, k: int, guidance: dict | None = None):
        self.policy, self.config, self.k, self.guidance = policy, policy.config, k, guidance or {}

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return self.policy.predict_action_chunk(
            {key: v.repeat(self.k, *([1] * (v.ndim - 1))) if isinstance(v, torch.Tensor) else v
             for key, v in batch.items()}, **self.guidance)


class Pi052Policy(Policy):
    """`planner_memory`: π0.5²'s own planner also writes its running memory note on each subtask change (LeRobot's
    default; reported in `info["memory"]`). Its subtask prompt is the task text only (`pi052_adapter.build_messages`),
    so the note does not condition the next subtask or the actions; off by default (no extra decode)."""

    def __init__(self, device: str, tokenizer: str, planner_memory: bool = False, rtc_horizon: int | None = None):
        self.device = device
        self.policy, self.pre, self.post = load_policy(device, tokenizer)
        self.rtc_horizon = rtc_horizon
        if rtc_horizon is not None:
            self.policy.config.rtc_config = RTCConfig(enabled=True, execution_horizon=rtc_horizon,
                                                      prefix_attention_schedule=RTCAttentionSchedule.EXP)
            self.policy.init_rtc_processor()
        self.normalize = next(step for step in self.pre.steps if type(step).__name__ == "NormalizerProcessorStep")
        self.factory = get_language_adapter_factory(self.policy.config.type)
        self.planner = self.factory(self.policy, GenerationConfig(enable_memory=planner_memory, enable_subtask=True,
                                                                  chunks_per_regen=1))
        self.plan_state: RuntimeState | None = None

    def _sync(self) -> float:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter()

    accepts_continuation = True

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None,
            continuation: Continuation | None = None) -> ActionChunk:
        return self.act_samples(obs, subtask, 1, timeout_s, continuation)[0]

    def guidance(self, continuation: Continuation | None, k: int) -> dict:
        """LeRobot's RTC arguments for `continuation` (empty without --rtc or without a continuation): the prefix in
        the policy's normalized action space, repeated for the K rows."""
        if self.rtc_horizon is None or continuation is None:
            return {}
        prefix = torch.as_tensor(np.asarray(continuation.actions[:self.policy.config.chunk_size], dtype=np.float32),
                                 device=self.device)
        prefix = self.normalize(create_transition(action=prefix))[TransitionKey.ACTION]
        return {"prev_chunk_left_over": prefix.unsqueeze(0).repeat(k, 1, 1),
                "inference_delay": continuation.delay_steps, "execution_horizon": self.rtc_horizon}

    def act_samples(self, obs: Observation, subtask: str, k: int, timeout_s: float | None = None,
                    continuation: Continuation | None = None) -> list[ActionChunk]:
        """K chunks for one observation: one preprocessing, one subtask (planner mode), one K-row forward batch; each
        chunk postprocessed per action and mapped by `to_action_space`. Stage times are shared by the K chunks."""
        t0 = self._sync()
        raw = {"pixels": {cam: img[None] for cam, img in obs.images.items()}, "agent_pos": obs.state[None]}
        batch = preprocess_observation(raw)
        batch["task"] = [obs.task]
        batch = {key: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for key, v in self.pre(batch).items() if isinstance(key, str) and key.startswith("observation.")}
        guidance = self.guidance(continuation, k)
        info = {"rtc": bool(guidance)}
        t1 = self._sync()
        with torch.inference_mode():
            if subtask:
                state = RuntimeState(task=obs.task)
                state.language_context["subtask"] = subtask
            else:
                if self.plan_state is None or self.plan_state.task != obs.task:
                    self.plan_state = RuntimeState(task=obs.task)
                self.planner.update_language_state(batch, self.plan_state)
                state = self.plan_state
                info["subtask"] = state.language_context.get("subtask")
                if state.language_context.get("memory"):
                    info["memory"] = state.language_context["memory"]
            t2 = self._sync()
        drawer = self.factory(Draws(self.policy, k, guidance), GenerationConfig(enable_memory=False,
                                                                                  enable_subtask=False))
        with torch.no_grad() if guidance else torch.inference_mode():
            chunks = drawer.select_action(batch, state)
        t3 = self._sync()
        actions = [to_action_space(np.stack([self.post(s.unsqueeze(0)).detach().float().cpu().numpy().reshape(-1)
                                             for s in chunk])).astype(np.float32) for chunk in chunks]
        t4 = self._sync()
        timings = {"preprocess": round((t1 - t0) * 1000, 1), "plan": round((t2 - t1) * 1000, 1),
                   "infer": round((t3 - t2) * 1000, 1), "postprocess": round((t4 - t3) * 1000, 1)}
        return [ActionChunk(a, timings, info) for a in actions]

    def reset(self) -> None:
        self.plan_state = None
        self.policy.reset()
        for p in (self.pre, self.post):
            p.reset()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer-dir", default=PALIGEMMA)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--idle-min", type=float, default=30)
    p.add_argument("--planner-memory", action="store_true",
                   help="π0.5²'s planner also writes its memory note on subtask changes (reported in info)")
    p.add_argument("--rtc", type=int, metavar="H",
                   help="real-time chunking with execution horizon H steps (uses each request's continuation)")
    p.add_argument("--h3-port", type=int, help="also serve HTTP/3 (QUIC, UDP) on this port, all interfaces")
    p.add_argument("--cert", type=Path, help="HTTP/3 certificate (made by rojev.transport.make_certificate)")
    p.add_argument("--key", type=Path, help="HTTP/3 private key")
    a = p.parse_args()
    if a.h3_port is not None and (a.cert is None or a.key is None):
        p.error("--h3-port needs --cert and --key")
    service = PolicyService(Pi052Policy(a.device, a.tokenizer_dir, a.planner_memory, a.rtc),
                      {"policy": POLICY, "device": a.device, "planner_memory": a.planner_memory, "rtc": a.rtc})
    endpoint = Endpoint(service.handle, os.environ["ROJEV_TOKEN"], MAX_BODY)
    serve_policy(endpoint, service, a.port, a.idle_min, (a.h3_port, a.cert, a.key) if a.h3_port is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
