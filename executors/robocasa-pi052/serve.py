"""π0.5² as a remote System 1 policy: HTTP service for `rojev.system1.RemotePolicy` (runs on the GPU box, e.g. Kaggle).

Per /act request it does exactly what LeRobot's language runtime does for one chunk (`lerobot.runtime`):
raw env observation → `preprocess_observation` → checkpoint preprocessor → PI052 adapter `select_action` with the
given subtask (prompt "{subtask}, State: …") → postprocessor applied per action. No sim, no assets on this side.
An empty subtask runs π0.5²'s own planner instead: it regenerates the subtask from the task every request, keeps that
state across requests until the task text changes or /reset, and returns the chosen subtask in each chunk's `info`
(the native baseline).

K samples (`samples` in the wire request): the subtask is generated once, then the conditioned batch is repeated K
times and π0.5²'s `sample_actions` draws K independent noises in one forward batch (`Draws`). The prefix (images +
prompt) is computed for each of the K rows: LeRobot's `sample_actions` builds the prefix KV cache and runs the
denoising loop in one method, so sharing one prefix across K draws would mean re-implementing it here.

  ROJEV_TOKEN=… uv run python serve.py --tokenizer-dir /path/to/paligemma --port 8000 \
      [--h3-port 4433 --cert cert.pem --key key.pem]
Endpoints (all need `Authorization: Bearer $ROJEV_TOKEN`): GET /health · POST /reset · POST /act (rojev.system1.wire),
over HTTP/1.1 on loopback (SSH tunnel) and, with --h3-port, over HTTP/3 (QUIC) on all interfaces with a pinned
certificate; both serve one `rojev.transport.Endpoint` around `Service.handle` (token and MAX_BODY checked before a
body is buffered; a retried request id is answered from the replay cache, never run twice).
Exits after --idle-min minutes without requests (frees the GPU quota).
"""
import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import GenerationConfig
from lerobot.runtime.registry import get_language_adapter_factory

from drive import PALIGEMMA, POLICY, load_policy
from embodiment import PANDA_OMRON
from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy, wire
from rojev.transport import Endpoint, H3Server, HttpServer

MAX_BODY = 32 * 1024 * 1024


def to_action_space(actions: np.ndarray) -> np.ndarray:
    """π0.5² regresses PandaOmron's on/off switches (gripper, base mode) as numbers and overshoots ±1 slightly
    (measured up to 1.018); the robot takes them as switches. Map each to the level RoboCasa's own gym wrapper uses
    (robocasa/wrappers/gym_wrapper.py:125,137: below the threshold → low, else → high), so the chunk is in the
    robot's action space; continuous dimensions are left as the policy produced them."""
    out = actions.copy()
    for i, d in enumerate(PANDA_OMRON.dims):
        if d.threshold is not None:
            out[:, i] = np.where(out[:, i] >= d.threshold, d.high, d.low)
    return out


class Draws:
    """The policy as the PI052 adapter sees it when K chunks are wanted for one observation: `predict_action_chunk`
    repeats every tensor of the batch the adapter conditioned (images, state, prompt tokens and masks) `k` times along
    the batch dimension, so `sample_actions` draws k independent noises against one conditioning in one forward batch
    and returns k chunks (k × n × d)."""

    def __init__(self, policy, k: int):
        self.policy, self.config, self.k = policy, policy.config, k

    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        return self.policy.predict_action_chunk(
            {key: v.repeat(self.k, *([1] * (v.ndim - 1))) if isinstance(v, torch.Tensor) else v
             for key, v in batch.items()})


class Pi052Policy(Policy):
    def __init__(self, device: str, tokenizer: str):
        self.device = device
        self.policy, self.pre, self.post = load_policy(device, tokenizer)
        self.factory = get_language_adapter_factory(self.policy.config.type)
        self.planner = self.factory(self.policy, GenerationConfig(enable_memory=False, enable_subtask=True,
                                                                  chunks_per_regen=1))
        self.plan_state: RuntimeState | None = None

    def _sync(self) -> float:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter()

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        return self.act_samples(obs, subtask, 1)[0]

    def act_samples(self, obs: Observation, subtask: str, k: int) -> list[ActionChunk]:
        """K chunks for one observation: one preprocessing, one subtask (planner mode), one K-row forward batch; each
        chunk postprocessed per action and mapped by `to_action_space`. Stage times are shared by the K chunks."""
        t0 = self._sync()
        raw = {"pixels": {cam: img[None] for cam, img in obs.images.items()}, "agent_pos": obs.state[None]}
        batch = preprocess_observation(raw)
        batch["task"] = [obs.task]
        batch = {key: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for key, v in self.pre(batch).items() if isinstance(key, str) and key.startswith("observation.")}
        info = {}
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
            t2 = self._sync()
            drawer = self.factory(Draws(self.policy, k), GenerationConfig(enable_memory=False, enable_subtask=False))
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


class Service:
    """The routing handler both transports serve (wrapped in a `rojev.transport.Endpoint`, which authorizes, bounds
    and de-duplicates): `handle(method, path, headers, body) -> (status, body, type)`. `lock` serializes the model;
    `last` feeds the idle watchdog and /health; `about` is what /health reports besides the request count."""

    def __init__(self, policy: Policy, about: dict):
        self.policy, self.about = policy, about
        self.last = {"t": time.time(), "requests": 0}
        self.lock = threading.Lock()

    def handle(self, method: str, path: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes, str]:
        if method == "GET" and path == "/health":
            return 200, json.dumps({**self.about, "requests": self.last["requests"]}).encode(), "application/json"
        if method != "POST" or path not in ("/reset", "/act"):
            return 404, b"not found", "text/plain"
        if path == "/act":
            try:
                request = wire.decode_request(body)
            except ValueError as e:
                return 400, str(e).encode(), "text/plain"
        with self.lock:
            self.last["t"], self.last["requests"] = time.time(), self.last["requests"] + 1
            if path == "/reset":
                self.policy.reset()
                return 200, b"ok", "text/plain"
            chunks = self.policy.act_samples(request.observation, request.subtask, request.samples)
            print(f"[serve] act #{self.last['requests']} subtask={request.subtask!r} k={request.samples} "
                  f"{chunks[0].timings_ms}", flush=True)
            return 200, wire.encode_response(chunks), "application/octet-stream"


def serve(endpoint: Endpoint, service: Service, port: int, idle_min: float, h3: tuple[int, Path, Path] | None) -> None:
    httpd = HttpServer(endpoint, "127.0.0.1", port)
    if h3 is not None:
        h3_port, cert, key = h3
        H3Server(endpoint, "0.0.0.0", h3_port, cert, key).start()
        print(f"[serve] HTTP/3 on udp 0.0.0.0:{h3_port} (pinned certificate {cert})", flush=True)

    def watchdog():
        while time.time() - service.last["t"] < idle_min * 60:
            time.sleep(30)
        print(f"[serve] idle for {idle_min} min, shutting down", flush=True)
        httpd.shutdown()

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[serve] ready on 127.0.0.1:{port}", flush=True)
    httpd.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer-dir", default=PALIGEMMA)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--idle-min", type=float, default=30)
    p.add_argument("--h3-port", type=int, help="also serve HTTP/3 (QUIC, UDP) on this port, all interfaces")
    p.add_argument("--cert", type=Path, help="HTTP/3 certificate (made by rojev.transport.make_certificate)")
    p.add_argument("--key", type=Path, help="HTTP/3 private key")
    a = p.parse_args()
    if a.h3_port is not None and (a.cert is None or a.key is None):
        p.error("--h3-port needs --cert and --key")
    service = Service(Pi052Policy(a.device, a.tokenizer_dir), {"policy": POLICY, "device": a.device})
    endpoint = Endpoint(service.handle, os.environ["ROJEV_TOKEN"], MAX_BODY)
    serve(endpoint, service, a.port, a.idle_min, (a.h3_port, a.cert, a.key) if a.h3_port is not None else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
