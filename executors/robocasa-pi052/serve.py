"""π0.5² as a remote executor: HTTP service for `rojev.executor.RemoteExecutor` (runs on the GPU box, e.g. Kaggle).

Per /act request it does exactly what LeRobot's language runtime does for one chunk (`lerobot.runtime`):
raw env observation → `preprocess_observation` → checkpoint preprocessor → PI052 adapter `select_action` with the
given subtask (prompt "{subtask}, State: …") → postprocessor applied per action. No sim, no assets on this side.

  ROJEV_TOKEN=… uv run python serve.py --tokenizer-dir /path/to/paligemma --port 8000
Endpoints (all need `Authorization: Bearer $ROJEV_TOKEN`): GET /health · POST /reset · POST /act (rojev.executor.wire).
Exits after --idle-min minutes without requests (frees the GPU quota).
"""
import argparse
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.runtime import RuntimeState
from lerobot.runtime.adapter import GenerationConfig
from lerobot.runtime.registry import get_language_adapter_factory

from drive import PALIGEMMA, POLICY, load_policy
from rojev.executor import ActionChunk, Executor, Observation
from rojev.executor import wire

MAX_BODY = 32 * 1024 * 1024  # 3 cameras at 256² are ~0.6 MB; anything far larger is not a valid request


class Pi052Executor(Executor):
    def __init__(self, device: str, tokenizer: str):
        self.device = device
        self.policy, self.pre, self.post = load_policy(device, tokenizer)
        # Subtask generation off: the caller supplies the subtask (OneJev / fixed / oracle arm).
        self.adapter = get_language_adapter_factory(self.policy.config.type)(
            self.policy, GenerationConfig(enable_memory=False, enable_subtask=False))

    def _sync(self) -> float:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return time.perf_counter()

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        t0 = self._sync()
        raw = {"pixels": {cam: img[None] for cam, img in obs.images.items()}, "agent_pos": obs.state[None]}
        batch = preprocess_observation(raw)
        batch["task"] = [obs.task]
        batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                 for k, v in self.pre(batch).items() if isinstance(k, str) and k.startswith("observation.")}
        state = RuntimeState(task=obs.task)
        state.language_context["subtask"] = subtask
        t1 = self._sync()
        with torch.inference_mode():
            chunk = self.adapter.select_action(batch, state)
        t2 = self._sync()
        steps = chunk[0] if chunk.ndim == 3 else chunk
        actions = np.stack([self.post(s.unsqueeze(0)).detach().float().cpu().numpy().reshape(-1) for s in steps])
        t3 = self._sync()
        return ActionChunk(actions.astype(np.float32), {"preprocess": round((t1 - t0) * 1000, 1),
                                                        "infer": round((t2 - t1) * 1000, 1),
                                                        "postprocess": round((t3 - t2) * 1000, 1)})

    def reset(self) -> None:
        self.policy.reset()
        for p in (self.pre, self.post):
            p.reset()


def serve(executor: Pi052Executor, token: str, port: int, idle_min: float) -> None:
    last = {"t": time.time(), "requests": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # one line per request, below
            pass

        def _authorized(self) -> bool:
            ok = hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")
            if not ok:
                self.send_error(401)
            return ok

        def _reply(self, body: bytes, ctype: str = "application/octet-stream"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health" and self._authorized():
                self._reply(json.dumps({"policy": POLICY, "device": executor.device,
                                        "requests": last["requests"]}).encode(), "application/json")

        def do_POST(self):  # noqa: N802
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                self.send_error(413)
                return
            body = self.rfile.read(size)  # read before rejecting, so the client gets a clean 401 (not a broken pipe)
            if not self._authorized():
                return
            with lock:
                last["t"], last["requests"] = time.time(), last["requests"] + 1
                if self.path == "/reset":
                    executor.reset()
                    self._reply(b"ok", "text/plain")
                elif self.path == "/act":
                    obs, subtask = wire.decode_request(body)
                    chunk = executor.act(obs, subtask)
                    print(f"[serve] act #{last['requests']} subtask={subtask!r} {chunk.timings_ms}", flush=True)
                    self._reply(wire.encode_response(chunk))
                else:
                    self.send_error(404)

    httpd = HTTPServer(("127.0.0.1", port), Handler)

    def watchdog():
        while time.time() - last["t"] < idle_min * 60:
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
    a = p.parse_args()
    serve(Pi052Executor(a.device, a.tokenizer_dir), os.environ["ROJEV_TOKEN"], a.port, a.idle_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
