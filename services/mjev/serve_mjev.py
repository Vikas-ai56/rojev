"""MJev-Spatial evidence service (cloud GPU): typed spatial questions about the current camera images.

Loads the checkpoint exactly as MJev's `scripts/predict.py` does (same config overrides), then — to fit a 15 GB T4 —
stores the language trunk and vision tower in fp16 (default; bf16 optional) while the native Q/K readout stays fp32 (as
`native_readout_fp32` asks), and runs the patch embedding as its exact matrix-multiply form. Upstream keeps fp32
language weights; both half precisions match fp32 top-1 on 12/12 judgements (E11, E12f). On a T4, fp16 = 1.3 s per
3-question request vs 3.5 s bf16 (E12f).

  ROJEV_TOKEN=… uv run python serve_mjev.py --checkpoint <dir> --device cuda:0 --port 8001
POST /ask (Bearer token), JSON:
  {"images": [<base64 PNG>, …], "questions": [{"id": …, "question": …, "answer_space": {…MJev contract…}}]}
→ {"answers": {id: {"prediction": …, "probabilities": {option: p} (choice), "path": […]}}, "timings_ms": {…}}
Every question sees all images. GET /health. Exits after --idle-min minutes without requests.
"""
import argparse
import base64
import hmac
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn.functional as F
from spatial_jev.kev import KevBuilder, build_model, make_processor, predict_batch
from torch import nn

MAX_BODY = 32 * 1024 * 1024


class PatchLinear(nn.Module):
    """A Conv3d whose kernel equals its stride, in its exact matrix-multiply form (non-overlapping patches).

    On a T4 the vision tower's bf16 Conv3d takes ≈ 21 s for 3 camera images; this form takes ≈ 5 ms with the same
    weights and output (E12e). Exposes `.weight` so the patch embedding's dtype logic is unchanged.
    """

    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        assert conv.kernel_size == conv.stride and conv.padding == (0, 0, 0) and conv.dilation == (1, 1, 1) \
            and conv.groups == 1, "only kernel == stride convolutions are matrix multiplies"
        self.weight = nn.Parameter(conv.weight.detach().reshape(conv.out_channels, -1), requires_grad=False)
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, C, T, P, P) -> (N, out, 1, 1, 1), as the conv returns
        return F.linear(x.reshape(x.shape[0], -1), self.weight, self.bias).view(x.shape[0], -1, 1, 1, 1)


def fast_patch_embed(model) -> float:
    """Swap the vision patch-embedding Conv3d for `PatchLinear`; refuse if outputs differ beyond bf16 rounding."""
    embed = model.native.visual.patch_embed
    conv = embed.proj
    probe = torch.randn(64, *conv.weight.shape[1:], device=conv.weight.device, dtype=conv.weight.dtype)
    linear = PatchLinear(conv)
    with torch.inference_mode():
        diff = (linear(probe).float() - conv(probe).float()).abs().max().item()
    if diff > 0.05:
        raise RuntimeError(f"PatchLinear differs from the Conv3d by {diff}")
    embed.proj = linear
    return diff


HALF = {"fp16": torch.float16, "bf16": torch.bfloat16}


def load(checkpoint: Path, device: str, language_dtype: str, base_dir: Path | None = None):
    """`scripts/predict.py` loading; fp16/bf16 store the trunk and vision tower in that dtype (readout stays fp32).

    `base_dir` = local snapshot of the base model at the checkpoint's `base_revision` (upstream loads `base_model` at
    whatever its `main` is; the vision tower comes from there).
    """
    config = json.loads((checkpoint / "config.json").read_text())
    if base_dir is not None:
        config["base_model"] = str(base_dir)
    config.update(preserve_option_order=True, max_image_tokens=1024,
                  max_sequence_length=max(8192, int(config.get("max_sequence_length", 4096))),
                  inference_language_dtype="fp32" if language_dtype == "fp32" else "bf16")
    builder = KevBuilder(make_processor(config), config)
    model = build_model(config, checkpoint, trainable=False)
    if language_dtype in HALF:
        lm = model.native.language_model
        lm.to(HALF[language_dtype])  # trunk = layers before the readout layer (the readout layer itself is not executed)
        lm.native_qk.float()  # readout projections + norms stay fp32
        model.native.visual.to(HALF[language_dtype])
    model = model.to(device).eval()
    print(f"[mjev] patch embedding as matrix multiply (max diff vs Conv3d {fast_patch_embed(model):.4f})", flush=True)
    return model, builder


def ask(model, builder, device: str, images: list[bytes], questions: list[dict]) -> dict:
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, png in enumerate(images):
            p = Path(d) / f"{i}.png"
            p.write_bytes(png)
            paths.append(str(p))
        media = [{"kind": "image", "uri": p} for p in paths]
        requests = [{"id": q["id"], "input": {"media": media, "question": q["question"],
                                              "answer_space": q["answer_space"]}} for q in questions]
        with torch.inference_mode():
            values = predict_batch(model, builder, requests, torch.device(device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    answers = {}
    for q, v in zip(questions, values, strict=True):
        a = {"prediction": v["prediction"], "path": v["path"]}
        if q["answer_space"]["kind"] == "choice":
            options = [o["id"] for o in q["answer_space"]["options"]]  # preserve_option_order: index i = option i
            probs = torch.tensor(v["logits"][-1][: len(options)]).softmax(-1).tolist()
            a["probabilities"] = dict(zip(options, (round(p, 4) for p in probs)))
        answers[q["id"]] = a
    return {"answers": answers, "timings_ms": {"mjev": round((time.perf_counter() - t0) * 1000, 1)}}


def serve(model, builder, device: str, token: str, port: int, idle_min: float) -> None:
    last = {"t": time.time(), "requests": 0}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        """HTTP/1.1 keep-alive (one client connection across requests) on a threaded server; `lock` serializes MJev."""

        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _authorized(self) -> bool:
            ok = hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}")
            if not ok:
                self.send_error(401)
            return ok

        def _reply(self, obj: dict):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health" and self._authorized():
                self._reply({"model": "MJev-Spatial-4B", "device": device, "requests": last["requests"]})

        def do_POST(self):  # noqa: N802
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                self.send_error(413)
                return
            body = self.rfile.read(size)  # read before rejecting: clean 401 instead of a broken pipe
            if not self._authorized():
                return
            if self.path != "/ask":
                self.send_error(404)
                return
            req = json.loads(body)
            with lock:
                last["t"], last["requests"] = time.time(), last["requests"] + 1
                out = ask(model, builder, device, [base64.b64decode(b) for b in req["images"]], req["questions"])
            print(f"[mjev] ask #{last['requests']} {[q['id'] for q in req['questions']]} {out['timings_ms']}",
                  flush=True)
            self._reply(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def watchdog():
        while time.time() - last["t"] < idle_min * 60:
            time.sleep(30)
        print(f"[mjev] idle for {idle_min} min, shutting down", flush=True)
        httpd.shutdown()

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[mjev] ready on 127.0.0.1:{port}", flush=True)
    httpd.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--language-dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--base-dir", type=Path, help="base model snapshot at the checkpoint's base_revision")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--idle-min", type=float, default=30)
    a = p.parse_args()
    if a.device.startswith("cuda"):
        torch.cuda.set_device(a.device)
    model, builder = load(a.checkpoint, a.device, a.language_dtype, a.base_dir)
    if a.device.startswith("cuda"):
        print(f"[mjev] GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    serve(model, builder, a.device, os.environ["ROJEV_TOKEN"], a.port, a.idle_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
