"""SAM 2.1 object-mask service (cloud GPU): the mask of each prompted object in each image.

Loads SAM 2.1 Hiera Tiny through transformers (weights pinned in config/sam2_weights.txt; one checkpoint serves the
image model `Sam2Model` and the streaming video model `Sam2VideoModel`).
  - segment: masks from point prompts on single images; a pure function of the request, so a retry is safe.
  - track: one streaming session per camera; objects are prompted on the first frame, then every frame sent is
    propagated with SAM 2's memory of the earlier frames (identity over time, through occlusion). Each step names its
    frame number; a repeated number (a transport retry) returns the stored reply instead of advancing the session, so
    a retry is safe. A session keeps its frames and per-frame outputs in CPU memory (only the model runs on the GPU,
    so GPU memory stays bounded however long a session runs) and lives until /track/end, or is closed after
    `SESSION_IDLE_S` without a step (a client that died never leaks it). An error answers 500 with its message.

  ROJEV_TOKEN=… uv run python serve_sam2.py --weights <snapshot dir> --device cuda:0 --port 8003
POST /segment (Bearer token), JSON:
  {"images": [<base64 PNG>, …],
   "prompts": [{"id": …, "image": <index>, "points": [[x, y], …], "labels": [1 | 0, …]}]}   pixel coordinates
→ {"masks": {id: {"mask": <base64 1-bit PNG, image size>, "score": predicted IoU}}, "timings_ms": {"sam2": …}}
POST /track/start {"images": [<one PNG per camera>], "prompts": [{"id", "image", "points", "labels"}]}
→ {"session": …, "masks": {id: {"mask", "score": P(object visible)}}, "timings_ms"}
POST /track/step {"session": …, "frame": n, "images": [<one PNG per camera, same order>]} → {"masks", "timings_ms"}
POST /track/end {"session": …} → {}
Label 1 = on the object, 0 = not on it. GET /health. Exits after --idle-min minutes without requests.
"""
import argparse
import base64
import hmac
import io
import json
import os
import threading
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

MAX_BODY = 32 * 1024 * 1024
SESSION_IDLE_S = 600
MODEL = "SAM2.1-hiera-tiny"


class Models:
    def __init__(self, weights: Path, device: str):
        self.device = device
        self.processor = Sam2Processor.from_pretrained(weights)
        self.model = Sam2Model.from_pretrained(weights).to(device).eval()
        self.video_processor = Sam2VideoProcessor.from_pretrained(weights)
        self.video_model = Sam2VideoModel.from_pretrained(weights).to(device).eval()


def mask_png(mask: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(mask).convert("1").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def by_image(prompts: list[dict]) -> dict[int, list[dict]]:
    out = defaultdict(list)
    for p in prompts:
        out[p["image"]].append(p)
    return out


def images_of(request: dict) -> list[bytes]:
    return [base64.b64decode(b) for b in request["images"]]


def decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def segment(m: Models, images: list[bytes], prompts: list[dict]) -> dict:
    """Per image, one forward pass over all of its prompts (one object each, single-mask output)."""
    t0 = time.perf_counter()
    masks = {}
    for index, mine in by_image(prompts).items():
        inputs = m.processor(images=decode(images[index]),
                             input_points=[[[list(map(float, xy)) for xy in p["points"]] for p in mine]],
                             input_labels=[[list(map(int, p["labels"])) for p in mine]],
                             return_tensors="pt").to(m.device)
        with torch.inference_mode():
            out = m.model(**inputs, multimask_output=False)
        binary = m.processor.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        scores = out.iou_scores[0, :, 0].float().cpu().tolist()
        for k, p in enumerate(mine):
            masks[p["id"]] = {"mask": mask_png(binary[k, 0].numpy().astype(bool)), "score": round(scores[k], 4)}
    if m.device.startswith("cuda"):
        torch.cuda.synchronize()
    return {"masks": masks, "timings_ms": {"sam2": round((time.perf_counter() - t0) * 1000, 1)}}


class Tracks:
    """Streaming sessions: per session and camera, a SAM 2 video inference session and its prompt ids."""

    def __init__(self, m: Models):
        self.m = m
        self.sessions: dict[str, list[tuple[object, list[str]]]] = {}
        self.replies: dict[str, tuple[int, dict]] = {}
        self.touched: dict[str, float] = {}

    def close_idle(self, now: float) -> None:
        for session in [s for s, t in self.touched.items() if now - t > SESSION_IDLE_S]:
            self.end(session)

    def _propagate(self, cameras: list[tuple[object, list[str]]], images: list[bytes], first: dict | None) -> dict:
        t0 = time.perf_counter()
        masks = {}
        for index, (session, ids) in enumerate(cameras):
            if not ids:
                continue
            inputs = self.m.video_processor(images=decode(images[index]), device=self.m.device, return_tensors="pt")
            if first is not None:
                mine = first[index]
                self.m.video_processor.add_inputs_to_inference_session(
                    inference_session=session, frame_idx=0, obj_ids=list(range(1, len(mine) + 1)),
                    input_points=[[[list(map(float, xy)) for xy in p["points"]] for p in mine]],
                    input_labels=[[list(map(int, p["labels"])) for p in mine]],
                    original_size=inputs.original_sizes[0])
            with torch.inference_mode():
                out = self.m.video_model(inference_session=session, frame=inputs.pixel_values[0])
            binary = self.m.video_processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes,
                                                               binarize=True)[0]
            visible = torch.sigmoid(out.object_score_logits.float()).flatten().cpu().tolist()
            for k, key in enumerate(ids):
                masks[key] = {"mask": mask_png(binary[k, 0].cpu().numpy().astype(bool)), "score": round(visible[k], 4)}
        if self.m.device.startswith("cuda"):
            torch.cuda.synchronize()
        return {"masks": masks, "timings_ms": {"sam2": round((time.perf_counter() - t0) * 1000, 1)}}

    def start(self, images: list[bytes], prompts: list[dict]) -> dict:
        grouped = by_image(prompts)
        cameras = [(self.m.video_processor.init_video_session(inference_device=self.m.device,
                                                              inference_state_device="cpu",
                                                              video_storage_device="cpu"),
                    [p["id"] for p in grouped.get(i, [])]) for i in range(len(images))]
        out = self._propagate(cameras, images, grouped)
        session = uuid.uuid4().hex
        self.sessions[session] = cameras
        self.replies[session] = (0, out)
        self.touched[session] = time.time()
        return {"session": session, **out}

    def step(self, session: str, frame: int, images: list[bytes]) -> dict:
        last, reply = self.replies[session]
        if frame == last:
            return reply
        out = self._propagate(self.sessions[session], images, None)
        self.replies[session] = (frame, out)
        self.touched[session] = time.time()
        return out

    def end(self, session: str) -> dict:
        self.sessions.pop(session, None)
        self.replies.pop(session, None)
        self.touched.pop(session, None)
        return {}


def serve(m: Models, token: str, port: int, idle_min: float) -> None:
    last = {"t": time.time(), "requests": 0}
    lock = threading.Lock()
    tracks = Tracks(m)
    routes = {"/segment": lambda r: segment(m, images_of(r), r["prompts"]),
              "/track/start": lambda r: tracks.start(images_of(r), r["prompts"]),
              "/track/step": lambda r: tracks.step(r["session"], r["frame"], images_of(r)),
              "/track/end": lambda r: tracks.end(r["session"])}

    class Handler(BaseHTTPRequestHandler):
        """HTTP/1.1 keep-alive on a threaded server; `lock` serializes the model."""

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

        def do_GET(self):
            if self.path == "/health" and self._authorized():
                self._reply({"model": MODEL, "device": m.device, "requests": last["requests"],
                             "sessions": len(tracks.sessions)})

        def do_POST(self):
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                self.send_error(413)
                return
            body = self.rfile.read(size)
            if not self._authorized():
                return
            if self.path not in routes:
                self.send_error(404)
                return
            req = json.loads(body)
            with lock:
                last["t"], last["requests"] = time.time(), last["requests"] + 1
                tracks.close_idle(last["t"])
                try:
                    out = routes[self.path](req)
                except KeyError as e:
                    self.send_error(404, f"unknown session or field: {e}")
                    return
                except Exception as e:
                    if m.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    print(f"[sam2] {self.path} failed: {type(e).__name__}: {e}", flush=True)
                    self.send_error(500, f"{type(e).__name__}: {e}"[:200])
                    return
            if self.path != "/track/step":
                print(f"[sam2] {self.path} #{last['requests']} {out.get('timings_ms', {})}", flush=True)
            self._reply(out)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    def watchdog():
        while time.time() - last["t"] < idle_min * 60:
            time.sleep(30)
        print(f"[sam2] idle for {idle_min} min, shutting down", flush=True)
        httpd.shutdown()

    threading.Thread(target=watchdog, daemon=True).start()
    print(f"[sam2] ready on 127.0.0.1:{port}", flush=True)
    httpd.serve_forever()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8003)
    p.add_argument("--idle-min", type=float, default=30)
    a = p.parse_args()
    m = Models(a.weights, a.device)
    if a.device.startswith("cuda"):
        print(f"[sam2] GPU memory after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
    serve(m, os.environ["ROJEV_TOKEN"], a.port, a.idle_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
