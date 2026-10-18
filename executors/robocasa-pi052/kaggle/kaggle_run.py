"""Kaggle kernel (2×T4) for the cloud models. Modes (MODE, set by `push.py push <mode>`):
  serve  π0.5² (`serve.py`) on GPU 0 behind a Cloudflare quick tunnel (notebook rojev-pi052).
  armd   arm D services (notebook rojev-armd): MJev-Spatial (`services/mjev/serve_mjev.py`) on GPU 1 and π0.5² on
         GPU 0, one tunnel each; with VALIDATE_MJEV, first checks bf16-trunk MJev against fp32 (CPU).
  smoke  previous setup check: also fetch assets + render, run `drive.py` RUNS in the cloud sim.
Service modes print one line `ROJEV_SERVICE {"pi052": {"url", "token"}, "mjev": {…}}` (also outputs/service.json)
for the Mac runner and exit once every service has been idle for IDLE_MIN minutes. No sim and no RoboCasa assets
in service modes (the sim runs on the Mac).

Pushed by `push.py`, which embeds the executor files (pyproject.toml, uv.lock, drive.py, serve.py), the rojev core
and commit pins as BUNDLE_B64, so the kernel runs exactly the committed lockfile. The gated PaliGemma tokenizer comes from the private
dataset vikask2006/rojev-paligemma-tokenizer and is passed to drive.py as a local directory, so no token is needed
(CLI-pushed runs cannot read Kaggle secrets); public repos download normally. Everything heavy lives in /kaggle/tmp; results go to /kaggle/working/outputs (downloaded by `pull`).
Each stage is timed and written to outputs/setup.json; any failure stops the kernel with the failing stage named.
"""
import base64
import io
import json
import os
import re
import secrets
import shlex
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

BUNDLE_B64 = ""  # filled by push.py
MODE = "serve"  # set by push.py
VALIDATE_MJEV = False  # set by push.py (`push armd validate`)
IDLE_MIN = 30  # serve: shut down after this long without requests (GPU quota)
CLOUDFLARED = ("https://github.com/cloudflare/cloudflared/releases/download/2026.10.1/cloudflared-linux-amd64")  # pinned:
# the version that served the 2026-10-18 E10 runs
# Setup check only: one 50-step action chunk through the subtask hook OneJev will use ("direct"). Proves load,
# inference, rendering and stepping work; says nothing about task success (π0.5² is evaluated in the workflow).
RUNS = [  # drive.py invocations; each writes outputs/<name>/
    {"name": "smoke_one_chunk", "args": ["--env-task", "CloseFridge", "--instruction", "close the refrigerator door",
                                         "--planner", "direct", "--max-steps", "50"]},
]
ASSET_TYPES = ["tex", "tex_generative", "fixtures_lw", "objs_objaverse", "objs_lw"]  # registries: objaverse,lightwheel

TMP = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")
ROOT = TMP / "rojev"
EXE = ROOT / "executors/robocasa-pi052"
MJEV = ROOT / "services/mjev"
OUT = Path("/kaggle/working/outputs")
MJEV_CKPT = ("mindlab-research/MJev-Spatial-4B", "0657319c58d342f54c5108cd3c4d983a79a8b3f7")  # HF repo, revision
REPOS = {"OneJev": "https://github.com/OmniJev/OneJev.git",  # qev client, a dependency of the rojev core
         "MJev-Spatial": "https://github.com/Fr0zenCrane/MJev-Spatial.git",
         "lerobot-steerable": "https://github.com/huggingface/lerobot.git",
         "robocasa": "https://github.com/robocasa/robocasa.git",
         "robosuite": "https://github.com/ARISE-Initiative/robosuite.git"}
ENV = {**os.environ, "UV_CACHE_DIR": str(TMP / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(TMP / "uv-python"),
       "HF_HOME": str(TMP / "hf"), "PYTHONUNBUFFERED": "1"}
report: dict = {"stages": {}}
TOKENIZER: dict = {}


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "setup.json").write_text(json.dumps(report, indent=2))


def sh(cmd: str, cwd: Path | None = None, stdin: str | None = None, check: bool = True,
       env: dict | None = None) -> subprocess.CompletedProcess:
    """Run `cmd`, streaming its output live; returns the full output in `.stdout`."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env or ENV, text=True, stdin=subprocess.PIPE if stdin else None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    if stdin:
        p.stdin.write(stdin)
        p.stdin.close()
    lines = []
    for line in p.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    r = subprocess.CompletedProcess(cmd, p.wait(), "".join(lines))
    if check and r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {cmd}")
    return r


def stage(name: str):
    def wrap(fn):
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        try:
            result = fn()
            report["stages"][name] = {"ok": True, "s": round(time.time() - t0, 1), **(result or {})}
        except Exception as e:  # noqa: BLE001 — record which stage failed, then stop the kernel
            report["stages"][name] = {"ok": False, "s": round(time.time() - t0, 1), "error": repr(e)}
            save()
            raise
        save()
        return fn
    return wrap


def main():
    @stage("tokenizer")
    def _():
        found = sorted(Path("/kaggle/input").rglob("tokenizer.json"))
        if not found:
            raise RuntimeError("dataset vikask2006/rojev-paligemma-tokenizer not attached")
        TOKENIZER["dir"] = str(found[0].parent)  # passed to drive.py --tokenizer-dir; no Hub access to the gated repo
        return {"dir": TOKENIZER["dir"], "revision": (found[0].parent / "REVISION").read_text().strip()}

    @stage("gpu")
    def _():
        return {"nvidia_smi": sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv").stdout}

    @stage("bundle")
    def _():
        EXE.mkdir(parents=True, exist_ok=True)
        tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE_B64)), mode="r:gz").extractall(ROOT, filter="data")
        return {"provenance": json.loads((ROOT / "provenance.json").read_text())}

    @stage("vendor_clone")
    def _():
        pins = json.loads((ROOT / "provenance.json").read_text())["pins"]
        for name, url in REPOS.items():
            d = ROOT / "vendor" / name
            d.mkdir(parents=True, exist_ok=True)
            sh(f"git init -q && git remote add origin {url} && git fetch -q --depth 1 origin {pins[name]} "
               f"&& git checkout -q FETCH_HEAD", cwd=d)
        return {"pins": pins}

    @stage("uv_sync")
    def _():
        sh("pip install -q uv")
        sh("uv sync --frozen", cwd=EXE)
        if MODE == "armd":
            sh("uv sync --frozen", cwd=MJEV)
        r = sh("uv run python -c \"import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(),"
               " torch.cuda.device_count())\"", cwd=EXE)
        if "True" not in r.stdout:
            raise RuntimeError("torch cannot see the GPU")
        return {"torch": r.stdout.strip().splitlines()[-1]}

    if MODE == "serve":
        model_stages()
        return services_stages([pi052_service()])
    if MODE == "armd":
        model_stages()
        return services_stages([mjev_stages(), pi052_service()])

    @stage("assets")
    def _():
        sh("uv run python -m robocasa.scripts.setup_macros", cwd=EXE, stdin="y\n", check=False)
        sh(f"uv run python -m robocasa.scripts.download_kitchen_assets --type {' '.join(ASSET_TYPES)}", cwd=EXE,
           stdin="y\n")
        return {"du": sh(f"du -sh {ROOT}/vendor/robocasa/robocasa/models/assets", cwd=EXE).stdout.strip()}

    @stage("render")
    def _():
        probe = "uv run python -c \"import mujoco;c=mujoco.GLContext(64,64);c.make_current();print('ok')\""
        ENV["MUJOCO_GL"] = "egl"
        if "ok" in sh(probe, cwd=EXE, check=False).stdout:
            return {"MUJOCO_GL": "egl"}
        sh("apt-get install -y -q libosmesa6 >/dev/null", check=False)
        ENV.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa")
        if "ok" not in sh(probe, cwd=EXE, check=False).stdout:
            raise RuntimeError("neither EGL nor OSMesa rendering works")
        return {"MUJOCO_GL": "osmesa"}

    model_stages()

    for run in RUNS:
        @stage(f"run:{run['name']}")
        def _(run=run):
            out = OUT / run["name"]
            sh(f"uv run python drive.py {shlex.join(run['args'])} --tokenizer-dir {TOKENIZER['dir']} --out {out}", cwd=EXE)
            return {"episode": json.loads((out / "episode.json").read_text())}


def model_stages():
    @stage("bf16_probe")
    def _():
        code = ("import torch;a=torch.randn(512,512,device='cuda');b=torch.randn(512,512,device='cuda');"
                "r=(a.bfloat16()@b.bfloat16()).float();print('rel_err',((r-a@b).norm()/(a@b).norm()).item())")
        r = sh(f"uv run python -c \"{code}\"", cwd=EXE)
        return {"result": r.stdout.strip().splitlines()[-1]}

    @stage("model_download")
    def _():
        # Prefetch (timed separately): public repos the checkpoint loads; PaliGemma comes from the dataset.
        for repo in ("lerobot/pi052_robocasa", "lerobot/fast-action-tokenizer"):
            sh(f"uv run hf download {repo}", cwd=EXE)


def background(cmd: list[str], env: dict, cwd: Path | None = None) -> tuple[subprocess.Popen, list[str]]:
    """Start `cmd`, echo its output live from a thread; returns the process and the growing list of lines."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    lines: list[str] = []

    def pump():
        for line in p.stdout:
            print(line, end="", flush=True)
            lines.append(line)

    threading.Thread(target=pump, daemon=True).start()
    return p, lines


def wait_for(lines: list[str], proc: subprocess.Popen, pattern: str, timeout_s: float) -> re.Match:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for line in list(lines):
            if m := re.search(pattern, line):
                return m
        if proc.poll() is not None:
            raise RuntimeError(f"process exited ({proc.returncode}) before {pattern!r}")
        time.sleep(1)
    raise RuntimeError(f"timeout waiting for {pattern!r}")


def pi052_service() -> dict:
    return {"name": "pi052", "cwd": EXE, "port": 8000, "gpu": "0", "ready": r"\[serve\] ready",
            "cmd": ["uv", "run", "python", "serve.py", "--tokenizer-dir", TOKENIZER["dir"], "--port", "8000",
                    "--idle-min", str(IDLE_MIN)]}


def mjev_stages() -> dict:
    """Download MJev + its base model at the pinned revisions, optionally validate bf16 vs fp32; return its service."""
    paths: dict = {}

    @stage("mjev_download")
    def _():
        repo, rev = MJEV_CKPT
        sh(f"uv run hf download {repo} --revision {rev}", cwd=MJEV)
        ckpt = Path(ENV["HF_HOME"]) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
        config = json.loads((ckpt / "config.json").read_text())
        base, base_rev = config["base_model"], config["base_revision"]  # vision tower comes from the base model
        sh(f"uv run hf download {base} --revision {base_rev}", cwd=MJEV)
        paths["ckpt"] = ckpt
        paths["base"] = Path(ENV["HF_HOME"]) / "hub" / f"models--{base.replace('/', '--')}" / "snapshots" / base_rev
        return {"checkpoint": f"{repo}@{rev}", "base": f"{base}@{base_rev}"}

    if VALIDATE_MJEV:
        @stage("mjev_precision")
        def _():
            out = OUT / "mjev_precision.json"
            found = sorted(Path("/kaggle/input").rglob("meta.json"))  # dataset vikask2006/rojev-mjev-testdata
            if not found:
                raise RuntimeError("dataset vikask2006/rojev-mjev-testdata not attached")
            testdata = found[0].parent.parent
            sh(f"uv run python check_precision.py --checkpoint {paths['ckpt']} --base-dir {paths['base']} "
               f"--testdata {testdata} --device cuda:0 --out {out}", cwd=MJEV, env={**ENV, "CUDA_VISIBLE_DEVICES": "1"})
            r = json.loads(out.read_text())
            return {"top1_agreement": r["top1_agreement"], "max_abs_prob_diff": r["max_abs_prob_diff"]}

    return {"name": "mjev", "cwd": MJEV, "port": 8001, "gpu": "1", "ready": r"\[mjev\] ready",
            "cmd": ["uv", "run", "python", "serve_mjev.py", "--checkpoint", str(paths["ckpt"]), "--base-dir",
                    str(paths["base"]), "--device", "cuda:0", "--port", "8001", "--idle-min", str(IDLE_MIN)]}


def open_tunnel(binary: Path, port: int, token: str) -> tuple:
    proc = background([str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"], ENV)
    url = wait_for(proc[1], proc[0], r"https://[a-z0-9-]+\.trycloudflare\.com", 120)[0]
    for _ in range(30):  # the public hostname takes a few seconds to propagate
        try:
            req = urllib.request.Request(url + "/health", headers={"Authorization": f"Bearer {token}"})
            return proc, url, urllib.request.urlopen(req, timeout=10).read().decode()
        except Exception as e:  # noqa: BLE001
            health = repr(e)
            time.sleep(5)
    raise RuntimeError(f"tunnel to :{port} not reachable: {health}")


def services_stages(services: list[dict]):
    """Start each service (in order, one GPU each), one tunnel each; print their addresses; wait until all idle out."""
    procs: dict = {}
    tokens = {s["name"]: secrets.token_urlsafe(24) for s in services}
    for svc in services:
        @stage(f"service_start:{svc['name']}")
        def _(svc=svc):
            env = {**ENV, "ROJEV_TOKEN": tokens[svc["name"]], "CUDA_VISIBLE_DEVICES": svc["gpu"]}
            procs[svc["name"]] = background(svc["cmd"], env, svc["cwd"])
            wait_for(procs[svc["name"]][1], procs[svc["name"]][0], svc["ready"], 1500)
            mem = sh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader").stdout.strip()
            return {"gpu": svc["gpu"], "nvidia_smi_memory": mem}

    addresses: dict = {}

    @stage("tunnels")
    def _():
        binary = TMP / "cloudflared"
        sh(f"curl -sSL -o {binary} {CLOUDFLARED} && chmod +x {binary}")
        result = {"cloudflared": sh(f"{binary} --version").stdout.strip()}
        for svc in services:
            proc, url, health = open_tunnel(binary, svc["port"], tokens[svc["name"]])
            procs[f"tunnel:{svc['name']}"] = proc
            addresses[svc["name"]] = {"url": url, "token": tokens[svc["name"]]}
            result[svc["name"]] = {"url": url, "health_via_tunnel": health}
        (OUT / "service.json").write_text(json.dumps(addresses))
        print("ROJEV_SERVICE " + json.dumps(addresses), flush=True)
        return result

    @stage("serving")
    def _():
        codes = {s["name"]: procs[s["name"]][0].wait() for s in services}  # each exits after IDLE_MIN idle
        for svc in services:
            procs[f"tunnel:{svc['name']}"][0].terminate()
        counts = {s["name"]: sum(" act " in l or " ask " in l for l in procs[s["name"]][1]) for s in services}
        return {"exit": codes, "requests": counts}

if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        report["total_s"] = round(time.time() - t0, 1)
        save()
        print(json.dumps(report, indent=2, default=str)[-5000:])
    sys.exit(0)
