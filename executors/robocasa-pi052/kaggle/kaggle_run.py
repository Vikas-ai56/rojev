"""Kaggle kernel (2×T4): install the π0.5² executor env, fetch assets + model, check the GPU, run `drive.py`.

Pushed by `push.py`, which embeds this repo's executor files (pyproject.toml, uv.lock, drive.py, commit pins) as
BUNDLE_B64, so the kernel runs exactly the committed lockfile. The gated PaliGemma tokenizer comes from the private
dataset vikask2006/rojev-paligemma-tokenizer and is passed to drive.py as a local directory, so no token is needed
(CLI-pushed runs cannot read Kaggle secrets); public repos download normally. Everything heavy lives in /kaggle/tmp; results go to /kaggle/working/outputs (downloaded by `pull`).
Each stage is timed and written to outputs/setup.json; any failure stops the kernel with the failing stage named.
"""
import base64
import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

BUNDLE_B64 = ""  # filled by push.py
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
OUT = Path("/kaggle/working/outputs")
REPOS = {"lerobot-steerable": "https://github.com/huggingface/lerobot.git",
         "robocasa": "https://github.com/robocasa/robocasa.git",
         "robosuite": "https://github.com/ARISE-Initiative/robosuite.git"}
ENV = {**os.environ, "UV_CACHE_DIR": str(TMP / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(TMP / "uv-python"),
       "HF_HOME": str(TMP / "hf"), "PYTHONUNBUFFERED": "1"}
report: dict = {"stages": {}}
TOKENIZER: dict = {}


def save():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "setup.json").write_text(json.dumps(report, indent=2))


def sh(cmd: str, cwd: Path | None = None, stdin: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run `cmd`, streaming its output live; returns the full output in `.stdout`."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, cwd=cwd, env=ENV, text=True, stdin=subprocess.PIPE if stdin else None,
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
        r = sh("uv run python -c \"import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(),"
               " torch.cuda.device_count())\"", cwd=EXE)
        if "True" not in r.stdout:
            raise RuntimeError("torch cannot see the GPU")
        return {"torch": r.stdout.strip().splitlines()[-1]}

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

    for run in RUNS:
        @stage(f"run:{run['name']}")
        def _(run=run):
            out = OUT / run["name"]
            sh(f"uv run python drive.py {shlex.join(run['args'])} --tokenizer-dir {TOKENIZER['dir']} --out {out}", cwd=EXE)
            return {"episode": json.loads((out / "episode.json").read_text())}


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        report["total_s"] = round(time.time() - t0, 1)
        save()
        print(json.dumps(report, indent=2, default=str)[-5000:])
    sys.exit(0)
