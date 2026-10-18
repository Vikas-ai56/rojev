"""Build and push a Kaggle kernel, or fetch its status/outputs. Mode picks the notebook:
  serve / smoke → vikask2006/rojev-pi052 (π0.5² only)      armd → vikask2006/rojev-armd (MJev GPU 1 + π0.5² GPU 0)

  python executors/robocasa-pi052/kaggle/push.py push <mode> [validate]   # push = start a GPU run
  python executors/robocasa-pi052/kaggle/push.py service <mode>           # wait for and print service URLs + tokens
  python executors/robocasa-pi052/kaggle/push.py status <mode>
  python executors/robocasa-pi052/kaggle/push.py pull <mode>              # outputs -> results/kaggle/<mode>_<time>/
`validate` (armd): first check bf16-trunk MJev against fp32 (outputs/mjev_precision.json).

The bundle (executor pyproject/uv.lock/drive.py/serve.py; services/mjev files; the rojev core with its pyproject/README;
provenance.json with commit pins, sha256 and git state) is tarred into `kaggle_run.py` as BUNDLE_B64 in a staging
dir; the repo copy of kaggle_run.py is never modified.
"""
import base64
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = HERE.parent
REPO = EXE.parents[1]
KERNELS = {"serve": "vikask2006/rojev-pi052", "smoke": "vikask2006/rojev-pi052", "armd": "vikask2006/rojev-armd"}
TIMEOUT_S = 4 * 3600
FILES = ("pyproject.toml", "uv.lock", "drive.py", "serve.py")  # executor dir
CORE = ("pyproject.toml", "README.md")  # repo root: the rojev package (path dependency of the executor env)
MJEV = REPO / "services/mjev"  # MJev-Spatial service env (own torch/transformers pins)
PINS = {"OneJev": "onejev.commit", "MJev-Spatial": "mjev.commit", "lerobot-steerable": "lerobot-steerable.commit",
        "robocasa": "robocasa.commit", "robosuite": "robosuite.commit"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def bundle() -> str:
    pins = {k: (REPO / "config" / f).read_text().strip() for k, f in PINS.items()}
    for name, sha in pins.items():
        head = subprocess.run(["git", "-C", str(REPO / "vendor" / name), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if head != sha:
            sys.exit(f"vendor/{name} is at {head}, config pins {sha}: fix before pushing")
    core = [REPO / f for f in CORE] + sorted((REPO / "rojev").rglob("*.py"))
    mjev = [MJEV / f for f in ("pyproject.toml", "uv.lock", "serve_mjev.py", "check_precision.py")]
    files = {f"executors/robocasa-pi052/{f}": EXE / f for f in FILES} | {
        str(p.relative_to(REPO)): p for p in core + mjev}
    provenance = {"pins": pins, "repo_head": git("rev-parse", "HEAD"),
                  "repo_dirty_files": git("status", "--porcelain", "--", "executors/robocasa-pi052", "rojev",
                                          "services").splitlines(),
                  "sha256": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()},
                  "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for arc, path in files.items():
            tar.add(path, arcname=arc)
        data = json.dumps(provenance, indent=2).encode()
        info = tarfile.TarInfo("provenance.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def push(mode: str, validate: bool):
    src = (HERE / "kaggle_run.py").read_text()
    assert src.count('BUNDLE_B64 = ""') == 1 and src.count('MODE = "serve"') == 1 and mode in KERNELS
    assert src.count("VALIDATE_MJEV = False") == 1
    src = src.replace('MODE = "serve"', f'MODE = "{mode}"').replace("VALIDATE_MJEV = False", f"VALIDATE_MJEV = {validate}")
    meta = json.loads((HERE / "kernel-metadata.json").read_text())
    meta.update(id=KERNELS[mode], title=KERNELS[mode].split("/")[1])
    if mode == "armd":  # MJev precision-check scenes (make_testdata.py) are data, not code: a private dataset
        meta["dataset_sources"] = meta["dataset_sources"] + ["vikask2006/rojev-mjev-testdata"]
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "kaggle_run.py").write_text(src.replace('BUNDLE_B64 = ""', f'BUNDLE_B64 = "{bundle()}"'))
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        subprocess.run(["kaggle", "kernels", "push", "-p", d, "-t", str(TIMEOUT_S)], check=True)


def service(kernel: str) -> dict:
    """Wait (through QUEUED) until the kernel prints its service URLs + tokens in the live log; returns them."""
    while True:
        status = subprocess.run(["kaggle", "kernels", "status", kernel], capture_output=True, text=True).stdout
        if re.search(r"COMPLETE|ERROR|CANCEL", status):
            sys.exit(f"kernel is not running: {status.strip()}")
        if "RUNNING" in status:
            p = subprocess.Popen(["kaggle", "kernels", "logs", "-f", kernel], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True)
            for line in p.stdout:
                if "ROJEV_SERVICE " in line:
                    p.terminate()
                    return json.loads(line.split("ROJEV_SERVICE ", 1)[1])
        time.sleep(20)


def pull(mode: str):
    out = REPO / "results/kaggle" / f"{mode}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    subprocess.run(["kaggle", "kernels", "output", KERNELS[mode], "-p", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    mode = sys.argv[2] if len(sys.argv) > 2 else "serve"
    if mode not in KERNELS:
        sys.exit(f"unknown mode {mode!r}; one of {sorted(KERNELS)}")
    if cmd == "push":
        push(mode, validate="validate" in sys.argv[3:])
    elif cmd == "service":
        print(json.dumps(service(KERNELS[mode])))
    elif cmd == "status":
        subprocess.run(["kaggle", "kernels", "status", KERNELS[mode]], check=True)
    elif cmd == "pull":
        pull(mode)
    else:
        sys.exit(__doc__)
