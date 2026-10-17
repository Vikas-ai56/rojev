"""Build and push the Kaggle kernel `vikask2006/rojev-pi052`, or fetch its status/outputs.

  python executors/robocasa-pi052/kaggle/push.py push      # embed bundle, push = start a GPU run
  python executors/robocasa-pi052/kaggle/push.py status
  python executors/robocasa-pi052/kaggle/push.py pull      # outputs -> results/kaggle/<time>/

The bundle (pyproject.toml, uv.lock, drive.py, provenance.json with commit pins + git state) is tarred into
`kaggle_run.py` as BUNDLE_B64 in a staging dir; the repo copy of kaggle_run.py is never modified.
"""
import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXE = HERE.parent
REPO = EXE.parents[1]
KERNEL = "vikask2006/rojev-pi052"
TIMEOUT_S = 4 * 3600
FILES = ("pyproject.toml", "uv.lock", "drive.py")
PINS = {"lerobot-steerable": "lerobot-steerable.commit", "robocasa": "robocasa.commit", "robosuite": "robosuite.commit"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def bundle() -> str:
    pins = {k: (REPO / "config" / f).read_text().strip() for k, f in PINS.items()}
    for name, sha in pins.items():
        head = subprocess.run(["git", "-C", str(REPO / "vendor" / name), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        if head != sha:
            sys.exit(f"vendor/{name} is at {head}, config pins {sha}: fix before pushing")
    provenance = {"pins": pins, "repo_head": git("rev-parse", "HEAD"),
                  "repo_dirty_files": git("status", "--porcelain", "--", "executors/robocasa-pi052").splitlines(),
                  "sha256": {f: hashlib.sha256((EXE / f).read_bytes()).hexdigest() for f in FILES},
                  "built": time.strftime("%Y-%m-%dT%H:%M:%S")}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            tar.add(EXE / f, arcname=f"executors/robocasa-pi052/{f}")
        data = json.dumps(provenance, indent=2).encode()
        info = tarfile.TarInfo("provenance.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def push():
    src = (HERE / "kaggle_run.py").read_text()
    assert src.count('BUNDLE_B64 = ""') == 1
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "kaggle_run.py").write_text(src.replace('BUNDLE_B64 = ""', f'BUNDLE_B64 = "{bundle()}"'))
        (Path(d) / "kernel-metadata.json").write_text((HERE / "kernel-metadata.json").read_text())
        subprocess.run(["kaggle", "kernels", "push", "-p", d, "-t", str(TIMEOUT_S)], check=True)


def pull():
    out = REPO / "results/kaggle" / f"pi052_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    subprocess.run(["kaggle", "kernels", "output", KERNEL, "-p", str(out)], check=True)
    print(out)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "push":
        push()
    elif cmd == "status":
        subprocess.run(["kaggle", "kernels", "status", KERNEL], check=True)
    elif cmd == "pull":
        pull()
    else:
        sys.exit(__doc__)
