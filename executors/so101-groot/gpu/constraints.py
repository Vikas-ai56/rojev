"""Rewrites this executor's `[tool.uv] constraint-dependencies` from Isaac-GR00T's own `uv.lock` at the pinned commit,
so the GR00T env runs NVIDIA's tested versions and only rojev's few dependencies sit on top.

Isaac-GR00T's `pyproject.toml` pins only some packages; resolving it fresh drifted from its tested lock on 72 of 139
shared packages (2026-11-07), among them pyarrow 26 (needs NumPy 2, against GR00T's NumPy 1.26.4: every parquet
read failed) and TensorRT 11 for 10. The lock is read with `git show <pin>:uv.lock`, so the constraints always match
`config/isaac-groot.commit`; local version labels (`+cu128`) are dropped, which the CUDA index then supplies. Run
after changing the pin, then `uv lock`:

  uv run --frozen --no-sync python -m gpu.constraints <Isaac-GR00T checkout>
"""
import argparse
import re
import subprocess
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PIN = HERE.parents[1] / "config/isaac-groot.commit"
PYPROJECT = HERE / "pyproject.toml"
BLOCK = re.compile(r"constraint-dependencies = \[\n(?:    \".*\",\n)*\]\n")
ANCHOR = "override-dependencies = "


def tested_versions(checkout: Path, pin: str) -> list[str]:
    lock = subprocess.run(["git", "-C", str(checkout), "show", f"{pin}:uv.lock"], check=True, capture_output=True,
                          text=True).stdout
    versions = {}
    for package in tomllib.loads(lock).get("package", []):
        source = package.get("source", {})
        if "registry" in source or "url" in source:
            versions[package["name"]] = re.sub(r"\+.*$", "", package["version"])
    return [f"{name}=={version}" for name, version in sorted(versions.items())]


def rewrite(pyproject: str, constraints: list[str]) -> str:
    block = "constraint-dependencies = [\n" + "".join(f'    "{c}",\n' for c in constraints) + "]\n"
    if BLOCK.search(pyproject):
        return BLOCK.sub(lambda _: block, pyproject)
    line_end = pyproject.index("\n", pyproject.index(ANCHOR)) + 1
    return pyproject[:line_end] + block + pyproject[line_end:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkout", type=Path, help="an Isaac-GR00T git checkout containing the pinned commit")
    a = p.parse_args()
    pin = PIN.read_text().strip()
    constraints = tested_versions(a.checkout, pin)
    PYPROJECT.write_text(rewrite(PYPROJECT.read_text(), constraints))
    print(f"{len(constraints)} constraints from Isaac-GR00T {pin[:8]} uv.lock -> {PYPROJECT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
