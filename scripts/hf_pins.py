"""Pinned Hugging Face snapshots for the GPU setup hooks; run with the calling component's own uv env.

  python scripts/hf_pins.py snapshot REPO@REVISION [--offline]
  python scripts/hf_pins.py snapshot --pin-file config/<name>_weights.txt [--offline]
  python scripts/hf_pins.py repo --pin-file config/<name>_weights.txt

`snapshot` downloads (or, with --offline, only looks up in the local cache) exactly that revision and prints the local
snapshot directory; a miss with --offline exits 1, which is how a hook's check_remote decides "done". `repo` prints
the pinned repo id. A pin file holds one line `repo <hf id> @ <revision>`; its other lines are notes for people.
Only public repos are fetched this way: no token is sent, even when the machine has a Hugging Face login (an expired
login would otherwise turn a public download into a 401).
"""
import argparse
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

PIN_LINE = re.compile(r"^repo\s+(\S+)\s+@\s+([0-9a-f]{40})\s*$")


def read_pin(pin_file: Path) -> tuple[str, str]:
    for line in pin_file.read_text().splitlines():
        if m := PIN_LINE.match(line):
            return m[1], m[2]
    raise SystemExit(f"{pin_file}: no line `repo <id> @ <40-hex revision>`")


def parse_ref(ref: str) -> tuple[str, str]:
    repo, sep, revision = ref.partition("@")
    if not sep or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"{ref!r}: expected REPO@<40-hex revision>")
    return repo, revision


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["snapshot", "repo"])
    p.add_argument("ref", nargs="?", help="REPO@REVISION (instead of --pin-file)")
    p.add_argument("--pin-file", type=Path)
    p.add_argument("--offline", action="store_true", help="local cache only; exit 1 if missing")
    a = p.parse_args()
    if (a.ref is None) == (a.pin_file is None):
        p.error("give exactly one of REPO@REVISION or --pin-file")
    repo, revision = read_pin(a.pin_file) if a.pin_file else parse_ref(a.ref)
    if a.what == "repo":
        print(repo)
        return 0
    try:
        print(snapshot_download(repo, revision=revision, local_files_only=a.offline, token=False))
    except LocalEntryNotFoundError:
        print(f"{repo}@{revision} is not in the local Hugging Face cache", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
