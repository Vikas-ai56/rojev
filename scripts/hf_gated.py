"""Gated Hugging Face repos for the GPU setup hooks; run with the calling component's own uv env (huggingface_hub).

  python scripts/hf_gated.py login [--asset]                              Mac: this Mac's Hugging Face login file
                                                                           (--asset: as the hook's asset line)
  python scripts/hf_gated.py snapshot --pin-file config/<x>.txt [--check]  box: download the pinned revision with
                                                                           the login and point refs/main at it;
                                                                           --check: complete and still the Hub's main

A gated repo is downloaded on the box itself with the owner's Hugging Face login (decision 2026-11-07: uploading
gigabytes from the Mac dominated setup time). `login --asset` names the Mac's token file (huggingface_hub's
HF_TOKEN_PATH) as an asset that the hook copies to the box's `~/.cache/huggingface/` (the library's default path
there), and refuses unless the file has mode 600, which the copy keeps. This script never reads, prints or passes the
token: the hub library finds the file by itself; only the path is printed.

`snapshot` downloads every file of the revision pinned in the pin file (one line `repo <id> @ <40-hex revision>`, as
for scripts/hf_pins.py) and then writes the cache's `refs/main` = that revision (a download by commit writes no ref),
so a model that loads the repo by id finds the pinned files. Such models run online (transformers asks the Hub for
the repo while loading even with HF_HUB_OFFLINE=1; found on the box 2026-11-07), and online a repo id resolves to the
Hub's current main: `--check` therefore also requires that main still equals the pin, so a moved repo stops the run
instead of loading an unpinned revision. `--check` prints what is wrong and exits 1, which is how hooks decide.
"""
import argparse
import os
import stat
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE, HF_TOKEN_PATH

from hf_pins import read_pin

BOX_LOGIN_DIR = ".cache/huggingface"


def repo_folder(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def login_file(path: Path = Path(HF_TOKEN_PATH)) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"no Hugging Face login at {path}: run `hf auth login` on this Mac")
    mode = stat.S_IMODE(os.stat(path).st_mode)
    if mode != 0o600:
        raise PermissionError(f"{path} has mode {oct(mode)}; set it to 600 before it is copied anywhere")
    return path


def missing_files(cache: Path, repo: str, revision: str, files: list[str]) -> list[str]:
    snapshot = cache / repo_folder(repo) / "snapshots" / revision
    return [name for name in files if not (snapshot / name).is_file()]


def ref_points_at(cache: Path, repo: str, revision: str) -> bool:
    ref = cache / repo_folder(repo) / "refs/main"
    return ref.is_file() and ref.read_text().strip() == revision


def write_ref(cache: Path, repo: str, revision: str, files: list[str]) -> None:
    missing = missing_files(cache, repo, revision, files)
    if missing:
        raise FileNotFoundError(f"{repo} @ {revision}: files missing in the cache: {missing}")
    ref = cache / repo_folder(repo) / "refs/main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(revision)


def cache_problem(cache: Path, pin_file: Path, repo: str, revision: str) -> str | None:
    if ref_points_at(cache, repo, revision):
        return None
    return (f"pinned {repo} @ {revision} is not in {cache} with refs/main on it: run "
            f"`scripts/hf_gated.py snapshot --pin-file {pin_file}`")


def upstream_problem(cache: Path, pin_file: Path, repo: str, revision: str, upstream: str,
                     upstream_files: list[str]) -> str | None:
    if upstream != revision:
        return (f"{repo} main is now {upstream}, pinned {revision}: it is loaded by repo id, so review the change and "
                f"re-pin {pin_file}")
    missing = missing_files(cache, repo, revision, upstream_files)
    if missing:
        return f"pinned {repo} @ {revision} is incomplete in {cache} (missing {missing}): download it again"
    return None


def file_names(repo: str, revision: str | None = None) -> tuple[str, list[str]]:
    info = HfApi().model_info(repo, revision=revision)
    return info.sha, [sibling.rfilename for sibling in info.siblings]


def download(cache: Path, repo: str, revision: str) -> None:
    _, files = file_names(repo, revision)
    snapshot_download(repo, revision=revision, cache_dir=cache)
    write_ref(cache, repo, revision, files)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["login", "snapshot"])
    p.add_argument("--asset", action="store_true", help="login: print `<token file>\\t<box dir>` for the hook's assets")
    p.add_argument("--pin-file", type=Path, help="snapshot: config/<x>.txt with `repo <id> @ <revision>`")
    p.add_argument("--check", action="store_true", help="snapshot: exit 1 unless cached, referenced and still main")
    a = p.parse_args()
    if a.what == "login":
        try:
            path = login_file()
        except (FileNotFoundError, PermissionError) as e:
            print(e, file=sys.stderr)
            return 1
        print(f"{path}\t{BOX_LOGIN_DIR}" if a.asset else path)
        return 0
    if a.pin_file is None:
        p.error("snapshot needs --pin-file")
    cache, (repo, revision) = Path(HF_HUB_CACHE), read_pin(a.pin_file)
    if a.check:
        problem = (cache_problem(cache, a.pin_file, repo, revision)
                   or upstream_problem(cache, a.pin_file, repo, revision, *file_names(repo)))
        if problem:
            print(problem, file=sys.stderr)
        return 1 if problem else 0
    download(cache, repo, revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
