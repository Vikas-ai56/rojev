"""The box's LIBERO installation, behind this executor's GPU hook (`gpu/hook.sh`); run from the executor dir in its env.

  .venv/bin/python -m gpu.box_setup prepare          download the pinned LIBERO assets, write LIBERO's config file,
                                                     link LIBERO's asset cache to the assets
  .venv/bin/python -m gpu.box_setup check            exit 0 when all of that is in place (offline; prints what is not)
  .venv/bin/python -m gpu.box_setup render [--save DIR]
                                                     build a libero_goal task 0 backend (LeRobot's reset protocol),
                                                     step once, print each camera's frame statistics (PNGs into DIR);
                                                     exit 1 on a blank frame. Needs MUJOCO_GL (egl on a GPU box).

Follows LeRobot's LIBERO image (`docker/Dockerfile.sim`, libero_deps stage, lerobot main): assets = the dataset
`lerobot/libero-assets` at revision 0b3ea86b…; `config.yaml` names assets, bddl_files and init_states (the package's
`bddl_files` and `init_files`) and datasets; hf-libero 0.1.4's asset lookup (`libero/libero/__init__.py`,
`get_assets_path`) ignores the config's `assets` and takes ~/.cache/libero/assets when its four object folders exist
there, so that path links to the assets. One difference: the assets are the Hugging Face cache's snapshot of the pinned
revision (found offline by revision, which a plain download folder cannot prove) instead of a copy in ~/.libero/assets.
The config file lives in $LIBERO_CONFIG_PATH (LIBERO's own variable; default ~/.libero). LIBERO asks for input on its
first import when the file is missing, so `prepare` and `check` find the package without importing it. An existing
~/.cache/libero/assets that is a directory (an unpinned download by LIBERO itself) is reported, never deleted.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from PIL import Image

from embodiment import START

ASSETS_REPO = "lerobot/libero-assets"
ASSETS_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
ASSET_FOLDERS = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects", "stable_hope_objects")
ASSET_CACHE = Path.home() / ".cache/libero/assets"
RENDER_SUITE, RENDER_TASK, RENDER_SEED = "libero_goal", 0, 1000


def config_file() -> Path:
    return Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")) / "config.yaml"


def benchmark_root() -> Path:
    spec = importlib.util.find_spec("libero")
    if spec is None or spec.origin is None:
        raise SystemExit("the libero package (hf-libero) is not installed in this env")
    return Path(spec.origin).parent / "libero"


def assets_snapshot(offline: bool) -> Path:
    return Path(snapshot_download(ASSETS_REPO, repo_type="dataset", revision=ASSETS_REVISION,
                                  local_files_only=offline, token=False))


def config_document(assets: Path) -> dict:
    root = benchmark_root()
    return {"assets": str(assets), "bddl_files": str(root / "bddl_files"), "datasets": str(root.parent / "datasets"),
            "init_states": str(root / "init_files")}


def problems() -> list[str]:
    try:
        assets = assets_snapshot(offline=True)
    except LocalEntryNotFoundError:
        return [f"{ASSETS_REPO} @ {ASSETS_REVISION} is not in the Hugging Face cache"]
    found = []
    path = config_file()
    if not path.is_file() or yaml.safe_load(path.read_text()) != config_document(assets):
        found.append(f"{path} is missing or does not name the pinned assets and this env's LIBERO package")
    if ASSET_CACHE.resolve() != assets.resolve():
        found.append(f"{ASSET_CACHE} does not lead to {assets}")
    found += [f"asset folder {name} missing in {assets}" for name in ASSET_FOLDERS if not (assets / name).is_dir()]
    return found


def prepare() -> None:
    assets = assets_snapshot(offline=False)
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config_document(assets)))
    if ASSET_CACHE.is_symlink():
        ASSET_CACHE.unlink()
    elif ASSET_CACHE.exists():
        raise SystemExit(f"{ASSET_CACHE} is a directory (an unpinned download by LIBERO itself): move it away and "
                         "prepare again")
    ASSET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ASSET_CACHE.symlink_to(assets, target_is_directory=True)


def render(save: Path | None) -> int:
    if not os.environ.get("MUJOCO_GL"):
        raise SystemExit("set MUJOCO_GL (egl on a GPU box, osmesa without one)")
    from backend import LiberoBackend

    backend = LiberoBackend(RENDER_SUITE, RENDER_TASK, 0, RENDER_SEED)
    try:
        observation = backend.step(START)
    finally:
        backend.close()
    blank = []
    print(f"{RENDER_SUITE} task {RENDER_TASK}: {backend.instruction!r}, state {observation.state.round(4).tolist()}")
    for camera, image in observation.images.items():
        print(f"  {camera}: {image.shape} {image.dtype} mean {image.mean():.1f} std {image.std():.1f}")
        if image.std() == 0:
            blank.append(camera)
        if save is not None:
            save.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(save / f"{camera}.png")
    if blank:
        print(f"blank frames from {blank}: the renderer ({os.environ['MUJOCO_GL']}) is not working", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("prepare", "check", "render"))
    parser.add_argument("--save", type=Path, help="render: write each camera's frame as a PNG here")
    args = parser.parse_args(argv)
    if args.what == "prepare":
        prepare()
        return 0
    if args.what == "render":
        return render(args.save)
    found = problems()
    for problem in found:
        print(problem, file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
