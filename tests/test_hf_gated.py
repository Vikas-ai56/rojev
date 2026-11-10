"""`scripts/hf_gated.py`, the GPU hooks' handling of gated Hugging Face repos: the Mac's login is offered as an asset
only when it exists with mode 600, `refs/main` is written only once every file of the pinned revision is in the
cache, and a snapshot counts as ready only while it is referenced, complete and still the Hub's main (models load
such repos online by repo id). No network: the Hub's answers are given as arguments."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hf_pins = script("hf_pins")
hf_gated = script("hf_gated")

REPO = "nvidia/Cosmos-Reason2-2B"
REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
FILES = ["config.json", "model.safetensors", "tokenizer.json"]
PIN_FILE = Path("config/groot_backbone.txt")


def cache_with(tmp_path: Path, files: list[str]) -> Path:
    snapshot = tmp_path / hf_gated.repo_folder(REPO) / "snapshots" / REVISION
    snapshot.mkdir(parents=True)
    for name in files:
        (snapshot / name).write_text("x")
    return tmp_path


def test_the_groot_backbone_pin_names_a_full_revision():
    repo, revision = hf_pins.read_pin(ROOT / PIN_FILE)
    assert repo == REPO and len(revision) == 40 and int(revision, 16) >= 0


def test_ref_is_written_only_after_every_file_is_present(tmp_path):
    cache = cache_with(tmp_path, FILES[:1])
    with pytest.raises(FileNotFoundError, match="missing"):
        hf_gated.write_ref(cache, REPO, REVISION, FILES)
    assert not hf_gated.ref_points_at(cache, REPO, REVISION)
    for name in FILES[1:]:
        (cache / hf_gated.repo_folder(REPO) / "snapshots" / REVISION / name).write_text("x")
    hf_gated.write_ref(cache, REPO, REVISION, FILES)
    assert hf_gated.ref_points_at(cache, REPO, REVISION)
    assert (cache / hf_gated.repo_folder(REPO) / "refs/main").read_text() == REVISION
    assert not hf_gated.ref_points_at(cache, REPO, "0" * 40)


def test_snapshot_is_ready_only_when_referenced_complete_and_still_the_hubs_main(tmp_path):
    assert "hf_gated.py snapshot" in hf_gated.cache_problem(tmp_path, PIN_FILE, REPO, REVISION)
    cache = cache_with(tmp_path, FILES)
    hf_gated.write_ref(cache, REPO, REVISION, FILES)
    assert hf_gated.cache_problem(cache, PIN_FILE, REPO, REVISION) is None
    assert hf_gated.upstream_problem(cache, PIN_FILE, REPO, REVISION, REVISION, FILES) is None
    moved = hf_gated.upstream_problem(cache, PIN_FILE, REPO, REVISION, "f" * 40, FILES)
    assert "re-pin" in moved and str(PIN_FILE) in moved
    grown = hf_gated.upstream_problem(cache, PIN_FILE, REPO, REVISION, REVISION, [*FILES, "vocab.json"])
    assert "vocab.json" in grown


def test_login_is_offered_only_when_present_and_private(tmp_path):
    token = tmp_path / "token"
    with pytest.raises(FileNotFoundError, match="hf auth login"):
        hf_gated.login_file(token)
    token.write_text("x")
    os.chmod(token, 0o644)
    with pytest.raises(PermissionError, match="600"):
        hf_gated.login_file(token)
    os.chmod(token, 0o600)
    assert hf_gated.login_file(token) == token
