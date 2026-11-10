"""Model configs: every committed lerobot config in config/s1/ loads, is pinned at full revisions (GR00T's base at
the same revision as config/groot_base_weights.txt) and names its gated backbone pin; the loader refuses anything
it would otherwise half-apply; the hook's `pinned` / `loaded-by-repo-id` listings are deduplicated."""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from model_config import ROOT, HubSnapshot, load_model_config, lerobot_configs, pinned_snapshot

SERVICE = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "config/s1").glob("*.yaml"))
GOAL = ROOT / "config/s1/groot-libero-goal.yaml"


def pin_file(name: str) -> HubSnapshot:
    line = next(line for line in (ROOT / "config" / name).read_text().splitlines() if line.startswith("repo "))
    return pinned_snapshot(line.removeprefix("repo "))


def test_every_libero_suite_has_a_pinned_config_on_the_same_base_and_backbone():
    configs = {c.name: c for c in lerobot_configs(CONFIGS)}
    assert {"groot-libero-spatial", "groot-libero-object", "groot-libero-goal", "groot-libero-10"} <= set(configs)
    for config in configs.values():
        assert config.checkpoint.repo.startswith("nvidia/gr00t17-lerobot-libero_")
        assert len(config.checkpoint.revision) == 40
        assert config.overrides["base_model_path"] == pin_file("groot_base_weights.txt")
        assert config.overrides["n_action_steps"] == 16
        assert config.cameras == {"top": "observation.images.image", "wrist": "observation.images.wrist_image"}
        assert (config.state, config.action_dim, config.device) == ("observation.state", 7, "cuda")
        assert config.loaded_by_repo_id == ("config/groot_backbone.txt",)
        assert config.execute_steps == 8
    assert len({c.checkpoint for c in configs.values()}) == len(configs)


def test_pinned_snapshot_needs_a_full_revision():
    assert pinned_snapshot("a/b @ " + "0" * 40) == HubSnapshot("a/b", "0" * 40)
    assert pinned_snapshot("a/b@" + "f" * 40) == HubSnapshot("a/b", "f" * 40)
    assert pinned_snapshot("a/b") is None and pinned_snapshot("a/b @ 32a6ec7") is None and pinned_snapshot(16) is None


def write(tmp_path: Path, **changes) -> Path:
    raw = {**yaml.safe_load(GOAL.read_text()), **changes}
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump({k: v for k, v in raw.items() if v is not None}))
    return path


@pytest.mark.parametrize("changes, message", [
    ({"adapter": "openpi"}, "implements only 'lerobot'"),
    ({"checkpoint": "nvidia/gr00t17-lerobot-libero_goal-640"}, "checkpoint must be"),
    ({"cameras": None}, "missing \\['cameras'\\]"),
    ({"camera_map": {}}, "unknown \\['camera_map'\\]"),
    ({"cameras": {"top": "observation.images.image", "wrist": "observation.images.image"}}, "same checkpoint key"),
    ({"overrides": {"device": "cpu"}}, "may not set \\['device'\\]"),
    ({"action_dim": "7"}, "positive integer"),
    ({"loaded_by_repo_id": ["config/no_such_pin.txt"]}, "do not exist"),
    ({"execute_steps": 0}, "execute_steps must be a positive integer"),
    ({"execute_steps": True}, "execute_steps must be a positive integer"),
])
def test_the_loader_refuses_what_it_would_half_apply(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_model_config(write(tmp_path, **changes))


def test_an_override_written_as_a_pin_is_a_snapshot_and_others_stay_values(tmp_path):
    config = load_model_config(write(tmp_path, overrides={"base_model_path": "a/b @ " + "1" * 40, "n_action_steps": 8}))
    assert config.overrides == {"base_model_path": HubSnapshot("a/b", "1" * 40), "n_action_steps": 8}
    assert config.snapshots() == [config.checkpoint, HubSnapshot("a/b", "1" * 40)]


def run(*args: str) -> list[str]:
    out = subprocess.run([sys.executable, str(SERVICE / "model_config.py"), *args], capture_output=True, text=True,
                         check=True).stdout
    return out.splitlines()


def test_the_hook_listings_cover_every_config_once():
    pinned = run("pinned", *map(str, CONFIGS))
    assert len(pinned) == len(set(pinned)) == len(CONFIGS) + 1
    assert str(pin_file("groot_base_weights.txt")) in pinned
    assert run("loaded-by-repo-id", *map(str, CONFIGS)) == ["config/groot_backbone.txt"]
