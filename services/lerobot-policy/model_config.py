"""Model configs of the LeRobot policy service: `config/s1/<name>.yaml` → `ModelConfig`, and the Hub pins they name.

One file selects everything model-specific, so the same service serves any LeRobot-format checkpoint:
  adapter            `lerobot`, the only adapter this service implements (any other value is refused)
  checkpoint         `<repo> @ <40-hex revision>`: a LeRobot checkpoint (config.json, model.safetensors, processors)
  overrides          checkpoint config fields to replace (e.g. base_model_path, n_action_steps); a value written
                     `<repo> @ <40-hex revision>` is a pinned Hub snapshot and is replaced by its local directory;
                     device / pretrained_path / pretrained_revision are the service's (from `device`, `checkpoint`)
  cameras            our camera name → the checkpoint's image feature key
  state              the checkpoint's state feature key, fed from `Observation.state`
  action_dim         the action width the robot side expects
  device             torch device of the policy
  execute_steps      optional: how many actions of each chunk the model is meant to run before it is asked again (its
                     own evaluation recipe, e.g. GR00T on LIBERO: 8 of 16); sent with every chunk, so the robot side
                     replaces the chunk there; default: the whole chunk
  loaded_by_repo_id  optional: pin files (`config/*.txt`, repo-relative) of gated repos the model loads by repo id
                     while it loads; the GPU hook caches each at its pin with the owner's login and stops serving
                     when the Hub's main has moved off it (decision 2026-11-07)
  compile            optional: {modules: [attribute paths on the loaded policy], mode: <torch.compile mode>}: each
                     named submodule is replaced by its `torch.compile`d version at load (e.g. GR00T's action head
                     `_groot_model.action_head.model`, the part NVIDIA's own benchmark compiles)
  warmup             optional: number of synthetic requests the service runs before it accepts any (compilation and
                     lazy loading happen there, never inside a robot's request); default 0
  notes              optional free text for people
Unknown keys, a missing key and an unpinned checkpoint fail the load: a config never half-applies.

  python model_config.py pinned CONFIG...              every pinned snapshot `<repo>@<revision>` of the lerobot configs
  python model_config.py loaded-by-repo-id CONFIG...   their gated pin files, one per line, deduplicated
"""
import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = "lerobot"
PINNED = re.compile(r"^(\S+)\s*@\s*([0-9a-f]{40})$")
REQUIRED = {"adapter", "checkpoint", "overrides", "cameras", "state", "action_dim", "device"}
OPTIONAL = {"loaded_by_repo_id", "compile", "warmup", "execute_steps", "notes"}
COMPILE_KEYS = {"modules", "mode"}
SET_BY_SERVICE = {"device", "pretrained_path", "pretrained_revision"}


@dataclass(frozen=True)
class HubSnapshot:
    repo: str
    revision: str

    def __str__(self) -> str:
        return f"{self.repo}@{self.revision}"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint: HubSnapshot
    overrides: dict[str, object]
    cameras: dict[str, str]
    state: str
    action_dim: int
    device: str
    loaded_by_repo_id: tuple[str, ...] = ()
    compile_modules: tuple[str, ...] = ()
    compile_mode: str = "default"
    warmup: int = 0
    execute_steps: int | None = None
    notes: str = ""

    def snapshots(self) -> list[HubSnapshot]:
        return [self.checkpoint, *(v for v in self.overrides.values() if isinstance(v, HubSnapshot))]


def pinned_snapshot(text: object) -> HubSnapshot | None:
    match = PINNED.match(text.strip()) if isinstance(text, str) else None
    return HubSnapshot(match[1], match[2]) if match else None


def load_model_config(path: Path) -> ModelConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a model config is a mapping")
    keys = set(raw)
    problems = [f"missing {sorted(REQUIRED - keys)}"] if REQUIRED - keys else []
    problems += [f"unknown {sorted(keys - REQUIRED - OPTIONAL)}"] if keys - REQUIRED - OPTIONAL else []
    if problems:
        raise ValueError(f"{path}: {'; '.join(problems)}")
    if raw["adapter"] != ADAPTER:
        raise ValueError(f"{path}: adapter {raw['adapter']!r}; this service implements only {ADAPTER!r}")
    checkpoint = pinned_snapshot(raw["checkpoint"])
    if checkpoint is None:
        raise ValueError(f"{path}: checkpoint must be `<repo> @ <40-hex revision>`, got {raw['checkpoint']!r}")
    cameras = raw["cameras"]
    if not isinstance(cameras, dict) or not cameras or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in cameras.items()):
        raise ValueError(f"{path}: cameras must map our camera names to checkpoint image keys")
    if len(set(cameras.values())) != len(cameras):
        raise ValueError(f"{path}: two cameras feed the same checkpoint key: {cameras}")
    overrides = raw["overrides"] or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: overrides must be a mapping of checkpoint config fields")
    if SET_BY_SERVICE & set(overrides):
        raise ValueError(f"{path}: overrides may not set {sorted(SET_BY_SERVICE & set(overrides))} (the service "
                         f"sets them from `device` and `checkpoint`)")
    if not isinstance(raw["state"], str) or not isinstance(raw["device"], str):
        raise ValueError(f"{path}: state (a checkpoint feature key) and device must be strings")
    action_dim = raw["action_dim"]
    if not isinstance(action_dim, int) or isinstance(action_dim, bool) or action_dim <= 0:
        raise ValueError(f"{path}: action_dim must be a positive integer, got {action_dim!r}")
    pin_files = tuple(raw.get("loaded_by_repo_id") or ())
    missing = [p for p in pin_files if not (ROOT / p).is_file()]
    if missing:
        raise ValueError(f"{path}: loaded_by_repo_id names pin files that do not exist: {missing}")
    compile_spec = raw.get("compile") or {}
    if not isinstance(compile_spec, dict) or set(compile_spec) - COMPILE_KEYS or (
            compile_spec and not all(isinstance(m, str) and m for m in compile_spec.get("modules") or [])):
        raise ValueError(f"{path}: compile must be {{modules: [attribute paths], mode: <torch.compile mode>}}")
    if compile_spec and not compile_spec.get("modules"):
        raise ValueError(f"{path}: compile names no modules")
    warmup = raw.get("warmup", 0)
    if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
        raise ValueError(f"{path}: warmup must be a non-negative integer, got {warmup!r}")
    execute_steps = raw.get("execute_steps")
    if execute_steps is not None and (not isinstance(execute_steps, int) or isinstance(execute_steps, bool)
                                      or execute_steps <= 0):
        raise ValueError(f"{path}: execute_steps must be a positive integer, got {execute_steps!r}")
    return ModelConfig(name=Path(path).stem, checkpoint=checkpoint,
                       overrides={k: pinned_snapshot(v) or v for k, v in overrides.items()},
                       cameras=dict(cameras), state=raw["state"], action_dim=action_dim, device=raw["device"],
                       loaded_by_repo_id=pin_files, compile_modules=tuple(compile_spec.get("modules") or ()),
                       compile_mode=str(compile_spec.get("mode") or "default"), warmup=warmup,
                       execute_steps=execute_steps, notes=str(raw.get("notes") or ""))


def lerobot_configs(paths: list[Path]) -> list[ModelConfig]:
    configs = []
    for path in paths:
        raw = yaml.safe_load(Path(path).read_text())
        if isinstance(raw, dict) and raw.get("adapter") == ADAPTER:
            configs.append(load_model_config(path))
    return configs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["pinned", "loaded-by-repo-id"])
    p.add_argument("configs", nargs="+", type=Path)
    a = p.parse_args()
    configs = lerobot_configs(a.configs)
    if a.what == "pinned":
        lines = [str(s) for c in configs for s in c.snapshots()]
    else:
        lines = [f for c in configs for f in c.loaded_by_repo_id]
    for line in dict.fromkeys(lines):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
