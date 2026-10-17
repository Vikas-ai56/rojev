"""Facts about a LeRobot checkpoint that evaluation must match, read from the checkpoint itself, never from docs."""
import io
import json
from functools import lru_cache
from pathlib import Path

from huggingface_hub import hf_hub_download


@lru_cache
def rename_map(policy: str) -> dict[str, str]:
    """Camera-key mapping the policy was trained with (its preprocessor's rename step).

    `lerobot-eval` replaces the stored mapping with `--rename_map` (default: empty). LeRobot's RoboCasa docs/CI map
    eye_in_hand->camera2 and right->camera3, while smolvla_robocasa was trained with right->camera2, eye_in_hand->camera3.
    """
    cfg = json.loads(Path(hf_hub_download(policy, "policy_preprocessor.json")).read_text())
    maps = [s["config"]["rename_map"] for s in cfg["steps"] if "rename_map" in (s.get("config") or {})]
    if len(maps) != 1:
        raise ValueError(f"{policy}: expected exactly one rename step in policy_preprocessor.json, got {len(maps)}")
    return maps[0]  # empty = the policy reads the env's native camera keys (e.g. pi05_robocasa)


@lru_cache
def training_instructions(policy: str) -> tuple[str, ...]:
    """Language strings the policy saw in training: the task table of the dataset named in its train_config.json.

    Empty if the checkpoint ships no train_config.json (e.g. pi05_robocasa): the vocabulary is then unknown.
    """
    import pyarrow.parquet as pq
    from huggingface_hub.errors import EntryNotFoundError

    try:
        train = json.loads(Path(hf_hub_download(policy, "train_config.json")).read_text())
    except EntryNotFoundError:
        return ()
    repo = train["dataset"]["repo_id"]
    table = pq.read_table(io.BytesIO(Path(hf_hub_download(repo, "meta/tasks.parquet", repo_type="dataset")).read_bytes()))
    df = table.to_pandas()
    return tuple(str(t) for t in (df.index if "task" not in df.columns else df["task"]))
