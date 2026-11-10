"""A pinned LeRobot (v3.0) SO-101 recording, read through LeRobot's own metadata, parquet loader and video decoder.

`LeRobotDataset` itself is not used: it insists on every camera stream being on disk (one recording's 2 GB depth
stream, another's 60 GB of AV1 video) before it reads anything. The pieces it is built from are used as they are
instead: `LeRobotDatasetMetadata` (episode table, file paths, video offsets), `load_nested_dataset` (the data parquet)
and `decode_video_frames` (the pyav backend; torchcodec needs an FFmpeg this Mac does not have). Sources are public,
so downloads send no token (`token=False`): a stale saved login would otherwise turn them into 401s.

Contract:
  fetch_metadata(pin_file)                       download only `meta/` at the pinned revision; LeRobot's metadata
  Recording.fetch(pin_file, cameras, episodes)   download meta, data and only the named camera streams at the pinned
                                                 revision; with `episodes`, only the data and video files holding
                                                 those episodes, and the recording then holds just those episodes
  rec.episodes                                   episode indices
  rec.states(episode)                            (T, 6) float32 `observation.state`, LeRobot SO-101 units (degrees,
                                                 gripper 0-100)
  rec.frames(episode, indices, camera)           list of (H, W, 3) uint8 frames at those frame indices; every frame
                                                 between the first and last index is decoded and held, so far-apart
                                                 indices of a long episode are asked for one call each
  rec.task(episode)                              the recorded instruction
"""
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import get_hf_features_from_features, load_nested_dataset
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.utils.constants import HF_LEROBOT_HOME

PIN_LINE = re.compile(r"^repo\s+(\S+)\s+@\s+([0-9a-f]{40})\s*$")
VIDEO_BACKEND = "pyav"


def read_pin(pin_file: Path) -> tuple[str, str]:
    for line in pin_file.read_text().splitlines():
        if m := PIN_LINE.match(line):
            return m[1], m[2]
    raise ValueError(f"{pin_file}: no line `repo <id> @ <40-hex revision>`")


def fetch_metadata(pin_file: Path, root: Path | None = None) -> LeRobotDatasetMetadata:
    repo, revision = read_pin(pin_file)
    root = root or HF_LEROBOT_HOME / repo
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=root, allow_patterns=["meta/**"],
                      token=False)
    return LeRobotDatasetMetadata(repo, root=root, revision=revision)


def _episode_files(meta: LeRobotDatasetMetadata, episodes: Iterable[int], video_keys: list[str]) -> list[str]:
    files = set()
    for episode in episodes:
        files.add(str(meta.get_data_file_path(episode)))
        files.update(str(meta.get_video_file_path(episode, key)) for key in video_keys)
    return sorted(files)


@dataclass
class Recording:
    meta: LeRobotDatasetMetadata
    cameras: tuple[str, ...]
    _columns: dict

    @classmethod
    def fetch(cls, pin_file: Path, cameras: tuple[str, ...], root: Path | None = None,
              episodes: Iterable[int] | None = None) -> "Recording":
        meta = fetch_metadata(pin_file, root)
        video_keys = [f"observation.images.{camera}" for camera in cameras]
        if episodes is None:
            files = ["data/**", *(f"videos/{key}/**" for key in video_keys)]
        else:
            episodes = sorted(set(episodes))
            files = _episode_files(meta, episodes, video_keys)
        snapshot_download(meta.repo_id, repo_type="dataset", revision=meta.revision, local_dir=meta.root,
                          allow_patterns=files, token=False)
        table = load_nested_dataset(meta.root / "data", features=get_hf_features_from_features(meta.features))
        columns = {name: np.asarray(table[name]) for name in ("episode_index", "frame_index", "observation.state")}
        if episodes is not None:
            held = np.isin(columns["episode_index"], episodes)
            columns = {name: values[held] for name, values in columns.items()}
        return cls(meta, tuple(cameras), columns)

    @property
    def episodes(self) -> list[int]:
        return sorted(int(e) for e in np.unique(self._columns["episode_index"]))

    def _rows(self, episode: int) -> np.ndarray:
        rows = np.flatnonzero(self._columns["episode_index"] == episode)
        return rows[np.argsort(self._columns["frame_index"][rows])]

    def states(self, episode: int) -> np.ndarray:
        return self._columns["observation.state"][self._rows(episode)].astype(np.float32)

    def task(self, episode: int) -> str:
        return self.meta.episodes[episode]["tasks"][0]

    def frames(self, episode: int, indices: list[int], camera: str) -> list[np.ndarray]:
        if camera not in self.cameras:
            raise KeyError(f"camera {camera!r} was not fetched; fetched: {self.cameras}")
        key = f"observation.images.{camera}"
        offset = self.meta.episodes[episode][f"videos/{key}/from_timestamp"]
        timestamps = [float(offset) + int(i) / self.meta.fps for i in indices]
        path = self.meta.root / self.meta.get_video_file_path(episode, key)
        decoded = decode_video_frames(path, timestamps, 1 / self.meta.fps / 2, VIDEO_BACKEND)
        return [(f.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for f in decoded]
