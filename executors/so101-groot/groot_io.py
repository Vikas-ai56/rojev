"""Binds GR00T N1.7's policy API (Isaac-GR00T `Gr00tPolicy`, getting_started/policy.md) to rojev's `Observation` and
`ActionChunk` for the SO-101, without re-implementing GR00T's contract.

What the model expects comes from the model: `Gr00tPolicy.get_modality_config()` names the video, state, language
and action keys and their temporal horizons (the fine-tuned checkpoint's own config), and `Gr00tPolicy(strict=True)`
validates every observation and action against it. What only we know comes from `modality.json` beside this file, in
GR00T's own dataset-layout format (the file our recorded SO-101 datasets carry): each state / action key's slice of the
robot's joint vector, and each camera key's source `observation.images.<camera>`, whose suffix is the robot's camera
name. `Binding.of` checks at load that the two agree (every model key has a slice or a camera, the action slices cover
every joint once, one language key, one frame of video and state since no history is sent) and fails fast otherwise.

Per request: joint radians → LeRobot units (`rojev.robot.so101.LeRobotUnits`, decision 2026-11-07), GR00T's
(B=1, T=1) axes, the text under the model's own language key (`annotation.human.task_description` for a whole-task
checkpoint, `sub_task` for a step-trained one); back: each action key's (1, T, D) array into its joint slice →
radians. A non-finite action fails the request: GR00T's check covers keys, shapes and dtypes only, and Isaac-GR00T
#791 reports NaN reaching the arm.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rojev.protocol import Observation
from rojev.robot.so101 import LeRobotUnits

LAYOUT = Path(__file__).with_name("modality.json")
CAMERA_SOURCE = "observation.images."


def layout(path: Path = LAYOUT) -> dict:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Binding:
    cameras: dict[str, str]
    state: dict[str, tuple[int, int]]
    action: dict[str, tuple[int, int]]
    language: str
    joints: int

    @staticmethod
    def of(config: dict, data_layout: dict) -> "Binding":
        for modality in ("video", "state"):
            frames = len(config[modality].delta_indices)
            if frames != 1:
                raise ValueError(f"the model takes {frames} {modality} frames; this adapter sends the current one only")
        languages = list(config["language"].modality_keys)
        if len(languages) != 1:
            raise ValueError(f"expected one language key, the model has {languages}")
        cameras = {}
        for key in config["video"].modality_keys:
            source = data_layout["video"].get(key, {}).get("original_key", "")
            if not source.startswith(CAMERA_SOURCE):
                raise ValueError(f"camera key {key!r} has no `{CAMERA_SOURCE}<camera>` source in the layout")
            cameras[key] = source.removeprefix(CAMERA_SOURCE)
        state, action = (_slices(config, data_layout, m) for m in ("state", "action"))
        joints = max(end for _, end in state.values())
        covered = sorted(i for start, end in action.values() for i in range(start, end))
        if covered != list(range(joints)):
            raise ValueError(f"action slices {action} must cover joints 0..{joints - 1} exactly once")
        return Binding(cameras, state, action, languages[0], joints)

    def observation(self, obs: Observation, text: str, units: LeRobotUnits) -> dict:
        missing = sorted(set(self.cameras.values()) - set(obs.images))
        if missing:
            raise ValueError(f"observation lacks cameras {missing}; has {sorted(obs.images)}")
        q = units.to_lerobot(obs.state).astype(np.float32)
        return {"video": {key: np.asarray(obs.images[camera], dtype=np.uint8)[None, None]
                          for key, camera in self.cameras.items()},
                "state": {key: q[None, None, start:end] for key, (start, end) in self.state.items()},
                "language": {self.language: [[text]]}}

    def actions(self, action: dict, units: LeRobotUnits) -> np.ndarray:
        horizon = next(iter(action.values())).shape[1]
        values = np.empty((horizon, self.joints))
        for key, (start, end) in self.action.items():
            values[:, start:end] = action[key][0]
        if not np.isfinite(values).all():
            raise ValueError("GR00T returned a non-finite action")
        return units.from_lerobot(values).astype(np.float32)


def _slices(config: dict, data_layout: dict, modality: str) -> dict[str, tuple[int, int]]:
    slices = {}
    for key in config[modality].modality_keys:
        entry = data_layout[modality].get(key)
        if entry is None:
            raise ValueError(f"{modality} key {key!r} has no slice in the layout")
        slices[key] = (int(entry["start"]), int(entry["end"]))
    return slices
