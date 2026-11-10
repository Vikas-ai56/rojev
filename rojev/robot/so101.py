"""SO-101 backends and units, built from `config/so101.yaml` (values the MJCF does not carry, each with its source).

`simulated` builds the MuJoCo SO-101. `lerobot_units` reads the conversion between the core's joint radians and
LeRobot's SO-101 units, which recorded data and S1 models use (decision 2026-11-07): body joints in degrees around
the calibration midpoint, rad = zero_rad + sign · radians(degrees); the gripper as 0–100 of its calibrated range,
linear between `rad_at_0` and `rad_at_100`. Joint order is the body joints, then the gripper (LeRobot's order). The
simulator's values are declared in the config; the real arm's come from its calibration.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .sim import world
from .sim.mujoco_robot import CameraSpec, MujocoRobot

CONFIG = world.REPO / "config/so101.yaml"


def config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def simulated(scene: str = "pick_place.xml", cameras: tuple[CameraSpec, ...] = (CameraSpec("overhead"),),
              time_mode: str = "lockstep") -> MujocoRobot:
    cfg = config()
    vmax = cfg["max_velocity_rad_s"]
    model = world.compose(scene)
    names = tuple(model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu))
    return MujocoRobot(model, cameras, names, max_velocity={n: vmax["value"] for n in names},
                       velocity_source=vmax["source"], control_rate_hz=cfg["control_rate_hz"], time_mode=time_mode)


@dataclass(frozen=True)
class LeRobotUnits:
    names: tuple[str, ...]
    sign: np.ndarray
    zero_rad: np.ndarray
    gripper_rad: tuple[float, float]

    def to_lerobot(self, radians: np.ndarray) -> np.ndarray:
        q = np.asarray(radians, dtype=np.float64)
        self._check(q)
        body = self.sign * np.degrees(q[..., :-1] - self.zero_rad)
        lo, hi = self.gripper_rad
        gripper = 100.0 * (q[..., -1:] - lo) / (hi - lo)
        return np.concatenate([body, gripper], axis=-1)

    def from_lerobot(self, values: np.ndarray) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        self._check(v)
        body = self.zero_rad + self.sign * np.radians(v[..., :-1])
        lo, hi = self.gripper_rad
        gripper = lo + v[..., -1:] / 100.0 * (hi - lo)
        return np.concatenate([body, gripper], axis=-1)

    def _check(self, a: np.ndarray) -> None:
        if a.shape[-1] != len(self.names):
            raise ValueError(f"expected {len(self.names)} joints {self.names}, got last dimension {a.shape[-1]}")


def lerobot_units(path: Path = CONFIG) -> LeRobotUnits:
    section = config(path)["lerobot"]
    body = section["body"]
    gripper = section["gripper"]
    return LeRobotUnits(names=(*body, gripper["joint"]),
                        sign=np.array([float(j["sign"]) for j in body.values()]),
                        zero_rad=np.array([float(j["zero_rad"]) for j in body.values()]),
                        gripper_rad=(float(gripper["rad_at_0"]), float(gripper["rad_at_100"])))
