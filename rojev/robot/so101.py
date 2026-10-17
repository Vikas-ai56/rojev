"""SO-101 backends built from `config/so101.yaml` (values the MJCF does not carry, each with its source)."""
from pathlib import Path

import yaml

from .. import sim
from .mujoco_robot import CameraSpec, MujocoRobot

CONFIG = sim.REPO / "config/so101.yaml"


def config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(path.read_text())


def simulated(scene: str = "pick_place.xml", cameras: tuple[CameraSpec, ...] = (CameraSpec("overhead"),),
              realtime: bool = False) -> MujocoRobot:
    cfg = config()
    vmax = cfg["max_velocity_rad_s"]
    model = sim.compose(scene)
    names = tuple(model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu))
    return MujocoRobot(model, cameras, names, max_velocity={n: vmax["value"] for n in names},
                       velocity_source=vmax["source"], control_rate_hz=cfg["control_rate_hz"], realtime=realtime)
