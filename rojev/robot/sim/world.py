"""MuJoCo world utilities: loading and composing scenes, joint readout, contacts, rendering, segmentation ground truth.

Used by the simulator backend and by evaluators. Pipeline code reaches the robot through `rojev.robot` instead.

`ROBOT_OPTIONS` are the solver settings the robot model is tuned for; `MjSpec.attach` would otherwise keep the world
file's defaults.
"""
import os
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
SO101_DIR = Path(os.environ.get("ROJEV_SO101_DIR", REPO / "vendor/mujoco_menagerie/robotstudio_so101"))
SCENES_DIR = REPO / "scenes"
ROBOT_OPTIONS = ("timestep", "iterations", "ls_iterations", "impratio", "integrator", "cone")


def compose(world: str | Path, robot: str | Path = SO101_DIR / "so101.xml", mount: str = "robot_mount") -> mujoco.MjModel:
    """Attach the robot MJCF at the world's `mount` frame. Both files stay unmodified; the robot's solver options win."""
    world_path = Path(world) if Path(world).is_absolute() else SCENES_DIR / world
    w, r = mujoco.MjSpec.from_file(str(world_path)), mujoco.MjSpec.from_file(str(robot))
    for key in ROBOT_OPTIONS:
        setattr(w.option, key, getattr(r.option, key))
    frame = w.frame(mount)
    if frame is None:
        raise ValueError(f"{world_path.name} has no frame named {mount!r}")
    w.attach(r, frame=frame, prefix="")
    return w.compile()


def contacts_with(m: mujoco.MjModel, d: mujoco.MjData, geom: str, ignore_bodies: tuple[str, ...] = ()) -> int:
    """Active contacts between `geom` and any body other than the world and `ignore_bodies`."""
    gid = m.geom(geom).id
    ignored = {0} | {m.body(b).id for b in ignore_bodies}
    n = 0
    for c in d.contact[: d.ncon]:
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            n += m.geom_bodyid[other] not in ignored
    return n


def visible_pixels(m: mujoco.MjModel, renderer: mujoco.Renderer, d: mujoco.MjData, geom: str,
                   camera: int | str = -1) -> int:
    """Ground truth: pixels of `geom` in a segmentation render from `camera` (0 = hidden, occluded or off-screen).
    For evaluators only, never as model input."""
    renderer.enable_segmentation_rendering()
    try:
        renderer.update_scene(d, camera=camera)
        seg = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
    return int(((seg[..., 0] == m.geom(geom).id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)).sum())
