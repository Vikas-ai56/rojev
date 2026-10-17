"""MuJoCo as a robot backend: the same interface, data types and timing behaviour a real SO-101 presents.

Physics advances only inside `SimClock.wait_until`, so control loops pace the simulator exactly as they would pace
hardware. Cameras deliver frames at their own rate and latency, with calibration derived from the model.
Privileged truth is exposed separately through `SimGroundTruth`, for evaluators and tests only.
"""
import time
from contextlib import contextmanager
from dataclasses import dataclass

import mujoco
import numpy as np

from .. import sim
from .base import Clock, Robot
from .limits import JointLimits
from .types import CameraFrame, CameraInfo, JointCommand, JointState, Observation, Stamp


@dataclass(frozen=True)
class CameraSpec:
    name: str  # camera name in the model
    width: int = 640
    height: int = 480
    fps: float = 30.0
    depth: bool = False
    latency_s: float = 0.0  # time from exposure until the frame is available


class ViewerClosed(RuntimeError):
    pass


class SimClock(Clock):
    """Simulated time. `wait_until` steps physics; with `realtime` it also keeps the wall clock in step."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, realtime: bool = False):
        self.m, self.d, self.realtime = m, d, realtime
        self.viewer = None  # a mujoco.viewer handle, synced every step when set
        self._anchor: tuple[float, float] | None = None  # (sim time, wall time) for real-time pacing

    def now(self) -> float:
        return self.d.time

    def wait_until(self, t: float) -> None:
        if self._anchor is None:
            self._anchor = (self.d.time, time.perf_counter())
        while self.d.time + 0.5 * self.m.opt.timestep < t:
            mujoco.mj_step(self.m, self.d)
            if self.viewer is not None:
                if not self.viewer.is_running():
                    raise ViewerClosed("viewer window closed")
                self.viewer.sync()
            if self.realtime:
                sim0, wall0 = self._anchor
                time.sleep(max(0.0, (self.d.time - sim0) - (time.perf_counter() - wall0)))

    @contextmanager
    def paused(self):
        """Physics does not advance inside; with `realtime`, the wall time spent here is not caught up afterwards."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._anchor is not None:
                self._anchor = (self._anchor[0], self._anchor[1] + time.perf_counter() - t0)


def camera_info(m: mujoco.MjModel, cam: int, width: int, height: int) -> CameraInfo:
    """Pinhole calibration of a fixed MuJoCo camera, expressed in the ROS optical convention."""
    if m.cam_mode[cam] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
        raise ValueError(f"camera {m.camera(cam).name!r} is not fixed to its body; its extrinsics are not constant")
    if np.any(m.cam_sensorsize[cam] > 0):
        raise NotImplementedError(f"camera {m.camera(cam).name!r} uses sensorsize/focal intrinsics; only fovy is supported")
    f = (height / 2) / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.cam_quat[cam])
    T = np.eye(4)
    T[:3, :3] = R.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])  # MuJoCo camera looks along -z with y up
    T[:3, 3] = m.cam_pos[cam]
    body = m.cam_bodyid[cam]
    return CameraInfo(width, height, K, np.zeros(0), "none", "world" if body == 0 else m.body(body).name, T)


class _SimCamera:
    def __init__(self, m: mujoco.MjModel, spec: CameraSpec):
        self.m, self.spec, self.id = m, spec, m.camera(spec.name).id
        self.info = camera_info(m, self.id, spec.width, spec.height)
        self.renderer: mujoco.Renderer | None = None
        self.next_due, self.seq = 0.0, 0
        self.pending: list[tuple[float, float, np.ndarray, np.ndarray | None]] = []  # (available, acquired, rgb, depth)
        self.latest: CameraFrame | None = None

    def open(self) -> None:
        self.renderer = mujoco.Renderer(self.m, height=self.spec.height, width=self.spec.width)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def poll(self, d: mujoco.MjData, now: float) -> CameraFrame | None:
        period = 1.0 / self.spec.fps
        if now + 1e-9 >= self.next_due:
            self.renderer.update_scene(d, camera=self.id)
            rgb = self.renderer.render().copy()
            depth = None
            if self.spec.depth:
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render().astype(np.float32)
                self.renderer.disable_depth_rendering()
            self.pending.append((now + self.spec.latency_s, now, rgb, depth))
            self.next_due += period
            if self.next_due <= now:  # the loop fell behind the camera rate: drop the missed frames
                self.next_due = now + period
        while self.pending and self.pending[0][0] <= now + 1e-9:
            available, acquired, rgb, depth = self.pending.pop(0)
            self.latest = CameraFrame(Stamp(acquired, available, self.seq), self.spec.name, rgb, depth, self.info)
            self.seq += 1
        return self.latest


class SimGroundTruth:
    """Privileged simulator state. Evaluators and tests only; never an input to estimation, planning or OneJev."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.model, self.data = m, d
        self._seg: dict[tuple[int, int], mujoco.Renderer] = {}

    def body_pose(self, name: str) -> np.ndarray:
        b = self.data.body(name)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = b.xmat.reshape(3, 3), b.xpos
        return T

    def visible_pixels(self, geom: str, camera: str, width: int = 640, height: int = 480) -> int:
        key = (width, height)
        if key not in self._seg:
            self._seg[key] = mujoco.Renderer(self.model, height=height, width=width)
        return sim.visible_pixels(self.model, self._seg[key], self.data, geom, camera=camera)

    def contacts_with(self, geom: str, ignore_bodies: tuple[str, ...] = ()) -> int:
        return sim.contacts_with(self.model, self.data, geom, ignore_bodies)

    def close(self) -> None:
        for r in self._seg.values():
            r.close()
        self._seg.clear()


class MujocoRobot(Robot):
    def __init__(self, model: mujoco.MjModel, cameras: tuple[CameraSpec, ...] = (), joint_names: tuple[str, ...] | None = None,
                 max_velocity: dict[str, float] | None = None, velocity_source: str = "unspecified",
                 control_rate_hz: float = 30.0, realtime: bool = False):
        self.model, self.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        names = joint_names or tuple(model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu))
        limits = JointLimits.from_model(model, names, max_velocity, velocity_source)
        super().__init__(limits, SimClock(model, self.data, realtime), control_rate_hz)
        joints = [model.joint(n).id for n in names]
        self._qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self._dadr = np.array([model.jnt_dofadr[j] for j in joints])
        self._act = np.array([next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == j) for j in joints])
        for a, n in zip(self._act, names):
            if not (model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                    and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
                    and np.isclose(model.actuator_biasprm[a, 1], -model.actuator_gainprm[a, 0])):
                raise ValueError(f"actuator for {n!r} is not a position servo; its ctrl is not a joint target")
        self._cams = {c.name: _SimCamera(model, c) for c in cameras}
        self._seq = 0
        self.ground_truth = SimGroundTruth(model, self.data)

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._cams)

    def attach_viewer(self, viewer) -> None:
        self.clock.viewer = viewer

    def _connect(self) -> None:
        for c in self._cams.values():
            c.open()
        self.data.ctrl[self._act] = self.data.qpos[self._qadr]  # hold the current pose

    def _disconnect(self) -> None:
        for c in self._cams.values():
            c.close()
        self.ground_truth.close()

    def _read(self) -> Observation:
        now = self.data.time
        joints = JointState(Stamp(now, now, self._seq), self.joint_names, self.data.qpos[self._qadr].copy(),
                            self.data.qvel[self._dadr].copy())
        self._seq += 1
        frames = {name: f for name, c in self._cams.items() if (f := c.poll(self.data, now)) is not None}
        return Observation(joints, frames)

    def _write(self, cmd: JointCommand) -> None:
        self.data.ctrl[self._act] = cmd.position
