"""MuJoCo as a robot backend: the same interface, data types and timing behaviour a real SO-101 presents.

Three time modes (`TIME_MODES`):
- `lockstep`: physics advances only inside `SimClock.wait_until`. Deterministic; nothing moves while the caller
  computes, so compute delay must be injected explicitly by waiting. For tests and data generation.
- `paced`: lockstep, but never faster than the wall clock. For watching a lockstep run in the viewer.
- `realtime`: physics runs on its own thread in step with the wall clock (`RealtimeSimClock`). The world keeps moving
  while the caller computes, exactly as hardware does. For latency and responsiveness tests.
Cameras deliver frames at their own rate and latency, with calibration derived from the model.
Privileged truth is exposed separately through `SimGroundTruth`, for evaluators and tests only.
"""
import threading
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from ..base import Clock, Robot
from ..limits import JointLimits
from ..types import CameraFrame, CameraInfo, JointCommand, JointState, RobotObservation, Stamp
from . import world
from .gl import GLThread

TIME_MODES = ("lockstep", "paced", "realtime")


@dataclass(frozen=True)
class CameraSpec:
    """`name`: the camera in the model; `latency_s`: time from exposure until the frame is available."""

    name: str
    width: int = 640
    height: int = 480
    fps: float = 30.0
    depth: bool = False
    latency_s: float = 0.0


class ViewerClosed(RuntimeError):
    pass


class SimClock(Clock):
    """Lockstep simulated time: `wait_until` steps physics. With `paced`, stepping inside one `wait_until` never runs
    ahead of the wall clock; time the caller spends outside `wait_until` is not simulated time. `viewer` (a
    mujoco.viewer handle), when set, is synced every step."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, paced: bool = False):
        self.m, self.d, self.paced = m, d, paced
        self.lock = threading.RLock()
        self.viewer = None

    def now(self) -> float:
        return self.d.time

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def wait_until(self, t: float) -> None:
        sim0, wall0 = self.d.time, time.perf_counter()
        while self.d.time + 0.5 * self.m.opt.timestep < t:
            with self.lock:
                mujoco.mj_step(self.m, self.d)
                _sync_viewer(self.viewer)
            if self.paced:
                time.sleep(max(0.0, (self.d.time - sim0) - (time.perf_counter() - wall0)))


class RealtimeSimClock(Clock):
    """Physics on its own thread, in step with the wall clock, independent of the caller (as hardware is).

    `wait_until` sleeps until simulated time reaches `t`. If stepping cannot keep up with the wall clock, the
    simulation falls behind instead of skipping time; `lag_s` (now) and `max_lag_s` (worst so far) report it.
    An error on the physics thread (e.g. the viewer closing) is re-raised in the caller's next `wait_until`.
    """

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
        self.m, self.d = m, d
        self.lock = threading.RLock()
        self._stepped = threading.Condition(self.lock)
        self.viewer = None
        self.lag_s = self.max_lag_s = 0.0
        self._thread: threading.Thread | None = None
        self._running = False
        self._error: BaseException | None = None

    def now(self) -> float:
        return self.d.time

    def start(self) -> None:
        if self._thread is not None:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="mujoco-physics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def wait_until(self, t: float) -> None:
        if self._thread is None:
            raise RuntimeError("real-time physics is not running; connect the robot first")
        with self._stepped:
            self._stepped.wait_for(lambda: self._error is not None or not self._running
                                   or self.d.time + 0.5 * self.m.opt.timestep >= t)
        if self._error is not None:
            raise self._error
        if not self._running:
            raise RuntimeError("real-time physics stopped while waiting")

    def _run(self) -> None:
        dt = self.m.opt.timestep
        sim0, wall0 = self.d.time, time.perf_counter()
        try:
            while self._running:
                target = sim0 + (time.perf_counter() - wall0)
                with self._stepped:
                    while self.d.time + 0.5 * dt < target:
                        mujoco.mj_step(self.m, self.d)
                    self.lag_s = max(0.0, sim0 + (time.perf_counter() - wall0) - self.d.time)
                    self.max_lag_s = max(self.max_lag_s, self.lag_s)
                    _sync_viewer(self.viewer)
                    self._stepped.notify_all()
                time.sleep(max(0.0, (self.d.time + dt - sim0) - (time.perf_counter() - wall0)))
        except BaseException as e:
            with self._stepped:
                self._error = e
                self._stepped.notify_all()


def _sync_viewer(viewer) -> None:
    if viewer is not None:
        if not viewer.is_running():
            raise ViewerClosed("viewer window closed")
        viewer.sync()


def camera_info(m: mujoco.MjModel, cam: int, width: int, height: int) -> CameraInfo:
    """Pinhole calibration of a fixed MuJoCo camera, expressed in the ROS optical convention. MuJoCo cameras look
    along -z with y up; the optical frame looks along +z with y down."""
    if m.cam_mode[cam] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
        raise ValueError(f"camera {m.camera(cam).name!r} is not fixed to its body; its extrinsics are not constant")
    if np.any(m.cam_sensorsize[cam] > 0):
        raise NotImplementedError(f"camera {m.camera(cam).name!r} uses sensorsize/focal intrinsics; only fovy is supported")
    f = (height / 2) / np.tan(np.radians(m.cam_fovy[cam]) / 2)
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.cam_quat[cam])
    T = np.eye(4)
    mujoco_to_optical = np.diag([1.0, -1.0, -1.0])
    T[:3, :3] = R.reshape(3, 3) @ mujoco_to_optical
    T[:3, 3] = m.cam_pos[cam]
    body = m.cam_bodyid[cam]
    return CameraInfo(width, height, K, np.zeros(0), "none", "world" if body == 0 else m.body(body).name, T)


class _SimCamera:
    """`pending` holds rendered exposures as (available, acquired, rgb, depth) until their latency has passed."""

    def __init__(self, m: mujoco.MjModel, spec: CameraSpec, gl: GLThread):
        self.m, self.spec, self.id, self.gl = m, spec, m.camera(spec.name).id, gl
        self.info = camera_info(m, self.id, spec.width, spec.height)
        self.renderer: mujoco.Renderer | None = None
        self.next_due, self.seq = 0.0, 0
        self.pending: list[tuple[float, float, np.ndarray, np.ndarray | None]] = []
        self.latest: CameraFrame | None = None

    def open(self) -> None:
        self.renderer = self.gl.call(mujoco.Renderer, self.m, self.spec.height, self.spec.width)

    def close(self) -> None:
        if self.renderer is not None:
            self.gl.call(self.renderer.close)
            self.renderer = None

    def due(self, now: float) -> bool:
        return now + 1e-9 >= self.next_due

    def capture(self, d: mujoco.MjData) -> None:
        """Copy the scene for this exposure. Call under the physics lock; rendering happens outside it."""
        self.renderer.update_scene(d, camera=self.id)

    def expose(self, now: float) -> None:
        """Render the captured scene as an exposure at `now`; it becomes available after the camera latency."""
        rgb, depth = self.gl.call(self._render)
        self.pending.append((now + self.spec.latency_s, now, rgb, depth))
        self.next_due += 1.0 / self.spec.fps
        if self.next_due <= now:
            self.next_due = now + 1.0 / self.spec.fps

    def _render(self) -> tuple[np.ndarray, np.ndarray | None]:
        rgb = self.renderer.render().copy()
        depth = None
        if self.spec.depth:
            self.renderer.enable_depth_rendering()
            depth = self.renderer.render().astype(np.float32)
            self.renderer.disable_depth_rendering()
        return rgb, depth

    def deliver(self, now: float) -> CameraFrame | None:
        while self.pending and self.pending[0][0] <= now + 1e-9:
            available, acquired, rgb, depth = self.pending.pop(0)
            self.latest = CameraFrame(Stamp(acquired, available, self.seq), self.spec.name, rgb, depth, self.info)
            self.seq += 1
        return self.latest


class SimGroundTruth:
    """Privileged simulator state. Evaluators and tests only; never an input to estimation, planning or OneJev."""

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, lock: threading.RLock, gl: GLThread):
        self.model, self.data, self.lock, self.gl = m, d, lock, gl
        self._seg: dict[tuple[int, int], mujoco.Renderer] = {}

    def body_pose(self, name: str) -> np.ndarray:
        with self.lock:
            b = self.data.body(name)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = b.xmat.reshape(3, 3), b.xpos
        return T

    def visible_pixels(self, geom: str, camera: str, width: int = 640, height: int = 480) -> int:
        key = (width, height)
        if key not in self._seg:
            self._seg[key] = self.gl.call(mujoco.Renderer, self.model, height, width)
        with self.lock:
            return self.gl.call(world.visible_pixels, self.model, self._seg[key], self.data, geom, camera)

    def contacts_with(self, geom: str, ignore_bodies: tuple[str, ...] = ()) -> int:
        with self.lock:
            return world.contacts_with(self.model, self.data, geom, ignore_bodies)

    def close(self) -> None:
        for r in self._seg.values():
            self.gl.call(r.close)
        self._seg.clear()


class MujocoRobot(Robot):
    def __init__(self, model: mujoco.MjModel, cameras: tuple[CameraSpec, ...] = (), joint_names: tuple[str, ...] | None = None,
                 max_velocity: dict[str, float] | None = None, velocity_source: str = "unspecified",
                 control_rate_hz: float = 30.0, time_mode: str = "lockstep"):
        if time_mode not in TIME_MODES:
            raise ValueError(f"time_mode {time_mode!r} not in {TIME_MODES}")
        self.model, self.data, self.time_mode = model, mujoco.MjData(model), time_mode
        mujoco.mj_forward(model, self.data)
        names = joint_names or tuple(model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu))
        limits = JointLimits.from_model(model, names, max_velocity, velocity_source)
        clock = (RealtimeSimClock(model, self.data) if time_mode == "realtime"
                 else SimClock(model, self.data, paced=time_mode == "paced"))
        super().__init__(limits, clock, control_rate_hz)
        joints = [model.joint(n).id for n in names]
        self._qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self._dadr = np.array([model.jnt_dofadr[j] for j in joints])
        self._act = np.array([next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == j) for j in joints])
        for a, n in zip(self._act, names):
            if not (model.actuator_gaintype[a] == mujoco.mjtGain.mjGAIN_FIXED
                    and model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
                    and np.isclose(model.actuator_biasprm[a, 1], -model.actuator_gainprm[a, 0])):
                raise ValueError(f"actuator for {n!r} is not a position servo; its ctrl is not a joint target")
        self._gl = GLThread("mujoco-gl")
        self._cams = {c.name: _SimCamera(model, c, self._gl) for c in cameras}
        self._seq = 0
        self.ground_truth = SimGroundTruth(model, self.data, clock.lock, self._gl)
        self._gl_closed = False

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._cams)

    def attach_viewer(self, viewer) -> None:
        self.clock.viewer = viewer

    def _connect(self) -> None:
        if self._gl_closed:
            self._gl, self._gl_closed = GLThread("mujoco-gl"), False
            for c in self._cams.values():
                c.gl = self._gl
            self.ground_truth.gl = self._gl
        for c in self._cams.values():
            c.open()
        with self.clock.lock:
            self.data.ctrl[self._act] = self.data.qpos[self._qadr]
        self.clock.start()

    def _disconnect(self) -> None:
        self.clock.stop()
        for c in self._cams.values():
            c.close()
        self.ground_truth.close()
        self._gl.close()
        self._gl_closed = True

    def _read(self) -> RobotObservation:
        with self.clock.lock:
            now = self.data.time
            q, v = self.data.qpos[self._qadr].copy(), self.data.qvel[self._dadr].copy()
            due = [c for c in self._cams.values() if c.due(now)]
            for c in due:
                c.capture(self.data)
        for c in due:
            c.expose(now)
        joints = JointState(Stamp(now, now, self._seq), self.joint_names, q, v)
        self._seq += 1
        frames = {name: f for name, c in self._cams.items() if (f := c.deliver(now)) is not None}
        return RobotObservation(joints, frames)

    def _write(self, cmd: JointCommand) -> None:
        with self.clock.lock:
            self.data.ctrl[self._act] = cmd.position
