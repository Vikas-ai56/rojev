"""Live MuJoCo viewer for a running RoboCasa simulation, in its own process (a mirror).

The interactive viewer cannot share the simulation's process on macOS: robosuite renders cameras through a GLFW
window, which must be created on the main thread, and `mjpython` (required by the passive viewer) takes the main thread
for Cocoa. So the simulation process publishes its state and a separate `mjpython` process renders it:

  MirrorPublisher (simulation process)  saves the compiled model (.mjb), then after every step copies time, qpos and
                                        mocap poses into shared memory with a sequence number
  main (viewer process, `mjpython`)     loads the same .mjb, copies each new state in, runs mj_forward, syncs the viewer

The viewer only reads; it never feeds anything back, and only the publisher unlinks the shared memory (Python 3.12
registers an attached block with the attaching process's resource tracker, which would unlink it when the viewer
exits; the viewer unregisters it, the documented fix until `track=False` in 3.13). `mjpython` replaces dyld's default fallback library path with its
own, which hides the system zlib some wheels link against, so the publisher launches it with the default restored
(`/usr/local/lib:/usr/lib`, prepended to any existing value).
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np

HEADER = 2
DYLD_DEFAULT_FALLBACK = "/usr/local/lib:/usr/lib"


def layout(nq: int, nmocap: int) -> int:
    """Float64 slots: [sequence, time] + qpos + mocap positions + mocap quaternions."""
    return HEADER + nq + 3 * nmocap + 4 * nmocap


class MirrorPublisher:
    """Owned by the simulation process; `publish(data)` after each step (cheap: one array copy)."""

    def __init__(self, model, data):
        import mujoco

        self.nq, self.nmocap = model.nq, model.nmocap
        self.dir = Path(tempfile.mkdtemp(prefix="rojev-mirror-"))
        self.model_path = self.dir / "scene.mjb"
        mujoco.mj_saveModel(model, str(self.model_path), None)
        self.shm = shared_memory.SharedMemory(create=True, size=8 * layout(self.nq, self.nmocap))
        self.buf = np.ndarray((layout(self.nq, self.nmocap),), dtype=np.float64, buffer=self.shm.buf)
        self.buf[:] = 0.0
        self.process: subprocess.Popen | None = None
        self.publish(data)

    def publish(self, data) -> None:
        q = HEADER + self.nq
        self.buf[1] = data.time
        self.buf[HEADER:q] = data.qpos
        self.buf[q:q + 3 * self.nmocap] = data.mocap_pos.ravel()
        self.buf[q + 3 * self.nmocap:] = data.mocap_quat.ravel()
        self.buf[0] += 1.0

    def launch(self) -> None:
        mjpython = Path(sys.executable).with_name("mjpython")
        env = dict(os.environ)
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            p for p in (DYLD_DEFAULT_FALLBACK, os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")) if p)
        self.process = subprocess.Popen([str(mjpython), str(Path(__file__).resolve()), "--model", str(self.model_path),
                                         "--shm", self.shm.name], env=env)

    def close(self) -> None:
        self.buf[0] = -1.0
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        self.shm.close()
        self.shm.unlink()


def main() -> int:
    import mujoco
    import mujoco.viewer

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--shm", required=True)
    p.add_argument("--fps", type=float, default=60.0)
    a = p.parse_args()
    model = mujoco.MjModel.from_binary_path(a.model)
    data = mujoco.MjData(model)
    shm = shared_memory.SharedMemory(name=a.shm)
    resource_tracker.unregister(shm._name, "shared_memory")
    buf = np.ndarray((layout(model.nq, model.nmocap),), dtype=np.float64, buffer=shm.buf)
    q, seen = HEADER + model.nq, 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and buf[0] >= 0:
            if buf[0] != seen:
                seen = buf[0]
                data.time = buf[1]
                data.qpos[:] = buf[HEADER:q]
                data.mocap_pos[:] = buf[q:q + 3 * model.nmocap].reshape(-1, 3)
                data.mocap_quat[:] = buf[q + 3 * model.nmocap:].reshape(-1, 4)
                mujoco.mj_forward(model, data)
                viewer.sync()
            time.sleep(1.0 / a.fps)
    shm.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
