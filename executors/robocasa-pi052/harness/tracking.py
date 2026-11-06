"""Object-tracking measurement against simulator truth (executor side; moves nothing in the robot loop).

Simulator truth: `truth_masks` renders the simulator's per-camera segmentation and returns the pixels of the task's
object (its own geoms), its container fixture and the gripper. Geometry: `central_pixel` (a point on a mask),
`iou`, `filled` / `inside_share` (2D "inside": the share of object pixels inside the container's filled outline).

`FrameRecorder` (live, `realtime.py --record-frames S`): on its own thread, every S seconds of simulated time (the
capture clock, so lockstep runs are sampled evenly too), saves the newest capture's
camera images and the simulator state read between two steps (`sim.get_state().flatten()`). Nothing is rendered
during the run: the offscreen renderer returns blank images inside the live process (CGL context shared by the
control loop's renders), and rendering there would also steal control-loop time. `sam_check.py track` restores each
state in a fresh process, renders the truth masks and streams the recorded images through SAM 2.1 in order, so SAM
sees exactly the frames the live run saw; the state is read at most one control step after the capture.
"""
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rojev.protocol import Capture

OBJECT = {"MicrowaveThawingFridge": "meat", "MoveFridgeToFreezer": "refrigerated_item"}
CONTAINER = {"MicrowaveThawingFridge": lambda k: k.microwave.name, "MoveFridgeToFreezer": lambda k: k.fridge.name}
TRACKED = ("object", "container")
GEOM = 5
MIN_PIXELS = 4


def truth_masks(k, task: str, cameras: list[str], size: tuple[int, int]) -> dict[str, np.ndarray]:
    """Per camera (`<camera>:<role>`), the pixels of the task's object (its own geoms; a name prefix would also match
    "meat_container"), its container fixture and the gripper (geom-name prefixes)."""
    model = k.sim.model
    names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
    obj = k.objects[OBJECT[task]]
    own = set(obj.contact_geoms) | set(obj.visual_geoms)
    prefixes = {"container": f"{CONTAINER[task](k)}_", "gripper": "gripper0_"}
    ids = {"object": np.array([i for i, n in enumerate(names) if n in own])}
    ids |= {role: np.array([i for i, n in enumerate(names) if n.startswith(pre)]) for role, pre in prefixes.items()}
    out = {}
    for cam in cameras:
        seg = k.sim.render(camera_name=cam, width=size[1], height=size[0], segmentation=True)[::-1]
        geom = np.where(seg[..., 0] == GEOM, seg[..., 1], -1)
        out |= {f"{cam}:{role}": np.isin(geom, members) for role, members in ids.items()}
    return out


def central_pixel(mask: np.ndarray) -> tuple[float, float]:
    """The mask pixel nearest the mask's centroid (x, y), so the point lies on the object even when it is concave."""
    ys, xs = np.nonzero(mask)
    k = np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    return float(xs[k]), float(ys[k])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def filled(mask: np.ndarray) -> np.ndarray:
    """The container's outline filled: pixels between its leftmost and rightmost pixel in their row and between its
    top and bottom pixel in their column (an open appliance's interior shows through its walls, not as its pixels)."""
    rows = np.zeros_like(mask)
    cols = np.zeros_like(mask)
    for y in np.nonzero(mask.any(axis=1))[0]:
        xs = np.nonzero(mask[y])[0]
        rows[y, xs[0]:xs[-1] + 1] = True
    for x in np.nonzero(mask.any(axis=0))[0]:
        ys = np.nonzero(mask[:, x])[0]
        cols[ys[0]:ys[-1] + 1, x] = True
    return rows & cols


def inside_share(obj: np.ndarray, container: np.ndarray) -> float:
    return float((obj & filled(container)).sum() / obj.sum()) if obj.sum() else 0.0


POLL_S = 0.005


class FrameRecorder:
    def __init__(self, latest: Callable[[], Capture], read: Callable, cameras: list[str], folder: Path,
                 every_s: float = 1.0):
        self.latest, self.read, self.cameras, self.folder, self.every_s = latest, read, cameras, folder, every_s
        self.times: list[float] = []
        self.obs_ids: list[int] = []
        self.states: list[np.ndarray] = []
        self.images: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="frame-recorder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def record_once(self) -> None:
        capture = self.latest()
        state = self.read(lambda k: k.sim.get_state().flatten().copy())
        self.times.append(capture.t)
        self.obs_ids.append(capture.obs_id)
        self.states.append(state)
        self.images.append(np.stack([capture.observation.images[c] for c in self.cameras]))

    def save(self) -> None:
        """<folder>/frames.npz: t, obs_id, state (flattened MjSim state) and images (frame, camera, H, W, 3)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.folder / "frames.npz", t=np.array(self.times), obs_id=np.array(self.obs_ids),
                            state=np.array(self.states), images=np.array(self.images),
                            cameras=np.array(self.cameras))

    def _run(self) -> None:
        due = self.latest().t + self.every_s
        while not self._stop.wait(POLL_S):
            if self.latest().t >= due:
                self.record_once()
                due += self.every_s
