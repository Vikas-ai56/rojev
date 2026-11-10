"""A stand-in for the `libero` package (Linux-only; absent on the Mac) under LeRobot's real LIBERO env and processors.

`install()` puts `libero`, `libero.libero`, `libero.libero.benchmark` and `libero.libero.envs` into `sys.modules` with
the names LeRobot 0.6.1 imports (`lerobot/envs/libero.py:30-31`): a benchmark registry of two small suites, the
`Task` record, `get_libero_path` (rooted at `ROOT`, set per test) and `OffScreenRenderEnv`, a renderer stand-in that
records every call in order (`calls`) and returns robosuite-shaped raw observations: per-step pseudo-random camera
images of the requested size, and a fixed robot state (`EEF_POS`, `EEF_QUAT` in xyzw, `GRIPPER_QPOS`).
`check_success()` is true exactly at the step counts in `success_at`. `write_init_states(root)` saves each task's init
states as LIBERO does (torch pickles of numpy arrays); init state k is a vector filled with k, so the env records which
one was set.
"""
import sys
import types
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

SUITES = {
    "libero_goal": ("open the middle drawer of the cabinet", "put the bowl on the stove", "turn on the stove"),
    "libero_spatial": ("pick up the black bowl next to the plate and place it on the plate",),
}
INIT_STATES = 4
EEF_POS = np.array([0.1, -0.2, 0.9])
EEF_QUAT = np.array([0.5, 0.5, 0.5, 0.5])
GRIPPER_QPOS = np.array([0.03, -0.03])
ROOT = Path(".")


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def task_name(language: str) -> str:
    return language.replace(" ", "_")


def suite_class(suite: str) -> type:
    def __init__(self, task_order_index: int = 0):
        self.name = suite
        self.tasks = [Task(task_name(language), language, "Libero", suite, f"{task_name(language)}.bddl",
                           f"{task_name(language)}.pruned_init") for language in SUITES[suite]]
        self.n_tasks = len(self.tasks)

    def get_task(self, index: int) -> Task:
        return self.tasks[index]

    return type(suite.upper(), (), {"__init__": __init__, "get_task": get_task})


def get_benchmark_dict() -> dict:
    return {suite: suite_class(suite) for suite in SUITES}


def get_libero_path(key: str) -> str:
    return str(ROOT / key)


def write_init_states(root: Path) -> None:
    for suite, languages in SUITES.items():
        folder = root / "init_states" / suite
        folder.mkdir(parents=True, exist_ok=True)
        for language in languages:
            states = np.repeat(np.arange(INIT_STATES, dtype=np.float64)[:, None], 6, axis=1)
            torch.save(states, folder / f"{task_name(language)}.pruned_init")


class Controller:
    def __init__(self):
        self.use_delta = False
        self.ee_ori_mat = np.eye(3)


class Robot:
    def __init__(self):
        self.controller = Controller()


class OffScreenRenderEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple] = [("init",)]
        self.robots = [Robot()]
        self.steps = 0
        self.success_at: set[int] = set()
        self.last_raw: dict | None = None

    def seed(self, seed) -> None:
        self.calls.append(("seed", seed))

    def reset(self) -> dict:
        self.calls.append(("reset",))
        return self.raw()

    def set_init_state(self, state) -> dict:
        self.calls.append(("set_init_state", int(state[0])))
        return self.raw()

    def step(self, action):
        self.calls.append(("step", tuple(float(a) for a in action)))
        self.steps += 1
        return self.raw(), 0.0, False, {}

    def check_success(self) -> bool:
        return self.steps in self.success_at

    def close(self) -> None:
        self.calls.append(("close",))

    def raw(self) -> dict:
        rng = np.random.default_rng(self.steps)
        shape = (self.kwargs["camera_heights"], self.kwargs["camera_widths"], 3)
        self.last_raw = {
            "agentview_image": rng.integers(0, 256, shape, dtype=np.uint8),
            "robot0_eye_in_hand_image": rng.integers(0, 256, shape, dtype=np.uint8),
            "robot0_eef_pos": EEF_POS.copy(),
            "robot0_eef_quat": EEF_QUAT.copy(),
            "robot0_gripper_qpos": GRIPPER_QPOS.copy(),
            "robot0_gripper_qvel": np.zeros(2),
            "robot0_joint_pos": np.zeros(7),
            "robot0_joint_vel": np.zeros(7),
        }
        return self.last_raw


def install() -> None:
    package = types.ModuleType("libero")
    package.__path__ = []
    inner = types.ModuleType("libero.libero")
    inner.__path__ = []
    inner.get_libero_path = get_libero_path
    benchmark = types.ModuleType("libero.libero.benchmark")
    benchmark.get_benchmark_dict = get_benchmark_dict
    benchmark.Task = Task
    benchmark.Benchmark = object
    envs = types.ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = OffScreenRenderEnv
    inner.benchmark, inner.envs, package.libero = benchmark, envs, inner
    sys.modules.update({"libero": package, "libero.libero": inner, "libero.libero.benchmark": benchmark,
                        "libero.libero.envs": envs})
