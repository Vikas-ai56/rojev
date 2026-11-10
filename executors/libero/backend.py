"""LIBERO as a `rojev.control.Backend`: one episode of one task, built, reset and observed as LeRobot 0.6.1 evaluates it.

Env: LeRobot's own `LiberoEnv` (`lerobot/envs/libero.py:107`), given the arguments `_make_env_fns` gives it
(`envs/libero.py:419-434`) with the values of LeRobot's LIBERO config (`envs/configs.py:322-451`: pixels_agent_pos
observations, 360×360 cameras, 20 Hz, relative control, hard reset, init states on). Episode e of a task is
lerobot-eval's e-th rollout of that task with one env: init state e (`init_state_id = episode_index`,
`envs/libero.py:175, 345`) and seed `start seed + e` (`scripts/lerobot_eval.py:531-533`); the caller passes that seed.
The constructor resets with LeRobot's protocol (`envs/libero.py:339-364`): seed, reset, set_init_state, 10 settle steps
of the no-op action, OSC `use_delta`. LeRobot's env never autoresets: a step after success continues the simulation.
The loop steps it (no clock of its own).

Threads: every simulator call (build, reset, step, close) runs on one thread the backend owns (`libero-sim`). MuJoCo's
offscreen EGL context is made current by each render and never released (robosuite's `MjRenderContext`), and an EGL
context current on one thread cannot be made current on another; in real time the control loop steps from its own
thread, so without this the first render after the build would fail. Lockstep runs the same way.

Observation: lerobot-eval's path for a batch of one env. The raw observation is batched as gymnasium's vector env
batches it (`create_empty_array` + `concatenate` over the env's observation space, `vector/sync_vector_env.py:172,
269`), then `preprocess_observation` (`envs/utils.py:68`) and the config's env processor (`LiberoProcessorStep`,
`processor/env_processor.py:26-153`: both images rotated 180°, state = eef position (3) + axis-angle of the eef
quaternion (3) + gripper qpos (2), float32) run on it. The processed images, floats in [0, 1], become the uint8
H×W×3 frames they came from again (exact: v / 255 · 255 rounds back to v), so a service that applies LeRobot's policy
preprocessor to them computes on the inputs lerobot-eval would give the policy. Cameras: `top` = agentview, `wrist` =
eye-in-hand, each found under LeRobot's own camera name mapping. `task` = the task's language.

Success: LeRobot's `info["is_success"]` (= `env.check_success()`, `envs/libero.py:376`) after every step; sticky once
true, with the step and the robot time (steps / control rate) at which it first held.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from gymnasium.vector.utils import concatenate, create_empty_array
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

from rojev.protocol import Observation

from suites import load_suite

CAMERAS = {"top": "agentview_image", "wrist": "robot0_eye_in_hand_image"}


def to_frame(image: torch.Tensor) -> np.ndarray:
    return (image[0] * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().numpy()


class LiberoBackend:
    def __init__(self, suite: str, task: int, episode: int, seed: int):
        self.suite, self.task, self.episode, self.seed = suite, task, episode, seed
        self._sim_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="libero-sim")
        try:
            self._on_sim_thread(self._open)
        except BaseException:
            self._sim_thread.shutdown()
            raise

    def _on_sim_thread(self, call, *args):
        return self._sim_thread.submit(call, *args).result()

    def _open(self) -> None:
        config = LiberoConfig(task=self.suite)
        self.env = LiberoEnv(task_suite=load_suite(self.suite), task_id=self.task, task_suite_name=self.suite,
                             camera_name=config.camera_name, init_states=config.init_states,
                             episode_length=config.episode_length, episode_index=self.episode, n_envs=1,
                             control_mode=config.control_mode, camera_name_mapping=config.camera_name_mapping,
                             is_libero_plus=config.is_libero_plus, **config.gym_kwargs)
        self.env_preprocessor = config.get_env_processors()[0]
        self.instruction: str = self.env.task_description
        self.max_steps: int = self.env._max_episode_steps
        self.control_hz: float = float(self.env.control_freq)
        self.steps = 0
        self.success = False
        self.success_step: int | None = None
        raw, _ = self.env.reset(seed=self.seed)
        self._latest = self._observation(raw)

    @property
    def success_t(self) -> float | None:
        return None if self.success_step is None else self.success_step / self.control_hz

    def observe(self) -> Observation:
        return self._latest

    def step(self, action) -> Observation:
        return self._on_sim_thread(self._step, action)

    def _step(self, action) -> Observation:
        raw, _, _, _, info = self.env.step(np.asarray(action, dtype=np.float64))
        self.steps += 1
        if info["is_success"] and not self.success:
            self.success, self.success_step = True, self.steps
        self._latest = self._observation(raw)
        return self._latest

    def close(self) -> None:
        try:
            self._on_sim_thread(self.env.close)
        finally:
            self._sim_thread.shutdown()

    def _observation(self, raw: dict) -> Observation:
        space = self.env.observation_space
        batch = preprocess_observation(concatenate(space, [raw], create_empty_array(space, n=1, fn=np.zeros)))
        batch["task"] = [self.instruction]
        processed = self.env_preprocessor(batch)
        images = {name: to_frame(processed[f"{OBS_IMAGES}.{self.env.camera_name_mapping[source]}"])
                  for name, source in CAMERAS.items()}
        return Observation(images=images, state=processed[OBS_STATE][0].numpy(), task=self.instruction)
