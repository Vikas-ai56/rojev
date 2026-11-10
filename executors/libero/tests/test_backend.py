"""The LIBERO backend over LeRobot's real env and processor with the fake simulator: the reset protocol and its order,
the observation as LeRobot's LIBERO processing produces it (180° rotation, 8-D float32 state, camera names, task text),
sticky success, and closing."""
import math

import numpy as np
import pytest
import torch
from lerobot.envs.libero import get_libero_dummy_action

import fake_libero
from backend import LiberoBackend, to_frame
from embodiment import EMBODIMENT, START


def simulator(backend: LiberoBackend) -> fake_libero.OffScreenRenderEnv:
    return backend.env._env


def test_reset_follows_lerobots_protocol_in_order():
    backend = LiberoBackend("libero_goal", task=1, episode=2, seed=1002)
    sim = simulator(backend)
    settle = [("step", tuple(map(float, get_libero_dummy_action())))] * backend.env.num_steps_wait
    assert backend.env.num_steps_wait == 10
    assert sim.calls == [("init",), ("reset",), ("seed", 1002), ("reset",), ("set_init_state", 2), *settle]
    assert sim.robots[0].controller.use_delta is True
    assert sim.kwargs["camera_heights"] == sim.kwargs["camera_widths"] == 360
    assert sim.kwargs["control_freq"] == 20 and sim.kwargs["hard_reset"] is True
    assert sim.kwargs["bddl_file_name"].endswith("libero_goal/put_the_bowl_on_the_stove.bddl")


def test_episodes_past_the_init_states_wrap_around_as_in_lerobot():
    backend = LiberoBackend("libero_goal", task=0, episode=fake_libero.INIT_STATES + 1, seed=1000)
    assert ("set_init_state", 1) in simulator(backend).calls


def test_the_observation_is_lerobots_libero_processing_of_the_raw_frames():
    backend = LiberoBackend("libero_goal", task=0, episode=0, seed=1000)
    raw = simulator(backend).last_raw
    obs = backend.observe()
    assert tuple(obs.images) == EMBODIMENT.cameras
    for camera, source in (("top", "agentview_image"), ("wrist", "robot0_eye_in_hand_image")):
        assert obs.images[camera].dtype == np.uint8 and obs.images[camera].shape == (360, 360, 3)
        assert np.array_equal(obs.images[camera], raw[source][::-1, ::-1])
    assert obs.state.dtype == np.float32 and obs.state.shape == (8,)
    angle = 2 * math.pi / 3
    expected = [*fake_libero.EEF_POS, *[angle / math.sqrt(3)] * 3, *fake_libero.GRIPPER_QPOS]
    assert np.allclose(obs.state, expected, atol=1e-6)
    assert obs.task == backend.instruction == "open the middle drawer of the cabinet"


def test_frames_survive_the_float_round_trip_exactly():
    every_value = torch.arange(256, dtype=torch.float32).reshape(1, 1, 16, 16).repeat(1, 3, 1, 1) / 255
    assert np.array_equal(to_frame(every_value)[..., 0], np.arange(256, dtype=np.uint8).reshape(16, 16))


def test_a_step_applies_the_action_and_returns_the_new_observation():
    backend = LiberoBackend("libero_goal", task=0, episode=0, seed=1000)
    action = np.array([0.1, -0.2, 0.3, 0.0, 0.05, -0.05, 1.0])
    obs = backend.step(action)
    sim = simulator(backend)
    assert sim.calls[-1] == ("step", tuple(action))
    assert np.array_equal(obs.images["top"], sim.last_raw["agentview_image"][::-1, ::-1])
    assert backend.observe() is obs and backend.steps == 1


def test_success_is_sticky_and_records_when_it_first_held():
    backend = LiberoBackend("libero_goal", task=2, episode=0, seed=1000)
    sim = simulator(backend)
    sim.success_at = {backend.env.num_steps_wait + 3}
    for _ in range(2):
        backend.step(START)
    assert not backend.success and backend.success_t is None
    for _ in range(3):
        backend.step(START)
    assert backend.success and backend.success_step == 3
    assert backend.success_t == pytest.approx(3 / 20)


def test_budget_and_rate_come_from_lerobot():
    goal = LiberoBackend("libero_goal", task=0, episode=0, seed=1000)
    spatial = LiberoBackend("libero_spatial", task=0, episode=0, seed=1000)
    assert (goal.max_steps, spatial.max_steps) == (300, 280)
    assert goal.control_hz == EMBODIMENT.spec.control_hz


def test_close_closes_the_simulator():
    backend = LiberoBackend("libero_goal", task=0, episode=0, seed=1000)
    sim = simulator(backend)
    backend.close()
    assert sim.calls[-1] == ("close",) and backend.env._env is None


def test_every_simulator_call_runs_on_the_backends_own_thread_whoever_steps_it():
    """MuJoCo's EGL context stays current on the thread that rendered last, so build, steps from another thread
    (the real-time control loop) and close must all reach the simulator on one thread."""
    import threading
    seen = []
    original = fake_libero.OffScreenRenderEnv.step

    def recording_step(env, action):
        seen.append(threading.current_thread().name)
        return original(env, action)

    fake_libero.OffScreenRenderEnv.step = recording_step
    try:
        backend = LiberoBackend("libero_goal", task=0, episode=0, seed=1000)
        stepper = threading.Thread(target=backend.step, args=(START,), name="control-loop")
        stepper.start()
        stepper.join()
        sim = simulator(backend)
        backend.close()
    finally:
        fake_libero.OffScreenRenderEnv.step = original
    assert len(set(seen)) == 1 and seen[0].startswith("libero-sim") and len(seen) == backend.env.num_steps_wait + 1
    assert sim.calls[-1] == ("close",)

