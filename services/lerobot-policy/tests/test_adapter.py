"""The LeRobot adapter without LeRobot or torch: stand-ins play a LeRobot policy (`predict_action_chunk`, `reset`)
and its pre / post processor pipelines, behind a numpy `TensorBridge`; the features are the GR00T N1.7 LIBERO
checkpoints' (two 256×256 images, an 8-wide state, a 7-wide action) and the config is the committed
config/s1/groot-libero-goal.yaml. Covered: the batch LeRobot receives (keys, shapes, dtypes, camera routing, text),
the subtask overriding the task, the chunk and its stage timings, loud refusals of a config the checkpoint cannot be
fed by and of observations it cannot take, non-finite actions, reset, and a round trip through our service, wire and
transport."""
import dataclasses
import threading
from contextlib import nullcontext

import numpy as np
import pytest

from adapter import Feature, LeRobotPolicy, TensorBridge, check_features, compile_submodules, observation_batch
from model_config import ROOT, load_model_config
from rojev.protocol import Observation
from rojev.system1 import PolicyService, RemotePolicy
from rojev.transport import Endpoint, HttpServer
from serve import MAX_BODY, about

TOKEN = "t0ken"
HORIZON, ACTION_DIM = 16, 7
CONFIG = load_model_config(ROOT / "config/s1/groot-libero-goal.yaml")
INPUTS = {"observation.images.wrist_image": Feature("VISUAL", (256, 256, 3)),
          "observation.images.image": Feature("VISUAL", (256, 256, 3)),
          "observation.state": Feature("STATE", (8,))}
OUTPUTS = {"action": Feature("ACTION", (ACTION_DIM,))}
NUMPY = TensorBridge(from_numpy=np.asarray, to_numpy=np.asarray, inference=nullcontext, synchronize=lambda: None)


class StandInPipeline:
    def __init__(self, transform=lambda value: value):
        self.transform, self.calls, self.resets = transform, [], 0

    def __call__(self, value):
        self.calls.append(value)
        return self.transform(value)

    def reset(self) -> None:
        self.resets += 1


class StandInLeRobotPolicy:
    def __init__(self, actions: np.ndarray | None = None):
        self.actions = actions if actions is not None else np.full((1, HORIZON, ACTION_DIM), 0.25, np.float32)
        self.inputs, self.resets = [], 0

    def predict_action_chunk(self, batch: dict) -> np.ndarray:
        self.inputs.append(batch)
        return self.actions

    def reset(self) -> None:
        self.resets += 1


def obs(task: str = "put the bowl on the plate") -> Observation:
    images = {"top": np.full((128, 160, 3), 51, np.uint8), "wrist": np.full((128, 160, 3), 204, np.uint8)}
    return Observation(images, np.arange(8, dtype=np.float64) / 10, task)


def policy(actions: np.ndarray | None = None) -> tuple[LeRobotPolicy, StandInPipeline, StandInLeRobotPolicy,
                                                        StandInPipeline]:
    pre, model, post = StandInPipeline(), StandInLeRobotPolicy(actions), StandInPipeline(lambda chunk: chunk * 2)
    return LeRobotPolicy(CONFIG, model, pre, post, INPUTS, NUMPY, "groot"), pre, model, post


def test_batch_carries_the_checkpoint_keys_in_lerobots_convention():
    adapter, pre, model, _ = policy()
    adapter.act(obs(), "")
    batch = pre.calls[-1]
    assert set(batch) == {"observation.images.image", "observation.images.wrist_image", "observation.state", "task"}
    for key, value in (("observation.images.image", 0.2), ("observation.images.wrist_image", 0.8)):
        assert batch[key].dtype == np.float32 and batch[key].shape == (1, 3, 128, 160)
        assert batch[key] == pytest.approx(np.full((1, 3, 128, 160), value), abs=1e-6)
    assert batch["observation.state"].dtype == np.float32 and batch["observation.state"].shape == (1, 8)
    assert batch["observation.state"][0] == pytest.approx(np.arange(8) / 10)
    assert batch["task"] == ["put the bowl on the plate"]
    assert model.inputs[-1] is batch


def test_the_subtask_text_overrides_the_task():
    adapter, pre, _, _ = policy()
    adapter.act(obs(), "open the drawer")
    assert pre.calls[-1]["task"] == ["open the drawer"]


def test_the_chunk_is_postprocessed_float32_time_by_action_width_with_stage_timings():
    adapter, _, _, post = policy()
    chunk = adapter.act(obs(), "")
    assert chunk.actions.shape == (HORIZON, ACTION_DIM) and chunk.actions.dtype == np.float32
    assert chunk.actions == pytest.approx(np.full((HORIZON, ACTION_DIM), 0.5))
    assert post.calls and set(chunk.timings_ms) == {"pre", "model", "post"}
    assert chunk.execute_steps == CONFIG.execute_steps


def test_reset_resets_the_policy_and_both_pipelines():
    adapter, pre, model, post = policy()
    adapter.reset()
    assert (model.resets, pre.resets, post.resets) == (1, 1, 1)


def test_the_committed_camera_map_fits_the_checkpoint():
    check_features(CONFIG, INPUTS, OUTPUTS)


@pytest.mark.parametrize("cameras, state, action_dim, inputs, message", [
    ({"top": "observation.images.image", "wrist": "observation.images.wrist"}, "observation.state", 7, INPUTS,
     "'observation.images.wrist'"),
    ({"top": "observation.images.image"}, "observation.state", 7, INPUTS, "no camera or state of this config feeds"),
    ({"top": "observation.images.image", "wrist": "observation.state"}, "observation.state", 7, INPUTS,
     "not an image input"),
    (CONFIG.cameras, "observation.proprio", 7, INPUTS, "not a state input"),
    (CONFIG.cameras, "observation.state", 6, INPUTS, "config expects (6,)"),
    (CONFIG.cameras, "observation.state", 7, {**INPUTS, "observation.environment_state": Feature("ENV", (10,))},
     "ENV input 'observation.environment_state'"),
])
def test_a_config_the_checkpoint_cannot_be_fed_by_fails_loudly(cameras, state, action_dim, inputs, message):
    config = dataclasses.replace(CONFIG, cameras=cameras, state=state, action_dim=action_dim)
    with pytest.raises(ValueError, match="does not fit checkpoint") as error:
        check_features(config, inputs, OUTPUTS)
    assert message in str(error.value)


@pytest.mark.parametrize("observation, message", [
    (Observation({"top": np.zeros((8, 8, 3), np.uint8)}, np.zeros(8), "t"), "lacks cameras \\['wrist'\\]"),
    (Observation({"top": np.zeros((8, 8, 3), np.uint8), "wrist": np.zeros((8, 8, 3), np.uint8)}, np.zeros(7), "t"),
     "state has shape"),
    (Observation({"top": np.zeros((8, 8, 3), np.float32), "wrist": np.zeros((8, 8, 3), np.uint8)}, np.zeros(8), "t"),
     "H×W×3 uint8"),
])
def test_an_observation_the_checkpoint_cannot_take_is_refused(observation, message):
    with pytest.raises(ValueError, match=message):
        observation_batch(observation, "t", CONFIG, INPUTS)


def test_non_finite_or_misshaped_actions_fail_the_request():
    with pytest.raises(ValueError, match="non-finite"):
        policy(np.full((1, HORIZON, ACTION_DIM), np.nan, np.float32))[0].act(obs(), "")
    with pytest.raises(ValueError, match="expected \\(1, T, 7\\)"):
        policy(np.zeros((1, HORIZON, 6), np.float32))[0].act(obs(), "")
    with pytest.raises(ValueError, match="execute_steps 8 outside 1..4"):
        policy(np.zeros((1, 4, ACTION_DIM), np.float32))[0].act(obs(), "")


def serve(adapter: LeRobotPolicy) -> HttpServer:
    service = PolicyService(adapter, about(CONFIG, adapter.policy_type), name="lerobot-policy")
    server = HttpServer(Endpoint(service.handle, TOKEN, MAX_BODY), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_round_trip_through_service_wire_and_transport():
    adapter, pre, _, _ = policy()
    server = serve(adapter)
    try:
        remote = RemotePolicy(f"http://127.0.0.1:{server.server_port}", TOKEN, codec="raw")
        chunk = remote.act(obs(), "open the drawer")
    finally:
        server.shutdown()
    assert chunk.actions.shape == (HORIZON, ACTION_DIM)
    assert chunk.actions == pytest.approx(np.full((HORIZON, ACTION_DIM), 0.5))
    assert pre.calls[-1]["task"] == ["open the drawer"]
    assert pre.calls[-1]["observation.images.wrist_image"][0, 0, 0, 0] == pytest.approx(0.8)


def test_health_reports_the_pinned_checkpoint_overrides_and_camera_map():
    health = about(CONFIG, "groot")
    assert health["checkpoint"] == "nvidia/gr00t17-lerobot-libero_goal-640" and len(health["revision"]) == 40
    assert health["overrides"]["base_model_path"].startswith("nvidia/GR00T-N1.7-3B@")
    assert health["cameras"] == {"top": "observation.images.image", "wrist": "observation.images.wrist_image"}


def test_the_goal_config_compiles_the_action_head_and_warms_up_before_serving():
    assert CONFIG.compile_modules == ("_groot_model.action_head.model",) and CONFIG.compile_mode == "max-autotune-no-cudagraphs"
    assert CONFIG.warmup == 3


def test_compile_replaces_only_the_named_submodule():
    class Node:
        pass

    root, head = Node(), Node()
    root._groot_model, root._groot_model.action_head = Node(), head
    head.model, head.other = "dit", "kept"
    compile_submodules(root, ("_groot_model.action_head.model",), "max-autotune",
                       lambda module, mode: f"compiled({module},{mode})")
    assert head.model == "compiled(dit,max-autotune)" and head.other == "kept"


def test_warm_up_runs_synthetic_requests_of_the_checkpoint_shapes_then_resets():
    p, pre, model, post = policy()
    wall_ms = p.warm_up(2)
    assert len(wall_ms) == 2 and len(model.inputs) == 2 and model.resets == 1 and pre.resets == 1
    first = pre.calls[0]
    assert first["observation.images.image"].shape == (1, 3, 256, 256) and first["task"] == ["warm up"]
    assert pre.calls[1]["task"] == ["warm up warm up"]
    assert first["observation.state"].shape == (1, 8)
