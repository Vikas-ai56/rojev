"""The GR00T adapter without the GPU model. A stand-in plays `Gr00tPolicy`: it reports a modality config like a
fine-tuned checkpoint's and validates requests the way GR00T's strict mode does (keys, (B, T, ...) shapes, dtypes).
Covered: the binding built from that config and our layout (and its refusals), the observation dict (cameras, LeRobot
units, the model's own language key), actions back to radians, and a round trip through our service, wire and
transport."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from groot_io import Binding, layout
from rojev.protocol import Observation
from rojev.robot.so101 import lerobot_units
from rojev.system1 import PolicyService, RemotePolicy
from rojev.transport import Endpoint, HttpServer
from serve import MAX_BODY, GrootPolicy

TOKEN = "t0ken"
HORIZON = 16
UNITS = lerobot_units()
WHOLE_TASK = "annotation.human.task_description"


def config(language: str = WHOLE_TASK, frames: int = 1, cameras=("front", "wrist")) -> dict:
    keys = ["single_arm", "gripper"]
    return {"video": SimpleNamespace(modality_keys=list(cameras), delta_indices=list(range(frames))),
            "state": SimpleNamespace(modality_keys=keys, delta_indices=[0]),
            "language": SimpleNamespace(modality_keys=[language], delta_indices=[0]),
            "action": SimpleNamespace(modality_keys=keys, delta_indices=list(range(HORIZON)))}


class StandInGroot:
    def __init__(self, modality: dict | None = None, action: dict | None = None):
        self.modality = modality or config()
        self.requests: list[dict] = []
        self.action = action
        self.resets = 0

    def get_modality_config(self) -> dict:
        return self.modality

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        for key in self.modality["video"].modality_keys:
            video = observation["video"][key]
            assert video.dtype == np.uint8 and video.ndim == 5 and video.shape[1] == 1 and video.shape[-1] == 3
        for key in self.modality["state"].modality_keys:
            state = observation["state"][key]
            assert state.dtype == np.float32 and state.ndim == 3 and state.shape[:2] == (1, 1)
        for key in self.modality["language"].modality_keys:
            assert isinstance(observation["language"][key], list) and len(observation["language"][key][0]) == 1
        self.requests.append(observation)
        if self.action is not None:
            return self.action, {}
        return {key: np.repeat(observation["state"][key], HORIZON, axis=1) for key in observation["state"]}, {}

    def reset(self) -> dict:
        self.resets += 1
        return {}


def obs(task: str = "put the cube in the bowl") -> Observation:
    images = {"front": np.full((48, 64, 3), 10, np.uint8), "wrist_cam": np.full((48, 64, 3), 20, np.uint8)}
    return Observation(images, np.array([0.1, -0.2, 0.3, -0.4, 0.5, 0.2], np.float32), task)


def test_binding_reads_keys_from_the_model_and_slices_and_cameras_from_our_layout():
    binding = Binding.of(config(), layout())
    assert binding.cameras == {"front": "front", "wrist": "wrist_cam"}
    assert binding.state == {"single_arm": (0, 5), "gripper": (5, 6)} and binding.joints == 6
    assert binding.language == WHOLE_TASK
    request = binding.observation(obs(), "pick up the cube", UNITS)
    assert request["video"]["front"].shape == (1, 1, 48, 64, 3) and request["video"]["wrist"][0, 0, 0, 0, 0] == 20
    state = np.concatenate([request["state"]["single_arm"][0, 0], request["state"]["gripper"][0, 0]])
    assert state == pytest.approx(UNITS.to_lerobot(obs().state), abs=1e-4)
    assert request["language"] == {WHOLE_TASK: [["pick up the cube"]]}


def test_a_step_trained_checkpoint_gets_the_text_under_sub_task():
    request = Binding.of(config(language="sub_task"), layout()).observation(obs(), "open the gripper", UNITS)
    assert request["language"] == {"sub_task": [["open the gripper"]]}


@pytest.mark.parametrize("modality, message", [
    (config(frames=2), "takes 2 video frames"),
    (config(cameras=("front", "top")), "camera key 'top'"),
])
def test_a_model_the_layout_cannot_feed_is_refused_at_load(modality, message):
    with pytest.raises(ValueError, match=message):
        Binding.of(modality, layout())


def test_layout_errors_are_refused_at_load():
    two_languages = config()
    two_languages["language"] = SimpleNamespace(modality_keys=["task", "sub_task"], delta_indices=[0])
    with pytest.raises(ValueError, match="one language key"):
        Binding.of(two_languages, layout())
    gap = layout()
    gap["action"]["gripper"] = {"start": 4, "end": 5}
    with pytest.raises(ValueError, match="cover joints"):
        Binding.of(config(), gap)


def test_actions_return_in_radians_and_non_finite_values_are_refused():
    binding = Binding.of(config(), layout())
    target = UNITS.to_lerobot(np.tile(obs().state, (HORIZON, 1)))
    actions = binding.actions({"single_arm": target[None, :, :5], "gripper": target[None, :, 5:]}, UNITS)
    assert actions.shape == (HORIZON, 6) and actions.dtype == np.float32
    assert actions == pytest.approx(np.tile(obs().state, (HORIZON, 1)), abs=1e-5)
    target[3, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        binding.actions({"single_arm": target[None, :, :5], "gripper": target[None, :, 5:]}, UNITS)


def test_policy_sends_the_subtask_else_the_task_resets_the_model_and_takes_no_continuation():
    model = StandInGroot()
    policy = GrootPolicy(model, UNITS)
    assert not policy.accepts_continuation
    chunk = policy.act(obs(), "open the gripper")
    assert model.requests[-1]["language"][WHOLE_TASK] == [["open the gripper"]]
    assert set(chunk.timings_ms) == {"preprocess", "infer", "postprocess"}
    policy.act(obs("stack the blocks"), "")
    assert model.requests[-1]["language"][WHOLE_TASK] == [["stack the blocks"]]
    policy.reset()
    assert model.resets == 1


def serve(model: StandInGroot) -> HttpServer:
    service = PolicyService(GrootPolicy(model, UNITS), {"policy": "stand-in"}, name="groot")
    server = HttpServer(Endpoint(service.handle, TOKEN, MAX_BODY), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_round_trip_through_service_wire_and_transport():
    model = StandInGroot()
    server = serve(model)
    try:
        chunk = RemotePolicy(f"http://127.0.0.1:{server.server_port}", TOKEN, codec="raw").act(obs(), "pick")
    finally:
        server.shutdown()
    assert chunk.actions.shape == (HORIZON, 6) and np.isfinite(chunk.actions).all()
    assert chunk.actions == pytest.approx(np.tile(obs().state, (HORIZON, 1)), abs=1e-5)
    assert model.requests[-1]["video"]["front"][0, 0, 0, 0, 0] == 10


def test_a_non_finite_model_answer_fails_the_request_on_the_service():
    bad = {"single_arm": np.full((1, HORIZON, 5), np.nan, np.float32), "gripper": np.zeros((1, HORIZON, 1), np.float32)}
    server = serve(StandInGroot(action=bad))
    try:
        with pytest.raises(ConnectionError, match="HTTP 500"):
            RemotePolicy(f"http://127.0.0.1:{server.server_port}", TOKEN, codec="raw").act(obs(), "pick")
    finally:
        server.shutdown()
