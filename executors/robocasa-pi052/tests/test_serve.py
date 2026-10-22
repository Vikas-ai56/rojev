"""serve.py without the GPU model: `Service` routing behind the transport `Endpoint` with a stand-in policy (K samples in
one request, one policy call), and `Draws` (the K-row batch the PI052 adapter's prompt is repeated into)."""
import io
import json
import threading

import numpy as np
import torch

from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy, RemotePolicy, wire
from rojev.transport import Endpoint, HttpServer
from serve import MAX_BODY, Draws, Service

TOKEN = "t0ken"


class StandIn(Policy):
    def __init__(self):
        self.calls: list[tuple[str, int]] = []

    def act(self, obs: Observation, subtask: str) -> ActionChunk:
        raise AssertionError("the service asks for samples, never single chunks")

    def act_samples(self, obs: Observation, subtask: str, k: int) -> list[ActionChunk]:
        self.calls.append((subtask, k))
        return [ActionChunk(np.full((50, 12), i, np.float32), {"infer": 2.0}, {"subtask": subtask}) for i in range(k)]

    def reset(self) -> None:
        self.calls.append(("reset", 0))


def obs():
    return Observation({"cam": np.zeros((8, 8, 3), np.uint8)}, np.zeros(16, np.float32), "task")


def test_service_routes_k_samples_reset_health_and_refuses_another_wire_version():
    policy = StandIn()
    service = Service(policy, {"policy": "stand-in"})
    status, body, _ = service.handle("POST", "/act", {}, wire.encode_request(obs(), "pick", 3))
    chunks = wire.decode_response(body)
    assert status == 200 and [c.actions[0, 0] for c in chunks] == [0, 1, 2] and policy.calls == [("pick", 3)]
    assert service.handle("POST", "/reset", {}, b"")[:2] == (200, b"ok") and policy.calls[-1] == ("reset", 0)
    status, body, _ = service.handle("GET", "/health", {}, b"")
    assert status == 200 and json.loads(body) == {"policy": "stand-in", "requests": 2}
    legacy = io.BytesIO()
    np.savez_compressed(legacy, state=np.zeros(16), task=np.array("t"), subtask=np.array("pick"))
    assert service.handle("POST", "/act", {}, legacy.getvalue())[:2] == (400, b"wire version 1, expected 2")
    assert service.handle("POST", "/nope", {}, b"")[0] == 404


def test_remote_k_samples_reach_the_policy_as_one_call_through_the_endpoint():
    policy = StandIn()
    server = HttpServer(Endpoint(Service(policy, {}).handle, TOKEN, MAX_BODY), "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    remote = RemotePolicy(f"http://127.0.0.1:{server.server_port}", TOKEN)
    chunks = remote.act_samples(obs(), "", 5)
    assert len(chunks) == 5 and policy.calls == [("", 5)] and all(c.info == {"subtask": ""} for c in chunks)
    assert remote.act(obs(), "pick").actions.shape == (50, 12) and policy.calls[-1] == ("pick", 1)
    server.shutdown()


def test_draws_repeat_every_conditioned_tensor_k_times_along_the_batch():
    seen = {}

    class Model:
        config = object()

        def predict_action_chunk(self, batch):
            seen.update(batch)
            return torch.randn(batch["observation.state"].shape[0], 50, 12)

    batch = {"observation.images.front": torch.rand(1, 3, 4, 4), "observation.state": torch.rand(1, 16),
             "observation.language.tokens": torch.ones(1, 7, dtype=torch.long), "task": ["t"]}
    draws = Draws(Model(), 4)
    out = draws.predict_action_chunk(batch)
    assert out.shape == (4, 50, 12) and draws.config is Model.config and seen["task"] == ["t"]
    for key in ("observation.images.front", "observation.state", "observation.language.tokens"):
        assert seen[key].shape[0] == 4 and all(torch.equal(row, batch[key][0]) for row in seen[key])
