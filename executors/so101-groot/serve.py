"""GR00T N1.7 fine-tuned for the SO-101 as a remote System 1 policy: `rojev.system1.PolicyService` on the GPU box.

Loads the checkpoint with Isaac-GR00T's own `Gr00tPolicy` (embodiment `NEW_EMBODIMENT`; modality config and
normalization statistics come from the fine-tuned checkpoint) and binds it to our contract with `groot_io.Binding`,
built from the checkpoint's `get_modality_config()` and our dataset layout (`modality.json`). Served behind our wire
and `Endpoint` (token, body limit, replay cache) rather than GR00T's ZeroMQ `PolicyServer`: one transport for every S1
model, retried requests answered from the replay cache, and a timed-out request simply abandoned (a ZeroMQ REQ socket
must receive before it may send again, so the runner's deadlines would wedge it until the socket is rebuilt).

Per /act request: `Binding.observation` (text = the subtask if given, else the task) → `Gr00tPolicy.get_action`
(validated with `--strict`, the default) → `Binding.actions` (absolute targets back in radians, finite or the request
fails). Stateless, so a retried request answers the same. `accepts_continuation` is False: GR00T's policy path has
no real-time chunking. `gr00t` is imported only when the model loads, so the adapter is testable without a GPU.

  ROJEV_TOKEN=… uv run python serve.py --model <fine-tuned checkpoint dir> --port 8005
"""
import argparse
import os
import time

from groot_io import Binding, layout
from rojev.protocol import Observation
from rojev.robot.so101 import LeRobotUnits, lerobot_units
from rojev.system1 import ActionChunk, Policy, PolicyService, serve_policy
from rojev.transport import Endpoint

MAX_BODY = 32 * 1024 * 1024


class GrootPolicy(Policy):
    def __init__(self, model, units: LeRobotUnits):
        self.model, self.units = model, units
        self.binding = Binding.of(model.get_modality_config(), layout())

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        t0 = time.perf_counter()
        request = self.binding.observation(obs, subtask or obs.task, self.units)
        t1 = time.perf_counter()
        action, _ = self.model.get_action(request)
        t2 = time.perf_counter()
        actions = self.binding.actions(action, self.units)
        t3 = time.perf_counter()
        return ActionChunk(actions, {"preprocess": round((t1 - t0) * 1000, 1), "infer": round((t2 - t1) * 1000, 1),
                                     "postprocess": round((t3 - t2) * 1000, 1)})

    def reset(self) -> None:
        self.model.reset()


def load(model_path: str, device: str, strict: bool):
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy import Gr00tPolicy

    return Gr00tPolicy(model_path=model_path, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, device=device,
                       strict=strict)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="fine-tuned GR00T N1.7 checkpoint directory")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=8005)
    p.add_argument("--idle-min", type=float, default=30)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True,
                   help="GR00T's own observation / action validation (policy.md: on while developing)")
    a = p.parse_args()
    policy = GrootPolicy(load(a.model, a.device, a.strict), lerobot_units())
    service = PolicyService(policy, {"policy": "GR00T N1.7 (SO-101 fine-tune)", "checkpoint": a.model,
                                     "device": a.device, "language_key": policy.binding.language}, name="groot")
    serve_policy(Endpoint(service.handle, os.environ["ROJEV_TOKEN"], MAX_BODY), service, a.port, a.idle_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
