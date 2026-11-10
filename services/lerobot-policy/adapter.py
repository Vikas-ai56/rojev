"""Any LeRobot-format policy checkpoint as a System 1 `Policy`, bound by a `ModelConfig` (model_config.py).

LeRobot does the model work through its own generic API (lerobot 0.6.1): the checkpoint's `PreTrainedConfig`, the
policy class its `type` names (`get_policy_class`), and the checkpoint's own pre / post processor pipelines
(`make_pre_post_processors`). Nothing here is specific to one policy; GR00T N1.7 is only the first config.

Load (`load`): the pinned checkpoint and every pinned override value resolve to snapshot directories in the local
Hugging Face cache (the GPU hook downloads them; nothing is fetched here), so the policy, its processors and e.g.
GR00T's base model come from exactly the pinned revisions. Overrides replace checkpoint config fields through
`dataclasses.replace`, so the config's own validation runs on them; an unknown field fails. A device that is not
available fails (LeRobot would silently fall back to CPU). `check_features` then requires the checkpoint's inputs to be
exactly what the config feeds: every camera and the state key exist with the right feature type, every image input
of the checkpoint is fed by a camera (a missing view would silently change what the model sees), no other input kind
is expected, and the action width equals `action_dim`.

Per /act request (`LeRobotPolicy.act`), as LeRobot's own rollout does: a batch of one in LeRobot's convention (each
camera's H×W×3 uint8 frame → float32 (1, 3, H, W) in [0, 1] under its checkpoint key, `Observation.state` → float32
(1, D) with D the checkpoint's state width, text = the subtask if given else the task → "task": [text]) → preprocessor
→ `predict_action_chunk` → postprocessor, all under inference mode (plus autocast when the checkpoint uses AMP on
CUDA) → float32 (T, action_dim) in the checkpoint's environment units, carrying the config's `execute_steps`; a
non-finite action, or `execute_steps` beyond T, fails the request (the warm-up meets it before any robot does).
Timings: pre (batch + preprocessor), model (synchronised with the device), post.

Speed (`ModelConfig.compile_modules`, `warmup`): `compile_submodules` replaces each named submodule of the loaded
policy by its `torch.compile`d version (LeRobot's GR00T has no compile switch of its own; NVIDIA's benchmark compiles
its action head); `LeRobotPolicy.warm_up(n)` runs n requests on a synthetic observation of the checkpoint's input
shapes, request i with an instruction i + 1 times "warm up" long, and resets. The token count follows the instruction,
and a compiled module recompiles when an input length changes (one fixed text left a ~60 s recompile inside the first
real request, E33); two lengths during warm-up let torch.compile settle on a length-general version before any robot
asks (to be confirmed on the GPU). `reset` resets the policy and both
pipelines. torch, lerobot and the Hub client are imported only inside the loaders (`load`, `load_checkpoint`,
`torch_bridge`, `local_snapshot`), so everything else runs on the Mac with a stand-in policy behind `TensorBridge`.
"""
import dataclasses
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_config import HubSnapshot, ModelConfig
from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy

VISUAL, STATE, ACTION = "VISUAL", "STATE", "ACTION"
ACTION_KEY = "action"
TASK_KEY = "task"


@dataclass(frozen=True)
class Feature:
    kind: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class TensorBridge:
    from_numpy: Callable[[np.ndarray], object]
    to_numpy: Callable[[object], np.ndarray]
    inference: Callable[[], AbstractContextManager]
    synchronize: Callable[[], None]


def check_features(config: ModelConfig, inputs: dict[str, Feature], outputs: dict[str, Feature]) -> None:
    problems = []
    for camera, key in config.cameras.items():
        if inputs.get(key, Feature("", ())).kind != VISUAL:
            problems.append(f"camera {camera!r} feeds {key!r}, which is not an image input of the checkpoint")
    if inputs.get(config.state, Feature("", ())).kind != STATE:
        problems.append(f"state key {config.state!r} is not a state input of the checkpoint")
    fed = set(config.cameras.values()) | {config.state}
    for key, feature in inputs.items():
        if key not in fed:
            problems.append(f"the checkpoint expects {feature.kind} input {key!r} {feature.shape}, which no camera "
                            f"or state of this config feeds")
    action = outputs.get(ACTION_KEY)
    if action is None or action.kind != ACTION or action.shape != (config.action_dim,):
        problems.append(f"checkpoint action output is {action}, config expects ({config.action_dim},)")
    if problems:
        raise ValueError(f"config {config.name!r} does not fit checkpoint {config.checkpoint}:\n  "
                         + "\n  ".join(problems)
                         + f"\n  checkpoint inputs: {inputs}")


def observation_batch(obs: Observation, text: str, config: ModelConfig, inputs: dict[str, Feature]) -> dict:
    missing = sorted(set(config.cameras) - set(obs.images))
    if missing:
        raise ValueError(f"observation lacks cameras {missing} (has {sorted(obs.images)})")
    batch: dict = {}
    for camera, key in config.cameras.items():
        frame = obs.images[camera]
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"camera {camera!r}: expected H×W×3 uint8, got {frame.shape} {frame.dtype}")
        batch[key] = (np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255.0)[None]
    state = np.asarray(obs.state, dtype=np.float32).reshape(-1)
    if state.shape != inputs[config.state].shape:
        raise ValueError(f"state has shape {state.shape}, checkpoint {config.state!r} expects "
                         f"{inputs[config.state].shape}")
    batch[config.state] = state[None]
    batch[TASK_KEY] = [text]
    return batch


def chunk_actions(actions: np.ndarray, action_dim: int) -> np.ndarray:
    if actions.ndim != 3 or actions.shape[0] != 1 or actions.shape[2] != action_dim:
        raise ValueError(f"postprocessed chunk has shape {actions.shape}, expected (1, T, {action_dim})")
    if not np.isfinite(actions).all():
        raise ValueError("policy returned non-finite actions")
    return actions[0].astype(np.float32)


class LeRobotPolicy(Policy):
    def __init__(self, config: ModelConfig, policy, preprocessor, postprocessor, inputs: dict[str, Feature],
                 bridge: TensorBridge, policy_type: str):
        self.config, self.policy, self.inputs, self.bridge = config, policy, inputs, bridge
        self.preprocessor, self.postprocessor, self.policy_type = preprocessor, postprocessor, policy_type

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        bridge = self.bridge
        with bridge.inference():
            t0 = time.perf_counter()
            batch = observation_batch(obs, subtask or obs.task, self.config, self.inputs)
            model_input = self.preprocessor({key: bridge.from_numpy(value) if isinstance(value, np.ndarray) else value
                                             for key, value in batch.items()})
            t1 = time.perf_counter()
            chunk = self.policy.predict_action_chunk(model_input)
            bridge.synchronize()
            t2 = time.perf_counter()
            actions = bridge.to_numpy(self.postprocessor(chunk))
            t3 = time.perf_counter()
        return ActionChunk(chunk_actions(actions, self.config.action_dim),
                           {"pre": round((t1 - t0) * 1000, 1), "model": round((t2 - t1) * 1000, 1),
                            "post": round((t3 - t2) * 1000, 1)}, execute_steps=self.config.execute_steps)

    def reset(self) -> None:
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def warm_up(self, requests: int) -> list[float]:
        images = {camera: np.zeros(_hwc(self.inputs[key].shape), np.uint8) for camera, key in self.config.cameras.items()}
        state = np.zeros(self.inputs[self.config.state].shape, np.float32)
        wall_ms = []
        for request in range(requests):
            w0 = time.perf_counter()
            self.act(Observation(images, state, " ".join(["warm up"] * (request + 1))), "")
            wall_ms.append(round((time.perf_counter() - w0) * 1000, 1))
        self.reset()
        return wall_ms


def _hwc(shape: tuple[int, ...]) -> tuple[int, int, int]:
    return (shape[1], shape[2], shape[0]) if shape[0] == 3 else (shape[0], shape[1], shape[2])


def compile_submodules(policy, paths: tuple[str, ...], mode: str, compile_fn: Callable) -> None:
    for path in paths:
        *parents, name = path.split(".")
        owner = policy
        for part in parents:
            owner = getattr(owner, part)
        setattr(owner, name, compile_fn(getattr(owner, name), mode=mode))


@dataclass(frozen=True)
class Checkpoint:
    directory: Path
    policy_config: object
    preprocessor: object
    postprocessor: object
    inputs: dict[str, Feature]
    outputs: dict[str, Feature]


def local_snapshot(snapshot: HubSnapshot) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(snapshot.repo, revision=snapshot.revision, local_files_only=True))
    except LocalEntryNotFoundError as e:
        raise FileNotFoundError(f"{snapshot} is not in the local Hugging Face cache: run "
                                f"`python scripts/hf_pins.py snapshot {snapshot}` first") from e


def load_checkpoint(config: ModelConfig, resolve: Callable[[HubSnapshot], Path] = local_snapshot) -> Checkpoint:
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors

    directory = resolve(config.checkpoint)
    stored = PreTrainedConfig.from_pretrained(directory)
    fields = {f.name for f in dataclasses.fields(stored) if f.init}
    unknown = sorted(set(config.overrides) - fields)
    if unknown:
        raise ValueError(f"overrides {unknown} are not fields of the checkpoint's {type(stored).__name__}")
    overrides = {key: str(resolve(value)) if isinstance(value, HubSnapshot) else value
                 for key, value in config.overrides.items()}
    policy_config = dataclasses.replace(stored, **{**overrides, "device": config.device, "pretrained_path": directory})
    if policy_config.device != config.device:
        raise RuntimeError(f"device {config.device!r} is not available here (LeRobot would run on "
                           f"{policy_config.device!r})")
    inputs = {key: Feature(f.type.value, tuple(f.shape)) for key, f in policy_config.input_features.items()}
    outputs = {key: Feature(f.type.value, tuple(f.shape)) for key, f in policy_config.output_features.items()}
    check_features(config, inputs, outputs)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_config, pretrained_path=str(directory),
        preprocessor_overrides={"device_processor": {"device": config.device}})
    return Checkpoint(directory, policy_config, preprocessor, postprocessor, inputs, outputs)


def torch_bridge(device_name: str, use_amp: bool) -> TensorBridge:
    from contextlib import ExitStack, contextmanager

    import torch

    device = torch.device(device_name)

    @contextmanager
    def inference():
        with ExitStack() as stack:
            stack.enter_context(torch.inference_mode())
            if use_amp and device.type == "cuda":
                stack.enter_context(torch.autocast(device_type=device.type))
            yield

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return TensorBridge(from_numpy=torch.from_numpy,
                        to_numpy=lambda tensor: tensor.detach().to("cpu", torch.float32).numpy(),
                        inference=inference, synchronize=synchronize)


def load(config: ModelConfig) -> LeRobotPolicy:
    from lerobot.policies import get_policy_class

    checkpoint = load_checkpoint(config)
    policy_config = checkpoint.policy_config
    policy = get_policy_class(policy_config.type).from_pretrained(checkpoint.directory, config=policy_config)
    if config.compile_modules:
        import torch

        compile_submodules(policy, config.compile_modules, config.compile_mode, torch.compile)
    return LeRobotPolicy(config, policy, checkpoint.preprocessor, checkpoint.postprocessor, checkpoint.inputs,
                         torch_bridge(config.device, policy_config.use_amp), policy_config.type)
