"""`parity.compare` without the GPU model: a stand-in GR00T draws seeded noise like a diffusion head, so the direct
call, the adapter and the service agree only through the shared seed; the report carries per-step differences in
dataset units and latency percentiles for all three paths."""
from types import SimpleNamespace

import numpy as np

from parity import Sample, compare, to_observation
from groot_io import Binding, layout
from rojev.robot.so101 import lerobot_units

UNITS = lerobot_units()
HORIZON = 16
KEYS = ["single_arm", "gripper"]
CONFIG = {"video": SimpleNamespace(modality_keys=["front", "wrist"], delta_indices=[0]),
          "state": SimpleNamespace(modality_keys=KEYS, delta_indices=[0]),
          "language": SimpleNamespace(modality_keys=["annotation.human.task_description"], delta_indices=[0]),
          "action": SimpleNamespace(modality_keys=KEYS, delta_indices=list(range(HORIZON)))}


class NoisyGroot:
    def get_modality_config(self) -> dict:
        return CONFIG

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        return {key: np.repeat(observation["state"][key], HORIZON, axis=1)
                + np.random.normal(0, 1, (1, HORIZON, observation["state"][key].shape[-1])).astype(np.float32)
                for key in KEYS}, {}

    def reset(self) -> dict:
        return {}


def samples(count: int) -> list[Sample]:
    binding = Binding.of(CONFIG, layout())
    out = []
    for i in range(count):
        states = {"single_arm": np.array([[10.0 + i, -20.0, 30.0, -40.0, 5.0]], np.float32),
                  "gripper": np.array([[25.0]], np.float32)}
        images = {"front": [np.full((24, 32, 3), i, np.uint8)], "wrist": [np.full((24, 32, 3), 100 + i, np.uint8)]}
        observation = to_observation(binding, images, states, "put the cube in the bowl", UNITS)
        groot = {"video": {k: np.asarray(v[-1], np.uint8)[None, None] for k, v in images.items()},
                 "state": {k: v[None] for k, v in states.items()},
                 "language": {"annotation.human.task_description": [["put the cube in the bowl"]]}}
        out.append(Sample(i * 16, groot, observation))
    return out


def test_observation_in_radians_round_trips_to_the_dataset_state():
    sample = samples(1)[0]
    assert set(sample.observation.images) == {"front", "wrist_cam"}
    restored = UNITS.to_lerobot(sample.observation.state)
    assert np.allclose(restored, [10.0, -20.0, 30.0, -40.0, 5.0, 25.0], atol=1e-4)


def test_same_seed_gives_parity_through_adapter_and_service():
    report = compare(NoisyGroot(), UNITS, samples(4), np.random.seed)
    assert len(report["steps"]) == 4 and report["worst_max_abs_diff"] < 1e-3
    assert all(s["service_max_abs_diff"] < 1e-3 for s in report["steps"])
    assert report["service_ms"]["n"] == 4 and report["direct_ms"]["p95"] >= report["direct_ms"]["p50"]


def test_a_different_seed_shows_up_as_a_difference():
    calls = iter(range(100))
    report = compare(NoisyGroot(), UNITS, samples(2), lambda i: np.random.seed(next(calls)))
    assert report["worst_max_abs_diff"] > 0.1


class TextAwareGroot(NoisyGroot):
    def get_action(self, observation: dict) -> tuple[dict, dict]:
        actions, info = super().get_action(observation)
        shift = 5.0 if "green" in observation["language"]["annotation.human.task_description"][0][0] else 0.0
        return {key: value + shift for key, value in actions.items()}, info


def test_language_probe_reports_how_far_a_swapped_instruction_moves_the_actions():
    listening = compare(TextAwareGroot(), UNITS, samples(3), np.random.seed, swap_text="cube into green bowl")
    assert abs(listening["language_probe"]["mean_abs_diff"] - 5.0) < 1e-3 and listening["worst_max_abs_diff"] < 1e-3
    deaf = compare(NoisyGroot(), UNITS, samples(3), np.random.seed, swap_text="cube into green bowl")
    assert deaf["language_probe"]["max_abs_diff"] < 1e-6
    assert set(deaf["chunks"]) == {"direct", "adapter", "swapped"}
    assert deaf["chunks"]["swapped"].shape == (3, HORIZON, 6)
