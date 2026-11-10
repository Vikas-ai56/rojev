"""`probe.run` without the GPU model: a stand-in GR00T whose actions depend on the instruction, the seed and the
denoising setting, so the report must show the error against the recorded chunk, the spread across seeds, how far
each instruction variant moves the actions, latency per denoising setting, and a failing variant recorded, not fatal."""
import numpy as np

from probe import ORIGINAL, ProbeSample, run

HORIZON = 8
KEYS = ["arm", "gripper"]


class StandIn:
    def __init__(self):
        self.denoising = 4

    def get_action(self, observation: dict) -> tuple[dict, dict]:
        text = observation["language"]["task"][0][0]
        if text == "":
            raise ValueError("empty instruction")
        shift = 0.0 if "block" in text else 2.0
        noise = np.random.normal(0, 0.1 * self.denoising, (1, HORIZON, 4))
        return {"arm": noise[:, :, :3] + shift, "gripper": noise[:, :, 3:]}, {}


def samples() -> list[ProbeSample]:
    out = []
    for episode in (1, 2):
        for step in (0, 40):
            groot = {"video": {}, "state": {}, "language": {"task": [["Put the blue block in the green bowl"]]}}
            text = "Put the blue block in the green bowl"
            out.append(ProbeSample(episode, step, groot, np.zeros((HORIZON, 4)), text))
    return out


def test_probe_reports_error_spread_movement_latency_and_survives_a_failing_variant():
    model = StandIn()
    variants = {e: {"paraphrase": "Place the blue block into the green bowl", "unrelated": "Fold the towel",
                    "empty": ""} for e in (1, 2)}
    report = run(model, samples(), KEYS, variants, [0, 1, 2], [4, 8],
                 lambda d: setattr(model, "denoising", d), np.random.seed)
    summary = report["summary"]
    assert set(summary) == {ORIGINAL, "paraphrase", "unrelated", "empty"}
    assert summary["paraphrase"]["moved_from_original"] < 1e-9
    assert abs(summary["unrelated"]["moved_from_original"] - 1.5) < 1e-9
    assert summary["empty"]["failed"] == 2 * 2 * 2 and summary["empty"]["error_vs_recorded"] is None
    assert summary[ORIGINAL]["spread_across_seeds"] > 0
    assert set(report["latency_ms"]) == {"4", "8"} and report["latency_ms"]["4"]["n"] == 4 * 3 * 3
    assert report["chunks"].shape == (2 * 4 * 3, 3, HORIZON, 4) and len(report["labels"]) == 24
    spreads = {d: np.mean([r["spread_across_seeds"] for r in report["records"]
                           if r.get("denoising_steps") == d and "error" not in r]) for d in (4, 8)}
    assert spreads[8] > spreads[4]
