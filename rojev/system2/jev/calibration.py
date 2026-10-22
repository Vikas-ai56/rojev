"""Calibration of the interaction choice: temperature scaling and the abstention margin (spec §9, plan
`choose_or_abstain`).

The chooser's probabilities are sharpened or softened by one temperature T (p_i ∝ p_i^(1/T)), fitted on a labelled set
by minimising the negative log-likelihood of the right act. A decision whose calibrated top-two margin is below
`margin` abstains (the decider waits or asks instead). Until a labelled set is fitted (`fitted` False), T = 1 and
`margin` is a declared tuned constant, not a measured one.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    temperature: float = 1.0
    margin: float = 0.2
    fitted: bool = False
    cases: int = 0

    def apply(self, probabilities: dict[str, float]) -> dict[str, float]:
        labels = list(probabilities)
        logp = np.log(np.clip([probabilities[k] for k in labels], 1e-12, 1.0)) / self.temperature
        p = np.exp(logp - logp.max())
        return dict(zip(labels, (p / p.sum()).tolist()))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "Calibration":
        return Calibration(**json.loads(text))


def fit_temperature(rows: list[dict[str, float]], labels: list[str], margin: float,
                    grid: np.ndarray | None = None) -> Calibration:
    """The temperature minimising the mean negative log-likelihood of `labels` (the right act per row) over `rows`
    (the chooser's raw probabilities per row)."""
    if not rows or len(rows) != len(labels):
        raise ValueError("need one label per row and at least one row")
    grid = np.geomspace(0.05, 20.0, 400) if grid is None else grid

    def nll(t: float) -> float:
        c = Calibration(temperature=float(t))
        return -float(np.mean([np.log(max(c.apply(r)[y], 1e-12)) for r, y in zip(rows, labels)]))

    best = min(grid, key=nll)
    return Calibration(temperature=float(best), margin=margin, fitted=True, cases=len(rows))
