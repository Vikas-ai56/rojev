"""Calibration of the interaction choice: temperature scaling and the abstention margin (spec §9, plan
`choose_or_abstain`), both fitted offline on labelled cases; at runtime only `Calibration.apply` runs.

The chooser's probabilities are sharpened or softened by one temperature T (p_i ∝ p_i^(1/T)), fitted by minimising the
negative log-likelihood of the right act (`fit_temperature`). A decision whose calibrated top-two margin is below
`margin` abstains (the decider waits or asks instead); `fit_margin` picks the smallest margin at which the decisions
still made reach `target_accuracy`. `fit_calibration` does both. Until a labelled set is fitted (`fitted` False),
T = 1 and `margin` is a declared tuned constant, not a measured one.
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
    target_accuracy: float | None = None
    coverage: float | None = None

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


def _check(rows: list[dict[str, float]], labels: list[str]) -> None:
    if not rows or len(rows) != len(labels):
        raise ValueError("need one label per row and at least one row")


def nll(rows: list[dict[str, float]], labels: list[str], temperature: float) -> float:
    """Mean negative log-likelihood of the right act after temperature scaling."""
    c = Calibration(temperature=float(temperature))
    return -float(np.mean([np.log(max(c.apply(r)[y], 1e-12)) for r, y in zip(rows, labels)]))


def fit_temperature(rows: list[dict[str, float]], labels: list[str], grid: np.ndarray | None = None) -> float:
    """The temperature minimising `nll` over `rows` (the chooser's raw probabilities per case) and `labels`."""
    _check(rows, labels)
    grid = np.geomspace(0.05, 20.0, 400) if grid is None else grid
    return float(min(grid, key=lambda t: nll(rows, labels, t)))


def margins(rows: list[dict[str, float]], temperature: float) -> tuple[np.ndarray, list[str]]:
    """Per row: the calibrated top-two margin and the top act."""
    c = Calibration(temperature=temperature)
    out, tops = [], []
    for r in rows:
        p = sorted(c.apply(r).items(), key=lambda kv: kv[1], reverse=True)
        out.append(p[0][1] - (p[1][1] if len(p) > 1 else 0.0))
        tops.append(p[0][0])
    return np.array(out), tops


def fit_margin(rows: list[dict[str, float]], labels: list[str], temperature: float,
               target_accuracy: float) -> tuple[float, float]:
    """The smallest margin at which the decisions still made (margin ≥ it) are right at least `target_accuracy` of the
    time, and the share of cases still decided (coverage). If no margin reaches the target, 1.0 (always abstain) and
    coverage 0: the chooser is not good enough to act alone."""
    _check(rows, labels)
    m, tops = margins(rows, temperature)
    right = np.array([t == y for t, y in zip(tops, labels)])
    for threshold in np.unique(np.concatenate([[0.0], m])):
        kept = m >= threshold
        if kept.any() and right[kept].mean() >= target_accuracy:
            return float(threshold), float(kept.mean())
    return 1.0, 0.0


def fit_calibration(rows: list[dict[str, float]], labels: list[str], target_accuracy: float) -> Calibration:
    temperature = fit_temperature(rows, labels)
    margin, coverage = fit_margin(rows, labels, temperature, target_accuracy)
    return Calibration(temperature, margin, True, len(rows), target_accuracy, coverage)
