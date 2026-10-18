"""Evaluation-only components (baselines, ground-truth references); never part of the deployed supervisor."""
from .baselines import FixedSequence, OracleSequence

__all__ = ["FixedSequence", "OracleSequence"]
