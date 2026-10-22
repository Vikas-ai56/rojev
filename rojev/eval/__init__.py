"""Evaluation only: baseline deciders Jev must beat (scripted, simulator-truth oracle, S1's own planner). Never part of
a deployed system; the oracle reads simulator truth."""
from .deciders import AlwaysContinue, NativePlanner, ScriptedSequence, TruthSequence, next_after_latest_done

__all__ = ["AlwaysContinue", "NativePlanner", "ScriptedSequence", "TruthSequence", "next_after_latest_done"]
