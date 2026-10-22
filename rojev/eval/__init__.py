"""Evaluation only: baseline deciders Jev must beat (scripted, simulator-truth oracle, S1's own planner). Never part of
a deployed system; the oracle reads simulator truth."""
from .deciders import AlwaysContinue, NativePlanner, ScriptedSequence, TruthSequence, next_after_latest_done
from .scorecard import (ArmScore, Case, Injection, RunRecords, RunScore, aggregate, render_json, render_markdown,
                        score_run)

__all__ = ["AlwaysContinue", "ArmScore", "Case", "Injection", "NativePlanner", "RunRecords", "RunScore",
           "ScriptedSequence", "TruthSequence", "aggregate", "next_after_latest_done", "render_json", "render_markdown",
           "score_run"]
