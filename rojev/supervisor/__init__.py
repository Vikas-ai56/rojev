"""Supervisor: decides the next subtask for a frozen executor (docs/paper/method.md)."""
from .base import Decision, DecisionContext, SubtaskSource
from .candidates import CandidateBuilder, normalize
from .onejev_arm import OneJevSupervisor

__all__ = ["CandidateBuilder", "Decision", "DecisionContext", "OneJevSupervisor", "SubtaskSource", "normalize"]
