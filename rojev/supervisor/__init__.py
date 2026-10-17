"""Supervisor: decides the next subtask for a frozen executor (docs/paper/method.md)."""
from .candidates import CandidateBuilder, normalize

__all__ = ["CandidateBuilder", "normalize"]
