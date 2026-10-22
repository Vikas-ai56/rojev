"""QUARANTINED v2 code (chunk-synchronous supervisor): used only by `executors/robocasa-pi052/workflow.py`, the E9/E13
playground scripts and their tests. Kept as the arm-D comparison for roadmap test T8, then deleted. Do not build on it:
new deciders implement `rojev.system2.Decider`."""
from .arms import Decision, DecisionContext, SubtaskSource
from .baselines import FixedSequence, OracleSequence
from .candidates import CandidateBuilder, normalize
from .onejev_arm import OneJevSupervisor

__all__ = ["CandidateBuilder", "Decision", "DecisionContext", "FixedSequence", "OneJevSupervisor", "OracleSequence",
           "SubtaskSource", "normalize"]
