"""System 2: the slow decision layer that chooses what System 1 does next.

`Decider` is the contract (`ObservationSnapshot` → `Verdict`); `DecisionWorker` runs one on its own thread and stamps
its `DecisionProposal`; `planning.StepPlanner` chooses each step from a next-step model and the task memory;
`planning.CommitVerifyPlanner` commits a self-planning S1 to one subtask until it is verified. Subpackages: `jev` (Jev v1, the interaction lane's decider), `verify` (end-state checks and the step watcher
that posts step-completion events), `models` (the OneJev and MJev clients both are built from).
"""
from .decider import Decider, Verdict
from .planning import COMPLETE_STEP, CommitVerifyPlanner, NextStep, NextStepModel, StepPlanner
from .worker import DecisionWorker, S2Stats

__all__ = ["COMPLETE_STEP", "CommitVerifyPlanner", "Decider", "DecisionWorker", "NextStep", "NextStepModel", "S2Stats",
           "StepPlanner", "Verdict"]
