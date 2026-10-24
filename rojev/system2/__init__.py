"""System 2: the slow decision layer that chooses what System 1 does next.

`Decider` is the contract (`ObservationSnapshot` → `Verdict`); `DecisionWorker` runs one on its own thread and stamps
its `DecisionProposal`; `planning.CommitVerifyPlanner` commits a self-planning S1 to one subtask until it is
verified. Subpackages: `jev` (Jev v1, the interaction lane's decider), `verify` (end-state checks and the step watcher
that posts step-completion events), `models` (the OneJev and MJev clients both are built from).
"""
from .decider import Decider, Verdict
from .planning import CommitVerifyPlanner
from .worker import DecisionWorker, S2Stats

__all__ = ["CommitVerifyPlanner", "Decider", "DecisionWorker", "S2Stats", "Verdict"]
