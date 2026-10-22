"""System 1: the fast visuomotor policy and what keeps it supplied.

`Policy` is the contract (observation + subtask text → `ActionChunk`, or K sampled chunks); `RemotePolicy` reaches one
on a GPU over the versioned `wire` format (WebP-90 frames by default, K samples in one request); `S1Runner` keeps
the command that needs actions supplied with chunks, on its own thread; `commit` decides how much of each chunk is
committed (`speculative_predict` = K-sample consensus, opt-in); `monitor` checks per tick that the world responds to
the applied actions as on nominal runs (opt-in) → `Deviation`.
"""
from .commit import (COMMIT_POLICIES, Commitment, CommitPolicy, ConsensusCommit, FixedCommit, PhaseCommit,
                     make_commit_policy, motion, speed_profile)
from .monitor import ExecutionMonitor, MonitorModel, Recording, fit_monitor
from .policy import ActionChunk, Policy
from .remote import RemotePolicy
from .runner import S1Runner, S1Stats

__all__ = ["COMMIT_POLICIES", "ActionChunk", "CommitPolicy", "Commitment", "ConsensusCommit", "ExecutionMonitor", "FixedCommit",
           "MonitorModel", "PhaseCommit", "Policy", "Recording", "RemotePolicy", "S1Runner", "S1Stats", "fit_monitor", "make_commit_policy", "motion", "speed_profile"]
