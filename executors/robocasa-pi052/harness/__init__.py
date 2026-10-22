"""Live-scenario harness for RoboCasa (executor side only; the `rojev` core never depends on it).

  skills      SkillScriptPolicy: scripted S1 (spoof of π0.5² without a GPU)
  deciders    RuleBasedInteraction, RevisableSequence (spoofs of Jev and the planner); KeywordChooser, RewriteReviser
              (spoofs of OneJev's scores and π0.5²'s re-planning inside the real JevInteraction); TruthVerifier
              (spoof of MJev's end-state checks); SlowDecider, FailingDecider
  recording   RecordingPolicy, RecordingCommit, RecordingDecider: per-call logs through the core's interfaces
  viewer      MirrorPublisher: live MuJoCo viewer in its own process
  video       VideoRecorder: cameras + decision panel, encoded off the control loop
"""
from .deciders import (FailingDecider, KeywordChooser, RevisableSequence, RewriteReviser, RuleBasedInteraction,
                       SlowDecider, TruthVerifier)
from .recording import RecordingCommit, RecordingDecider, RecordingPolicy
from .skills import SkillScriptPolicy
from .video import VideoRecorder
from .viewer import MirrorPublisher

__all__ = ["FailingDecider", "KeywordChooser", "MirrorPublisher", "RecordingCommit", "RecordingDecider",
           "RecordingPolicy", "RevisableSequence", "RewriteReviser", "RuleBasedInteraction", "SkillScriptPolicy",
           "SlowDecider", "TruthVerifier", "VideoRecorder"]
