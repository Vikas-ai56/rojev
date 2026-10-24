"""Live-scenario harness for RoboCasa (executor side only; the `rojev` core never depends on it).

  skills      SkillScriptPolicy: scripted S1 (spoof of π0.5² without a GPU)
  deciders    RuleBasedInteraction, RevisableSequence (spoofs of Jev and the planner); KeywordChooser, RewriteReviser
              (spoofs of OneJev's scores and π0.5²'s re-planning inside the real JevInteraction); TruthVerifier
              (spoof of MJev's end-state checks); SlowDecider, FailingDecider
  recording   RecordingPolicy, RecordingCommit, RecordingDecider: per-call logs through the core's interfaces;
              InteractionCases: Jev decisions saved as calibration cases; RecordingVerifier: checks logged with truth
  frame_check FrameCheck: samples the camera frames during a run; a run with noise / blank frames is not scored
  tracking    FrameRecorder: cameras + simulator state during a live run (scored afterwards by sam_check.py track);
              truth_masks and the 2D geometry shared with done_check_ab.py / sam_check.py
  viewer      MirrorPublisher: live MuJoCo viewer in its own process
  video       VideoRecorder: cameras + decision panel, encoded off the control loop
  live_view   LiveView: what a watcher sees (cameras + status) written during the run, for a live user
  users       Line, FuzzUser: simulated users for campaigns (intents, phrasings, typos, timing)
  judge       judge_run: per user line, the answer vs what the state allowed (pass / safe miss / unsafe); per run
"""
from .deciders import (FailingDecider, KeywordChooser, RevisableSequence, RewriteReviser, RuleBasedInteraction,
                       SlowDecider, TruthVerifier)
from .frame_check import FrameCheck
from .live_view import LiveView
from .recording import InteractionCases, RecordingCommit, RecordingDecider, RecordingPolicy, RecordingVerifier
from .skills import SkillScriptPolicy
from .tracking import FrameRecorder
from .video import VideoRecorder
from .viewer import MirrorPublisher

__all__ = ["FailingDecider", "FrameCheck", "FrameRecorder", "InteractionCases", "KeywordChooser", "LiveView",
           "MirrorPublisher", "RecordingCommit", "RecordingDecider", "RecordingPolicy", "RecordingVerifier", "RevisableSequence",
           "RewriteReviser", "RuleBasedInteraction", "SkillScriptPolicy", "SlowDecider", "TruthVerifier",
           "VideoRecorder"]
