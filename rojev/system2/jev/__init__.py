"""Jev v1 (spec §9): the interaction lane's decider. A change is answered with one calibrated act chosen by OneJev
over the legal acts generated in code, inside question and wait budgets; a new task is then one of S1's trained task
sentences, chosen by a second typed question.

  JevInteraction    the decider (`Decider` contract)
  OneJevChooser     scores the options of one question with one OneJev Choice; `Chooser` is the contract for others
  CanonicalTask     default `Reviser`: REVISE = the trained task sentence the user asked for, or none (then ask)
  Calibration       temperature + abstention margin; `fit_calibration` fits both on labelled cases (offline)
  TEMPLATES         default wording of what the robot says (template id → text)
"""
from .acts import (ABORT, ASK, CLARIFY, CONTINUE, HOLD, MOVE, RESUME, REVISE, SWITCH, TEMPLATES, WAIT, WAITING, Option,
                   Reviser, Revision, candidates, effect)
from .calibration import Calibration, fit_calibration, fit_margin, fit_temperature
from .chooser import ACT_QUESTION, Chooser, OneJevChooser
from .interaction import JevInteraction
from .task_choice import NONE, TASK_QUESTION, CanonicalTask

__all__ = ["ABORT", "ACT_QUESTION", "ASK", "CLARIFY", "CONTINUE", "CanonicalTask", "Calibration", "Chooser", "HOLD",
           "JevInteraction", "MOVE", "NONE", "OneJevChooser", "Option", "RESUME", "REVISE", "Reviser", "Revision", "SWITCH",
           "TASK_QUESTION", "TEMPLATES", "WAIT", "WAITING", "candidates", "effect", "fit_calibration", "fit_margin",
           "fit_temperature"]
