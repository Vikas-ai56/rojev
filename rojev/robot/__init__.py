"""Robot layer: one API for the simulator and the real SO-101 (docs/architecture.md rule 19).

`Robot` reads timestamped sensors and writes joint targets through hard limits; `ActionSpec` describes a robot's action
vector for the control layer; `EffectModel` says what an action should do to the measured state (execution
monitor plug-in). The MuJoCo backend lives in `rojev.robot.sim`.
"""
from .action_spec import KEEP, ActionDim, ActionSpec, joint_target_spec
from .base import Clock, MonotonicClock, Rate, Robot
from .effect import EffectModel, JointTargetEffect
from .limits import JointLimits, SafetyFilter
from .mock import ManualClock, MockRobot
from .sim import (TIME_MODES, CameraSpec, MujocoRobot, RealtimeSimClock, SimClock, SimGroundTruth, ViewerClosed,
                  camera_info)
from .types import CameraFrame, CameraInfo, CommandResult, JointCommand, JointState, RobotObservation, Stamp

__all__ = ["KEEP", "TIME_MODES", "ActionDim", "ActionSpec", "CameraFrame", "CameraInfo", "CameraSpec", "Clock",
           "CommandResult", "EffectModel", "JointCommand", "JointLimits", "JointState", "JointTargetEffect", "ManualClock", "MockRobot", "MonotonicClock",
           "MujocoRobot", "Rate", "RealtimeSimClock", "Robot", "RobotObservation", "SafetyFilter", "SimClock",
           "SimGroundTruth", "Stamp", "ViewerClosed", "camera_info", "joint_target_spec"]
