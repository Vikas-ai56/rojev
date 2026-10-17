"""Timestamped data exchanged with any robot backend (simulator or hardware).

Canonical units: the robot model's (MJCF) joint coordinates, in radians, gripper hinge included. A backend converts
to and from its native units (e.g. LeRobot degrees / 0..100) through its calibration; nothing above it sees those.
Times are on the robot's `Clock`: simulated time in simulation, monotonic time on hardware, never mixed.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stamp:
    acquired: float  # when the measurement was taken (camera exposure / encoder read)
    received: float  # when it reached the SDK; `received - acquired` is the latency
    seq: int  # per-stream counter; gaps mean dropped samples


@dataclass(frozen=True)
class JointState:
    stamp: Stamp
    names: tuple[str, ...]
    position: np.ndarray  # rad
    velocity: np.ndarray | None  # rad/s, None if the backend cannot measure it


@dataclass(frozen=True)
class CameraInfo:
    """Calibration of one camera, in the ROS CameraInfo convention."""

    width: int
    height: int
    K: np.ndarray  # 3x3 intrinsics, pixels
    D: np.ndarray  # distortion coefficients for `distortion_model`
    distortion_model: str  # "none" or "plumb_bob"
    parent_frame: str  # body the camera is rigidly mounted on ("world" for a fixed camera)
    T_parent_optical: np.ndarray  # 4x4; optical frame: z forward, x right, y down


@dataclass(frozen=True)
class CameraFrame:
    stamp: Stamp
    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) float32 metres along the optical z axis, if the camera provides depth
    info: CameraInfo


@dataclass(frozen=True)
class Observation:
    joints: JointState
    cameras: dict[str, CameraFrame]  # latest frame per camera that has delivered one


@dataclass(frozen=True)
class JointCommand:
    names: tuple[str, ...]
    position: np.ndarray  # target joint positions, rad


@dataclass(frozen=True)
class CommandResult:
    requested: JointCommand
    applied: JointCommand  # what the backend actually received after the safety filter
    violations: tuple[str, ...]  # every limit the request broke; empty means applied == requested
