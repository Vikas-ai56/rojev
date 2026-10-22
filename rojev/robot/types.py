"""Timestamped data exchanged with any robot backend (simulator or hardware).

Canonical units: the robot model's (MJCF) joint coordinates, in radians, gripper hinge included. A backend converts
to and from its native units (e.g. LeRobot degrees / 0..100) through its calibration; nothing above it sees those.
Times are on the robot's `Clock`: simulated time in simulation, monotonic time on hardware, never mixed.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stamp:
    """`acquired`: when the measurement was taken (camera exposure / encoder read); `received`: when it reached the SDK
    (`received - acquired` is the latency); `seq`: per-stream counter, gaps mean dropped samples."""

    acquired: float
    received: float
    seq: int


@dataclass(frozen=True)
class JointState:
    """`position` in rad; `velocity` in rad/s, None if the backend cannot measure it."""

    stamp: Stamp
    names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray | None


@dataclass(frozen=True)
class CameraInfo:
    """Calibration of one camera, in the ROS CameraInfo convention: `K` 3×3 intrinsics in pixels; `D` distortion
    coefficients for `distortion_model` ("none" or "plumb_bob"); `parent_frame` the body the camera is rigidly mounted
    on ("world" for a fixed camera); `T_parent_optical` 4×4, optical frame z forward, x right, y down."""

    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    distortion_model: str
    parent_frame: str
    T_parent_optical: np.ndarray


@dataclass(frozen=True)
class CameraFrame:
    """`rgb` (H, W, 3) uint8; `depth` (H, W) float32 metres along the optical z axis, if the camera provides depth."""

    stamp: Stamp
    name: str
    rgb: np.ndarray
    depth: np.ndarray | None
    info: CameraInfo


@dataclass(frozen=True)
class RobotObservation:
    """Joint state plus the latest frame of every camera that has delivered one."""

    joints: JointState
    cameras: dict[str, CameraFrame]


@dataclass(frozen=True)
class JointCommand:
    """Target joint positions in rad."""

    names: tuple[str, ...]
    position: np.ndarray


@dataclass(frozen=True)
class CommandResult:
    """`applied` is what the backend actually received after the safety filter; `violations` lists every limit the
    request broke (empty means applied == requested)."""

    requested: JointCommand
    applied: JointCommand
    violations: tuple[str, ...]
