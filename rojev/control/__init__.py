"""Control layer: the real-time loop that owns what the robot physically does, independent of every model.

`ControlLoop` steps a backend on the robot's clock; `ExecutionRuntime` turns commands and timed chunks into one action
per tick under the authority rules; `Guard` keeps the robot safe when anything above fails; `RobotBackend` adapts any
`rojev.robot.Robot` to the loop.
"""
from .execution import ExecutionRuntime, Mode, TickResult
from .guard import Guard
from .loop import Backend, ControlLoop, TickRecord
from .robot_backend import RobotBackend

__all__ = ["Backend", "ControlLoop", "ExecutionRuntime", "Guard", "Mode", "RobotBackend", "TickRecord", "TickResult"]
