"""The shared assembly: any robot plug-in and any model plug-ins become one supervised run (the top layer; nothing in
`rojev` depends on it). `Embodiment` is what the core needs to know about a robot; `SessionConfig` (and
`load_config`) the run's settings; `plugins.build` makes a plug-in from a config section; `Session` builds and runs the
runtime and writes the run's records."""
from .config import SessionConfig, load_config
from .embodiment import Embodiment
from .plugins import build, resolve
from .session import RunResult, Session
from .video import VideoRecorder

__all__ = ["Embodiment", "RunResult", "Session", "SessionConfig", "VideoRecorder", "build", "load_config", "resolve"]
