"""A run's settings, from a config file or a mapping: runtime timings, planning triggers, done-check cascade, time mode.

`SessionConfig` holds every knob the shared assembly (`Session`) uses, with the defaults the runtime was measured with;
`SessionConfig.from_mapping` refuses an unknown key, so a misspelt setting never silently falls back to a default.
`load_config(path)` reads a YAML file whose `runtime` section is a `SessionConfig` and whose other sections (robot,
models) belong to the plug-ins that read them.

Time modes: `lockstep` stops time while any model thinks (one thread alternates S1 supply, the owner and one control
tick; how benchmarks are scored); otherwise every part runs on its own thread against the robot's clock.
How much of each S1 chunk runs before S1 is asked again is not a run setting: it is the model's own
(`ActionChunk.execute_steps`, from its service), and the S1 runner derives the request timing from it.
"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import yaml

INSTRUCTION_CHANGES = ("review", "revoke")


@dataclass(frozen=True)
class SessionConfig:
    lease_s: float = 3.0
    watchdog_s: float = 10.0
    step_budget_s: float | None = None
    liveness_s: float = 30.0
    review_s: float = 1.0
    on_change: str = "review"
    interaction_timeout_s: float = 1.0
    hold_recheck_s: float | None = None
    blend_s: float = 0.2
    stop_s: float = 0.0
    lockstep: bool = True
    verify_every_s: float = 1.0
    recheck_s: float = 0.5
    done_threshold: float = 0.7
    confirmations: int = 2
    dialogue_turns: int = 8

    def __post_init__(self):
        if self.on_change not in INSTRUCTION_CHANGES:
            raise ValueError(f"on_change {self.on_change!r} not in {INSTRUCTION_CHANGES}")

    @classmethod
    def from_mapping(cls, values: dict | None) -> "SessionConfig":
        values = dict(values or {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown runtime settings {unknown}; known: {sorted(known)}")
        return cls(**values)


def load_config(path: Path) -> tuple[SessionConfig, dict]:
    """The `runtime` section as a `SessionConfig`, and the whole file for the plug-ins."""
    document = yaml.safe_load(Path(path).read_text()) or {}
    return SessionConfig.from_mapping(document.get("runtime")), document
