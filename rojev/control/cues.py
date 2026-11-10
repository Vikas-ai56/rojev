"""Physical cues for the done check: moments when a step has likely just finished, read from the actions the control
loop applied, with no model in the path.

`SwitchCues` watches the action spec's switch dimensions (those with a `threshold`: a gripper's open / close, a base's
mode): when an applied action puts a switch on the other side of its threshold than the previous applied action did,
it calls `notify` with the switch's name and new side. A grasp, a release and a mode change end most manipulation
steps, so the step watcher checks the running step at that moment instead of waiting for its next periodic check.
Use it as (part of) the control loop's `on_tick`; it runs on the loop's thread and does no work beyond a comparison.
"""
from collections.abc import Callable

from rojev.robot import ActionSpec

from .loop import TickRecord


class SwitchCues:
    def __init__(self, spec: ActionSpec, notify: Callable[[str], None]):
        self.switches = [(i, d.name, d.threshold) for i, d in enumerate(spec.dims) if d.threshold is not None]
        self.notify = notify
        self._last: dict[int, bool] = {}

    def __call__(self, record: TickRecord) -> None:
        action = record.tick.action
        for i, name, threshold in self.switches:
            on = bool(action[i] >= threshold)
            if i in self._last and self._last[i] != on:
                self.notify(f"{name} {'on' if on else 'off'}")
            self._last[i] = on
