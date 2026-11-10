"""Step ends of the open-drawer / pick-red / stack-on-blue SO-101 recording, read from the gripper.

The recording's three steps leave one gripper signature each (LeRobot units, 0-100): the gripper closes fully on the
drawer's lid while lifting it, opens again once the lid is up, closes to a partly-open hold around the red cube, and
opens again once the cube sits on the blue cube. So the step ends are the edges of two plateaus:
  lid plateau   the longest run below CLOSED_MAX       → lid_closed (start), drawer_open (first frame after)
  hold plateau  the longest later run in HOLD_BAND      → grasped (start), released (first frame after)
Runs shorter than MIN_RUN frames are motion through the band, not a plateau. An episode without both plateaus in that
order raises; every labelled episode is also checked by eye on its contact sheet before it is used.
"""
from dataclasses import dataclass

import numpy as np

CLOSED_MAX = 8.0
HOLD_BAND = (10.0, 22.0)
MIN_RUN = 30


@dataclass(frozen=True)
class StepEnds:
    lid_closed: int
    drawer_open: int
    grasped: int
    released: int
    length: int


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _longest(runs: list[tuple[int, int]]) -> tuple[int, int] | None:
    long_enough = [r for r in runs if r[1] - r[0] >= MIN_RUN]
    return max(long_enough, key=lambda r: r[1] - r[0]) if long_enough else None


def step_ends(gripper: np.ndarray) -> StepEnds:
    lid = _longest(_runs(gripper < CLOSED_MAX))
    if lid is None:
        raise ValueError("no fully closed plateau (lid lift)")
    after = np.zeros_like(gripper, dtype=bool)
    after[lid[1]:] = True
    hold = _longest(_runs(after & (gripper > HOLD_BAND[0]) & (gripper < HOLD_BAND[1])))
    if hold is None:
        raise ValueError("no hold plateau after the lid (red cube grasp)")
    if hold[1] >= len(gripper):
        raise ValueError("the red cube is never released")
    return StepEnds(int(lid[0]), int(lid[1]), int(hold[0]), int(hold[1]), len(gripper))

