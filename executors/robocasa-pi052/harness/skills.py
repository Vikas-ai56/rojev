"""Scripted S1 for RoboCasa PandaOmron: a stand-in for π0.5² while no GPU is available.

SPOOFS (declared): it reads object and fixture poses, grasp and contact state from the simulator instead of seeing
them; it plans motion with a hand-written controller instead of a learned policy; its K samples differ only by injected
target noise; its latency is a sleep. Everything it produces goes through the real runtime unchanged (the
`rojev.system1.Policy` contract); the core does not know it is scripted.

It turns the subtask text into a skill and the skill into a 50-action chunk of the arm's own commands (OSC position
deltas in the robot base frame, `embodiment.PANDA_OMRON`). Each chunk is planned forward from the measured hand pose:
the skill picks a waypoint and gripper state from the predicted state, a proportional step moves the predicted hand by
`GAIN` metres per unit command per tick (measured on this robot: about 0.19 m for 20 ticks at full command, ramp
included), so later actions in the chunk continue where earlier ones lead. Skills are stateless: the phase comes from
the state (hand pose, gripper width, simulator grasp and contact checks), so a retry, a revision or an interruption
resumes from wherever the arm is.

  pick up the X                    above X → down to its grasp height (or first contact) → close → lift;
                                   lifted without X = failed grasp → open and start again
  place the X on/in the <shelf>    above the target → down → open → up    (top / middle / bottom shelf: RoboCasa's
                                   fridge rack regions; microwave: its tray; "back in the fridge": X's plate)
  press the start/stop button      in front of the button along the microwave's front normal → press → back off
  anything else                    hold still (`info["skill"] = None`)

Limits: orientation is held, never steered (OSC zero rotation command); the base never drives, so a target beyond the
arm's reach (`info["reach_m"]`) is approached as far as the controller allows and the step stays unfinished.
"""
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from robocasa.utils import object_utils as OU

from embodiment import PANDA_OMRON
from rojev.protocol import Observation
from rojev.system1 import ActionChunk, Policy

GAIN = 0.0125
TAU = 4.0
OPEN_WIDTH, CLOSED_WIDTH, WIDTH_RATE = 0.02, 0.001, 0.0015
GRASP_DEPTH, ABOVE, LIFT, SHELF_CLEARANCE = 0.03, 0.10, 0.12, 0.03
FAILED_LIFT = 0.04
PRESS_STANDOFF, PRESS_DEPTH = 0.08, 0.01
REACHED = 0.012
REACH_M = 0.85
SHELVES = {"top shelf": -1, "middle shelf": -2, "bottom shelf": 0}


def base_frame(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(base position, base rotation) in the world, from the observation state (agent_pos 7:14, quaternion xyzw)."""
    x, y, z, w = state[10:14]
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return np.asarray(state[7:10], dtype=float), rot


@dataclass
class Predicted:
    """The state a chunk is planned from and rolled forward: hand position (base frame), gripper half-width, whether
    the object is held, whether the hand touches it, whether the step's goal already holds."""

    hand: np.ndarray
    width: float
    holding: bool
    contact: bool
    done: bool


@dataclass(frozen=True)
class Pick:
    target: np.ndarray

    def step(self, s: Predicted) -> tuple[np.ndarray, bool]:
        above = self.target + [0, 0, ABOVE]
        if s.holding:
            return self.target + [0, 0, LIFT], True
        if s.width < OPEN_WIDTH / 2:
            failed = s.hand[2] > self.target[2] + FAILED_LIFT
            return (above, False) if failed else (self.target, True)
        if np.linalg.norm(s.hand[:2] - self.target[:2]) > REACHED:
            return above, False
        if s.hand[2] > self.target[2] + REACHED and not s.contact:
            return self.target, False
        return s.hand, True


@dataclass(frozen=True)
class Place:
    target: np.ndarray

    def step(self, s: Predicted) -> tuple[np.ndarray, bool]:
        above = self.target + [0, 0, LIFT]
        if not s.holding:
            return above, False
        if np.linalg.norm(s.hand[:2] - self.target[:2]) > REACHED:
            return above, True
        if s.hand[2] > self.target[2] + REACHED:
            return self.target, True
        return self.target, False


@dataclass(frozen=True)
class Press:
    target: np.ndarray
    normal: np.ndarray

    def step(self, s: Predicted) -> tuple[np.ndarray, bool]:
        before, into = self.target + PRESS_STANDOFF * self.normal, self.target - PRESS_DEPTH * self.normal
        rel = s.hand - self.target
        along = float(np.dot(rel, self.normal))
        off_line = np.linalg.norm(rel - along * self.normal) > REACHED or along > PRESS_STANDOFF + REACHED
        return (before if s.done or off_line else into), True

    def shifted(self, offset: np.ndarray) -> "Press":
        return Press(self.target + offset, self.normal)


SKILLS = (
    (re.compile(r"^pick up the (?P<obj>.+)$"), "pick"),
    (re.compile(r"^place the (?P<obj>.+?) (?:in|on|back in|back on) the (?P<dest>.+)$"), "place"),
    (re.compile(r"^press the (?P<which>start|stop) button$"), "press"),
)


class SkillScriptPolicy(Policy):
    """`world(check)` runs `check(kitchen)` under the simulator lock (`RoboCasaBackend.truth`). `noise_m` = target noise
    per sample (0: every sample identical), `latency_s` = simulated inference time of one batched request."""

    def __init__(self, world: Callable[[Callable], object], latency_s: float = 0.3, noise_m: float = 0.0,
                 horizon: int = 50, seed: int = 0):
        self.world, self.latency_s, self.noise_m, self.horizon = world, latency_s, noise_m, horizon
        self.rng = np.random.default_rng(seed)

    def act(self, obs: Observation, subtask: str, timeout_s: float | None = None) -> ActionChunk:
        return self.act_samples(obs, subtask, 1)[0]

    def act_samples(self, obs: Observation, subtask: str, k: int, timeout_s: float | None = None) -> list[ActionChunk]:
        t0 = time.perf_counter()
        skill, start, info = self.world(lambda kitchen: self._read(kitchen, np.asarray(obs.state, dtype=float), subtask))
        time.sleep(max(0.0, self.latency_s - (time.perf_counter() - t0)))
        chunks = []
        for i in range(k):
            noisy = None if skill is None else self._perturb(skill)
            state = Predicted(start.hand.copy(), start.width, start.holding, start.contact, start.done)
            chunks.append(ActionChunk(self._plan(noisy, state), {"infer": self.latency_s * 1000}, {**info, "sample": i}))
        return chunks

    def reset(self) -> None:
        pass

    def _perturb(self, skill):
        offset = self.rng.normal(0.0, self.noise_m, 3)
        return skill.shifted(offset) if isinstance(skill, Press) else type(skill)(skill.target + offset)

    def _read(self, kitchen, state: np.ndarray, subtask: str):
        """Privileged read (the spoof): the skill and its target in the base frame, plus the predicted start state."""
        origin, rot = base_frame(state)

        def to_base(p) -> np.ndarray:
            return rot.T @ (np.asarray(p, dtype=float) - origin)

        m, d = kitchen.sim.model, kitchen.sim.data
        keys = {kitchen.get_obj_lang(name): name for name in kitchen.objects}
        width = float(state[14] - state[15]) / 2
        for pattern, kind in SKILLS:
            match = pattern.match(subtask or "")
            if match is None:
                continue
            fields = match.groupdict()
            obj = keys.get(fields.get("obj"))
            holding = obj is not None and bool(OU.check_obj_grasped(kitchen, obj))
            contact = obj is not None and bool(kitchen.check_contact(kitchen.robots[0].gripper["right"],
                                                                     kitchen.objects[obj]))
            start = Predicted(np.asarray(state[:3], dtype=float), width, holding, contact, False)
            if kind == "pick" and obj is not None:
                o = kitchen.objects[obj]
                top = d.body_xpos[m.body_name2id(o.root_body)] + np.asarray(o.top_offset)
                height = float(o.top_offset[2] - o.bottom_offset[2])
                target = to_base(top - [0, 0, min(GRASP_DEPTH, height / 2)])
                return Pick(target), start, self._info(kind, subtask, target)
            if kind == "place" and obj is not None:
                target = self._destination(kitchen, fields["dest"], obj, to_base)
                if target is not None:
                    return Place(target), start, self._info(kind, subtask, target)
            if kind == "press":
                mw = kitchen.microwave
                body = m.body_name2id(mw.root_body)
                normal = rot.T @ (d.body_xmat[body].reshape(3, 3) @ np.array([0.0, -1.0, 0.0]))
                target = to_base(d.geom_xpos[m.geom_name2id(f"{mw.name}_{fields['which']}_button")])
                start.done = bool(mw.get_state()["turned_on"]) == (fields["which"] == "start")
                return Press(target, normal), start, self._info(kind, subtask, target)
        start = Predicted(np.asarray(state[:3], dtype=float), width, False, False, False)
        return None, start, {"skill": None, "subtask": subtask}

    @staticmethod
    def _destination(kitchen, dest: str, obj: str, to_base) -> np.ndarray | None:
        m, d = kitchen.sim.model, kitchen.sim.data
        o = kitchen.objects[obj]
        lift = float(-o.bottom_offset[2]) + SHELF_CLEARANCE
        shelf = next((index for name, index in SHELVES.items() if name in dest), None)
        if shelf is not None:
            fridge = kitchen.fridge
            regions = fridge.get_reset_regions(kitchen, compartment="fridge", rack_index=shelf, reg_type=("shelf",))
            offset = np.asarray(next(iter(regions.values()))["offset"], dtype=float)
            body = m.body_name2id(fridge.root_body)
            return to_base(d.body_xpos[body] + d.body_xmat[body].reshape(3, 3) @ offset + [0, 0, lift])
        if "microwave" in dest:
            mw = kitchen.microwave
            corners = np.array(mw.get_int_sites()["tray"])
            body = m.body_name2id(mw.root_body)
            return to_base(d.body_xpos[body] + d.body_xmat[body].reshape(3, 3) @ corners.mean(axis=0) + [0, 0, lift])
        if ("fridge" in dest or "plate" in dest) and f"{obj}_container" in kitchen.objects:
            plate = kitchen.objects[f"{obj}_container"]
            return to_base(d.body_xpos[m.body_name2id(plate.root_body)] + [0, 0, lift])
        return None

    @staticmethod
    def _info(kind: str, subtask: str, target: np.ndarray) -> dict:
        reach = float(np.linalg.norm(target))
        return {"skill": kind, "subtask": subtask, "target": np.round(target, 3).tolist(), "reach_m": round(reach, 3),
                "reachable": reach <= REACH_M}

    def _plan(self, skill, s: Predicted) -> np.ndarray:
        actions = np.zeros((self.horizon, PANDA_OMRON.size), dtype=np.float32)
        actions[:, 6] = -1.0 if s.width > OPEN_WIDTH / 2 else 1.0
        actions[:, 11] = -1.0
        if skill is None:
            return actions
        closed_ticks = 0
        for k in range(self.horizon):
            waypoint, close = skill.step(s)
            command = np.clip((waypoint - s.hand) / (GAIN * TAU), -1.0, 1.0)
            s.hand = s.hand + GAIN * command
            s.width = max(CLOSED_WIDTH, s.width - WIDTH_RATE) if close else min(OPEN_WIDTH, s.width + WIDTH_RATE)
            closed_ticks = closed_ticks + 1 if close and s.width <= CLOSED_WIDTH + WIDTH_RATE else 0
            if isinstance(skill, Pick) and closed_ticks >= 4 and np.linalg.norm(s.hand - skill.target) < 2 * REACHED:
                s.holding = True
            if isinstance(skill, Place) and not close:
                s.holding = False
            actions[k, :3] = command
            actions[k, 6] = 1.0 if close else -1.0
        return actions
