"""
In short here we observe robot state (observation) and decide how much amount of actions should be committed 
to the robot. There are 3 types of commitments, ConsensusCommit (needs to be activated by the user), PhaseCommit, 
FixedCommit. 

<BELOW Part is talking abt network latency>
A commitment is never shorter than the `floor` the caller passes (or the chunk). The floor is not
a policy parameter: it depends on the policy's latency, which only the S1 runner measures (`S1Runner.floor`: with one
request in flight the robot runs without gaps only if the commitment covers twice the latency plus a tick). Below it
the robot starves, or livelocks when the whole commitment elapses during inference.
</end>

Irreversible switches (gripper) under `speculative_predict`: if the candidates disagree on one at step k, the
commitment stops before it when k ≥ `floor` (the next request decides it closer to the event, on a fresher observation);
otherwise the switch is inside the floor and the majority of candidates decides it, step by step: at each step a switch
takes the value a strict majority of candidates holds there; where none does (a tie, possible with even K) no value is
backed, so the value in effect stays until a majority changes it: the previous step's, and at the chunk's first step
the robot's current one (the switch in `last`, the action last applied, which the caller passes). The committed
candidate is one that follows this majority sequence. Only before anything was applied (`last` None) can a first-step
tie have no value in effect; until a majority forms any candidate's value is accepted there, and the reason says so.

  FixedCommit       the whole chunk (up to `max_commit`): the default
  PhaseCommit       up to the first low-speed point of the predicted motion (PACE, arXiv 2606.00537): long commitments
                    in free motion, short ones where motion slows toward contact
  ConsensusCommit   `speculative_predict`: K candidates from the same observation; the prefix where they agree, from the
                    candidate closest to the majority's mean, while a majority of candidates still moves like it (CA³C,
                    arXiv 2610.07949, without its world model: agreement is measured on the actions' motion, not on
                    predicted consequences); then capped by `then`

Motion is robot-agnostic: velocity-like dimensions (numeric hold value) use their distance from the hold value,
position-like dimensions (`KEEP`) the step-to-step change; both are compared relative to the chunks' median speed
(when that is zero, any motion is disagreement).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rojev.robot import KEEP, ActionSpec


@dataclass(frozen=True)
class Commitment:
    """`actions` = the committed prefix of the chosen candidate: `chosen` = candidate index; `reason` and `detail` are logged."""

    actions: np.ndarray
    chosen: int
    reason: str
    detail: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])


def motion(spec: ActionSpec, actions: np.ndarray) -> np.ndarray:
    """Per-step motion over the continuous dimensions, shape (H, continuous dims) (see module docstring)."""
    actions = np.asarray(actions, dtype=float)
    continuous = spec.continuous
    hold = np.array([0.0 if d.hold == KEEP else float(d.hold) for d in spec.dims])
    position = np.array([d.hold == KEEP for d in spec.dims])
    step = np.vstack([np.zeros((1, actions.shape[1])), np.diff(actions, axis=0)])
    return np.where(position, step, actions - hold)[:, continuous]


def speed_profile(spec: ActionSpec, actions: np.ndarray) -> np.ndarray:
    """Per-step motion magnitude."""
    return np.linalg.norm(motion(spec, actions), axis=1)


def irreversible_states(spec: ActionSpec, actions: np.ndarray) -> np.ndarray:
    """Per-step on/off state of each irreversible switch, shape (H, irreversible dims)."""
    columns = [i for i, d in enumerate(spec.dims) if d.irreversible]
    return np.array([spec.switches(a)[columns] for a in np.asarray(actions)]).reshape(len(actions), len(columns))


def first_switch_disagreement(spec: ActionSpec, candidates: np.ndarray) -> int | None:
    """First step at which the candidates disagree on an irreversible switch (None if they never do)."""
    states = np.array([irreversible_states(spec, c) for c in candidates])
    differs = (states != states[0]).any(axis=(0, 2))
    return int(np.argmax(differs)) if differs.any() else None


def majority_switches(states: np.ndarray, in_effect: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per step and switch of `states` (K candidates × steps × switches, bool): the value a strict majority holds,
    else the value in effect (the previous step's; before the first step `in_effect`, the robot's current switches, or
    unknown if None). Returns (value, known, tied): `known` is False from an unknown tie until a strict majority forms;
    `tied` marks steps without a strict majority."""
    k = states.shape[0]
    on = states.sum(axis=0)
    decided_on, decided_off = 2 * on > k, 2 * (k - on) > k
    tied = ~(decided_on | decided_off)
    value, known = np.zeros(on.shape, dtype=bool), np.zeros(on.shape, dtype=bool)
    start_value = np.zeros(on.shape[1:], dtype=bool) if in_effect is None else np.asarray(in_effect, dtype=bool)
    start_known = np.full(on.shape[1:], in_effect is not None)
    for t in range(on.shape[0]):
        before_value = value[t - 1] if t else start_value
        before_known = known[t - 1] if t else start_known
        value[t] = np.where(tied[t], before_value, decided_on[t])
        known[t] = ~tied[t] | before_known
    return value, known, tied


class CommitPolicy(ABC):
    """`samples` = how many candidates the S1 runner requests per observation; `plan` never commits fewer than
    `min(floor, horizon)` actions. `last` is the action last applied to the robot (None before the first)."""

    samples: int = 1

    @abstractmethod
    def plan(self, candidates: list[np.ndarray], spec: ActionSpec, floor: int,
             last: np.ndarray | None) -> Commitment: ...


@dataclass
class FixedCommit(CommitPolicy):
    """The whole chunk, or `max_commit` actions (raised to the floor: a cap below it would starve the robot)."""

    max_commit: int | None = None
    samples: int = 1

    def plan(self, candidates: list[np.ndarray], spec: ActionSpec, floor: int,
             last: np.ndarray | None) -> Commitment:
        actions = np.asarray(candidates[0], dtype=float)
        horizon = actions.shape[0]
        n = horizon if self.max_commit is None else min(horizon, max(self.max_commit, floor))
        return Commitment(actions[:n], 0, "fixed", {"horizon": int(actions.shape[0])})


@dataclass
class PhaseCommit(CommitPolicy):
    """Commit through the first low-speed point after the floor: a local speed minimum below `slow_fraction` of the
    chunk's peak speed."""

    slow_fraction: float = 0.3
    max_commit: int | None = None
    samples: int = 1

    def boundary(self, speed: np.ndarray, floor: int) -> int:
        peak = float(speed.max()) if speed.size else 0.0
        end = speed.size if self.max_commit is None else min(speed.size, max(self.max_commit, floor))
        if peak <= 0.0:
            return end
        for k in range(max(1, min(floor, end)), end - 1):
            if speed[k] <= self.slow_fraction * peak and speed[k] <= speed[k - 1] and speed[k] <= speed[k + 1]:
                return k + 1
        return end

    def plan(self, candidates: list[np.ndarray], spec: ActionSpec, floor: int,
             last: np.ndarray | None) -> Commitment:
        actions = np.asarray(candidates[0], dtype=float)
        n = self.boundary(speed_profile(spec, actions), floor)
        return Commitment(actions[:n], 0, "phase boundary" if n < actions.shape[0] else "no slow point",
                          {"horizon": int(actions.shape[0])})


@dataclass
class ConsensusCommit(CommitPolicy):
    """Commit the prefix where a majority of the K candidates agrees with the committed one. The committed candidate is,
    among the candidates following the majority's switch values (module docstring), the one closest to their mean. A
    candidate agrees at step k while its motion differs from the committed one's by at most `tolerance` × the median
    candidate speed (smoothed over the last `smooth` steps); the prefix ends at the first step after the floor where
    fewer than a strict majority (counting the committed one) agree, so one outlier among three cannot shorten it.
    `detail["votes"]` = the fewest candidates backing the committed switch values at any step; `then` (Fixed or Phase)
    caps the result."""

    samples: int = 3
    tolerance: float = 0.5
    smooth: int = 3
    then: CommitPolicy = field(default_factory=FixedCommit)

    def __post_init__(self):
        if self.samples < 2:
            raise ValueError("speculative_predict needs at least 2 samples to compare")
        if self.smooth < 1 or self.tolerance <= 0:
            raise ValueError("smooth must be at least 1 and tolerance positive")

    def choose(self, spec: ActionSpec, c: np.ndarray, window: int, last: np.ndarray | None) -> tuple[int, int, str]:
        """Over the first `window` steps: the committed candidate, the fewest candidates backing the majority's switch
        values at any step where they are known, and how the switches were decided (the commitment's reason when a
        disputed switch lies inside the floor; "switch without a majority" = a tie with no value in effect, or no
        candidate follows the majority)."""
        states = np.array([irreversible_states(spec, x[:window]) for x in c])
        in_effect = None if last is None else irreversible_states(spec, np.asarray(last)[None])[0]
        value, known, tied = majority_switches(states, in_effect)
        backs = ((states == value) | ~known).all(axis=2)
        departs = np.array([window if b.all() else int(np.argmin(b)) for b in backs])
        group = departs == departs.max()
        mean = c[group].mean(axis=0)
        distance = ((c - mean)[:, :window][:, :, spec.continuous] ** 2).sum(axis=(1, 2))
        distance[~group] = np.inf
        steps = known.all(axis=1)
        votes = int((states[:, steps] == value[steps]).all(axis=2).sum(axis=0).min()) if steps.any() else 0
        decided = ("switch without a majority" if not known.all() or departs.max() < window else
                   "switch tie keeps the value in effect" if tied.any() else "switch decided by majority")
        return int(np.argmin(distance)), votes, decided

    def agreeing(self, spec: ActionSpec, c: np.ndarray, chosen: int) -> np.ndarray:
        """Per step, how many candidates (the committed one included) move like the committed one."""
        moves = np.array([motion(spec, x) for x in c])
        scale = float(np.median(np.linalg.norm(moves, axis=2).mean(axis=1)))
        diff = np.linalg.norm(moves - moves[chosen], axis=2)
        gap = diff / scale if scale > 0 else np.where(diff > 0, np.inf, 0.0)
        kernel = np.ones(self.smooth) / self.smooth
        gap = np.array([np.convolve(g, kernel, mode="full")[:c.shape[1]] for g in gap])
        return (gap <= self.tolerance).sum(axis=0)

    def plan(self, candidates: list[np.ndarray], spec: ActionSpec, floor: int,
             last: np.ndarray | None) -> Commitment:
        c = np.asarray(candidates, dtype=float)
        k_total, horizon = c.shape[0], c.shape[1]
        floor = max(1, min(floor, horizon))
        switch = first_switch_disagreement(spec, c)
        switch_limit = horizon if switch is None else max(switch, floor)
        chosen, votes, decided = self.choose(spec, c, switch_limit, last)
        agreeing = self.agreeing(spec, c, chosen)
        agreed = next((k for k in range(floor, horizon) if 2 * agreeing[k] <= k_total), horizon)
        window = min(agreed, switch_limit)
        then = self.then.plan([c[chosen]], spec, floor, last)
        n = min(window, then.steps)
        reason = ("candidates agree" if n == horizon else
                  then.reason if then.steps < window else
                  "candidates diverge" if switch_limit > agreed else
                  "stop before a disputed switch" if switch >= floor else decided)
        return Commitment(c[chosen][:n], chosen, reason,
                          {"horizon": horizon, "agreed": int(agreed), "capped": int(then.steps), "switch": switch,
                           "votes": f"{votes}/{k_total}", "min_agreeing": int(agreeing[floor:n].min()) if n > floor
                           else int(agreeing[:n].min())})


COMMIT_POLICIES = ("fixed", "phase")
_PARAMS = {"fixed": {"max_commit"}, "phase": {"slow_fraction", "max_commit"}}
_CONSENSUS_PARAMS = {"samples", "tolerance", "smooth"}


def make_commit_policy(commit_policy: str = "fixed", speculative_predict: bool = False, **params) -> CommitPolicy:
    """The plug-and-play switch (spec §14): `commit_policy` picks the horizon rule; `speculative_predict` wraps it in
    `ConsensusCommit` (K samples). A name or parameter that does not apply raises instead of being ignored."""
    if commit_policy not in COMMIT_POLICIES:
        raise ValueError(f"commit_policy {commit_policy!r} not in {COMMIT_POLICIES}")
    allowed = _PARAMS[commit_policy] | (_CONSENSUS_PARAMS if speculative_predict else set())
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(f"parameters {sorted(unknown)} do not apply to commit_policy={commit_policy!r}, "
                         f"speculative_predict={speculative_predict}")
    base_cls = PhaseCommit if commit_policy == "phase" else FixedCommit
    base = base_cls(**{k: v for k, v in params.items() if k in _PARAMS[commit_policy]})
    if not speculative_predict:
        return base
    return ConsensusCommit(then=base, **{k: v for k, v in params.items() if k in _CONSENSUS_PARAMS})
