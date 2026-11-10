"""Execution runtime: turns accepted commands and their timed action chunks into one action per control tick.

Owns what is physically active (docs/plan/v3-v1-spec.md §5). Robot-agnostic: everything robot-specific comes from the
`ActionSpec`. Called from three threads (owner: accept/revoke/renew; S1 runner: submit; control loop: tick), so every
public method takes the lock.

Rules:
- Only the current epoch, a known non-revoked command, and an unexpired lease are accepted; anything else is rejected
  with an ACK (`rejected` for commands, `chunk_rejected` for a single chunk), never silently applied. A chunk must use
  the robot's control period (`dt`); a policy with another rate is resampled in its adapter, never stretched here.
- A lease covers the whole life of a command: an active command whose lease ends is dropped in any mode.
- A command becomes active when its first action is applied (ACK `activated`); never earlier. A newer goal supersedes a
  pending one; a pending goal whose lease ends before its first action expires.
- Operations that do not move the robot (wait, abort, ask) hold at once and are active immediately.
- `Revoke` drops the command's future actions at once. A safety hold that takes authority away (owner lost, runtime
  error) ends the active and pending command with a `dropped` ACK; a starvation hold keeps the command.
- Irreversible switches (e.g. gripper) change only while RUNNING an active command's chunk that is not under review;
  in every other case they keep their last value.
- Review (`review`): after an instruction change the command keeps moving, its irreversible switches are frozen and its
  lease is shortened; a renewal or a new command ends the review.
- Decelerating hold (`stop_s` > 0): when motion turns into a hold, continuous dimensions with a numeric hold value
  ramp to it over `stop_s` instead of jumping (KEEP dimensions already repeat the last value).
- Actions whose time has passed are skipped, never replayed; a new chunk replaces the queue from its first due time.
- Starvation is measured, not inferred: a tick where an active moving command has no due action (running, or held by
  the watchdog as "starved") is `TickResult.starved` and counts in `starved_ticks`.
- Each new segment (activation or chunk splice) is inertialized: on its k-th tick it keeps 1 - s(k/ticks) of the jump
  in continuous dimensions (s = quintic smoothstep), gone after `blend_s`. The decaying offset rides on later actions
  of the segment, so it can push one past a bound; the runtime projects its own output onto the bounds.
- Bounds: the incoming action (chunk or hold) is checked and clamped, and every clamp is reported (`violations`: an
  upstream component sent an out-of-range action); the runtime's own blends and holds never count as violations.
"""
import dataclasses
import threading
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from rojev.protocol import AcceptedGoal, AckStatus, ExecutionAck, Revoke, TimedChunk, Transition
from rojev.robot.action_spec import ActionSpec


class Mode(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    TRANSITION = "transition"
    HOLD = "hold"
    STOPPED = "stopped"


@dataclass(frozen=True)
class TickResult:
    """One control tick: the `action` sent to the robot, from `source` "chunk" or "hold"; `cmd_id` is the active
    command; `index` counts applied actions; `skipped` counts due actions that had already passed; `held_switches` are
    irreversible switches kept at their last value; `violations` are clamps of the incoming action (not of the runtime's
    own blending); `starved` = an active moving command had no action due."""

    t: float
    action: np.ndarray
    mode: Mode
    cmd_id: int | None
    source: str
    index: int
    skipped: int = 0
    held_switches: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    hold_reason: str = ""
    review: bool = False
    starved: bool = False


@dataclass
class _Entry:
    """A queued action; `first` marks the start of a new segment (activation or splice), where a blend starts."""

    t: float
    action: np.ndarray
    cmd_id: int
    first: bool = False


@dataclass
class _Blend:
    offset: np.ndarray
    ticks: int
    k: int = 0


def _smoothstep5(s: float) -> float:
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


@dataclass
class ExecutionRuntime:
    """`last` is the last applied action; seed it with the robot's current command when the spec has KEEP dimensions
    (see `ActionSpec.hold`). `last_valid_t` is when a chunk was last accepted or a chunk action applied (watchdog
    supply). `acks` is the outbox the owner drains. `stop_s` = 0 holds instantly (default); > 0 decelerates."""

    spec: ActionSpec
    epoch: int
    last: np.ndarray | None = None
    blend_s: float = 0.2
    stop_s: float = 0.0
    mode: Mode = Mode.IDLE
    active: AcceptedGoal | None = None
    pending: AcceptedGoal | None = None
    hold_reason: str = ""
    applied_index: int = 0
    starved_ticks: int = 0
    last_valid_t: float | None = None
    acks: list[ExecutionAck] = field(default_factory=list)
    _queue: list[_Entry] = field(default_factory=list)
    _revoked: set[int] = field(default_factory=set)
    _max_cmd: int = -1
    _seq: int = 0
    _blend: _Blend | None = None
    _decel: _Blend | None = None
    _last_source: str = "hold"
    _review_cmd: int | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def accept(self, goal: AcceptedGoal, now: float) -> bool:
        with self._lock:
            problem = (f"epoch {goal.epoch} != {self.epoch}" if goal.epoch != self.epoch else
                       "stopped" if self.mode == Mode.STOPPED else
                       f"cmd {goal.cmd_id} revoked" if goal.cmd_id in self._revoked else
                       f"cmd {goal.cmd_id} not newer than {self._max_cmd}" if goal.cmd_id <= self._max_cmd else
                       "lease already expired" if goal.expires_t <= now else "")
            if problem:
                self._ack(goal.cmd_id, AckStatus.REJECTED, now, problem)
                return False
            self._max_cmd = goal.cmd_id
            if self.pending is not None:
                self._revoked.add(self.pending.cmd_id)
                self._ack(self.pending.cmd_id, AckStatus.REVOKED, now, f"superseded by {goal.cmd_id}")
            if not goal.operation.moves_robot():
                self.pending = None
                self._drop_queue()
                self.active = goal
                self._enter_hold(goal.operation.kind.value)
                self._ack(goal.cmd_id, AckStatus.ACTIVATED, now, "hold")
                return True
            self.pending = goal
            if goal.transition == Transition.PAUSE_HOLD or self.mode != Mode.RUNNING:
                self._drop_queue()
                self.mode = Mode.TRANSITION
            return True

    def renew(self, cmd_id: int, expires_t: float, now: float) -> bool:
        """Extend the lease of the active or pending command (the owner's CONTINUE)."""
        with self._lock:
            if not self._set_lease(cmd_id, lambda _: expires_t):
                self._ack(cmd_id, AckStatus.REJECTED, now, "renew: neither active nor pending, or revoked")
                return False
            if self._review_cmd == cmd_id:
                self._review_cmd = None
            return True

    def review(self, cmd_id: int, until_t: float, now: float) -> bool:
        """Put the active or pending command under review: it keeps moving, its irreversible switches keep their value,
        and its lease ends by `until_t` unless the owner renews it or replaces it with a new command."""
        with self._lock:
            if not self._set_lease(cmd_id, lambda current: min(current, until_t)):
                self._ack(cmd_id, AckStatus.REJECTED, now, "review: neither active nor pending, or revoked")
                return False
            self._review_cmd = cmd_id
            return True

    def under_review(self) -> bool:
        with self._lock:
            return self._review_cmd is not None and self.active is not None and self.active.cmd_id == self._review_cmd

    def revoke(self, r: Revoke, now: float) -> bool:
        with self._lock:
            if r.epoch != self.epoch:
                self._ack(r.cmd_id, AckStatus.REJECTED, now, f"revoke: epoch {r.epoch} != {self.epoch}")
                return False
            self._revoked.add(r.cmd_id)
            self._queue = [e for e in self._queue if e.cmd_id != r.cmd_id]
            if self.pending is not None and self.pending.cmd_id == r.cmd_id:
                self.pending = None
            if self.active is not None and self.active.cmd_id == r.cmd_id:
                self.active = None
                if self.pending is not None:
                    self.mode = Mode.TRANSITION
                else:
                    self._enter_hold(f"revoked: {r.reason}")
            elif self.pending is None and self.mode == Mode.TRANSITION:
                self._enter_hold(f"revoked: {r.reason}")
            self._ack(r.cmd_id, AckStatus.REVOKED, now, r.reason)
            return True

    def stop(self, now: float, reason: str) -> None:
        with self._lock:
            self._drop_queue()
            self.pending = None
            self.mode, self.hold_reason = Mode.STOPPED, reason
            self._ack(self.active.cmd_id if self.active else None, AckStatus.STOPPED, now, reason)

    def hold(self, now: float, reason: str, drop_commands: bool) -> None:
        """Hold still (the guard). Without `drop_commands` the active command keeps its authority and resumes when its
        chunks arrive again ("starved"). With it the hold takes authority away (the owner is gone, the runtime failed):
        the active and the pending command end with a `dropped` ACK, so the owner's record shows nothing running and
        it issues a new command when it can, instead of renewing one that can no longer move."""
        with self._lock:
            if drop_commands:
                for goal in (self.pending, self.active):
                    if goal is not None:
                        self._revoked.add(goal.cmd_id)
                        self._ack(goal.cmd_id, AckStatus.DROPPED, now, reason)
                self.pending = self.active = None
            if self.mode in (Mode.HOLD, Mode.STOPPED) and self.hold_reason == reason:
                return
            if self.mode == Mode.STOPPED:
                return
            self._drop_queue()
            self._enter_hold(reason)

    def submit(self, chunk: TimedChunk, now: float) -> bool:
        with self._lock:
            target = (self.pending if self.pending is not None and chunk.cmd_id == self.pending.cmd_id else
                      self.active if self.active is not None and chunk.cmd_id == self.active.cmd_id else None)
            actions = np.asarray(chunk.actions, dtype=float)
            problem = (f"epoch {chunk.epoch} != {self.epoch}" if chunk.epoch != self.epoch else
                       "stopped" if self.mode == Mode.STOPPED else
                       f"cmd {chunk.cmd_id} revoked" if chunk.cmd_id in self._revoked else
                       f"cmd {chunk.cmd_id} is neither active nor pending" if target is None else
                       "lease expired" if target.expires_t <= now else
                       f"chunk dt {chunk.dt} != control dt {self.spec.dt}"
                       if not np.isclose(chunk.dt, self.spec.dt) else
                       f"actions shape {actions.shape} != (H, {self.spec.size})"
                       if actions.ndim != 2 or actions.shape[1] != self.spec.size else
                       f"cmd {chunk.cmd_id} does not move the robot" if not target.operation.moves_robot() else
                       f"cmd {chunk.cmd_id} is paused for a transition"
                       if target is self.active and self.mode == Mode.TRANSITION else
                       f"holding: {self.hold_reason}"
                       if target is self.active and self.mode == Mode.HOLD and self.hold_reason != "starved" else "")
            if problem:
                self._ack(chunk.cmd_id, AckStatus.CHUNK_REJECTED, now, problem)
                return False
            entries = [_Entry(chunk.t_start + k * chunk.dt, a, chunk.cmd_id) for k, a in enumerate(actions)]
            entries = [e for e in entries if e.t >= now - self.spec.dt / 2]
            if not entries:
                self._ack(chunk.cmd_id, AckStatus.CHUNK_REJECTED, now, "chunk entirely in the past")
                return False
            if target is self.active and self.pending is not None:
                takeover = min((e.t for e in self._queue if e.cmd_id == self.pending.cmd_id), default=None)
                if takeover is not None:
                    entries = [e for e in entries if e.t < takeover - self.spec.dt / 2]
                if not entries:
                    return True
            entries[0].first = True
            t0, t_end = entries[0].t - self.spec.dt / 2, entries[-1].t + self.spec.dt / 2
            if target is self.pending:
                active = self.active.cmd_id if self.active is not None else None
                keep = [e for e in self._queue if e.t < t0 and e.cmd_id == active]
            else:
                keep = [e for e in self._queue if e.cmd_id != chunk.cmd_id and e.t >= t_end] + \
                       [e for e in self._queue if e.cmd_id == chunk.cmd_id and e.t < t0]
            self._queue = sorted(keep + entries, key=lambda e: e.t)
            if target is self.active and self.mode == Mode.HOLD and self.hold_reason == "starved":
                self.mode, self.hold_reason = Mode.RUNNING, ""
            self.last_valid_t = now
            return True

    def target(self) -> AcceptedGoal | None:
        """The command that needs chunks: the pending one, else the active one while it moves (or is starved)."""
        with self._lock:
            if self.pending is not None:
                return self.pending
            moving = self.mode == Mode.RUNNING or (self.mode == Mode.HOLD and self.hold_reason == "starved")
            if self.active is not None and moving and self.active.operation.moves_robot():
                return self.active
            return None

    def active_goal(self) -> AcceptedGoal | None:
        """The active command (None when nothing is active), read under the lock."""
        with self._lock:
            return self.active

    def last_action(self) -> np.ndarray | None:
        """A copy of the action last applied (None before the first, unless seeded)."""
        with self._lock:
            return None if self.last is None else self.last.copy()

    def queued_from(self, cmd_id: int, t: float) -> np.ndarray | None:
        """The queued actions of `cmd_id` due at or after `t`, in time order, as the chunks gave them (before blends
        and bounds); None if there are none."""
        with self._lock:
            actions = [e.action for e in self._queue if e.cmd_id == cmd_id and e.t >= t - self.spec.dt / 2]
        return np.stack(actions) if actions else None

    def queued_end(self, cmd_id: int) -> float | None:
        """When the queued actions of `cmd_id` run out: the last one's due time plus its control period (None if
        nothing is queued for it)."""
        with self._lock:
            last = max((e.t for e in self._queue if e.cmd_id == cmd_id), default=None)
        return None if last is None else last + self.spec.dt

    def tick(self, now: float) -> TickResult:
        with self._lock:
            if self.active is not None and self.mode != Mode.STOPPED and now >= self.active.expires_t:
                self._ack(self.active.cmd_id, AckStatus.EXPIRED, now, "lease expired")
                self._revoked.add(self.active.cmd_id)
                self._queue = [e for e in self._queue if e.cmd_id != self.active.cmd_id]
                self.active = None
                if self.mode in (Mode.RUNNING, Mode.HOLD):
                    self._blend = None
                    self._enter_hold("lease expired")
            if self.pending is not None and now >= self.pending.expires_t:
                self._ack(self.pending.cmd_id, AckStatus.EXPIRED, now, "lease expired before the first action")
                self._revoked.add(self.pending.cmd_id)
                self._queue = [e for e in self._queue if e.cmd_id != self.pending.cmd_id]
                self.pending = None
                if self.mode == Mode.TRANSITION:
                    self._enter_hold("pending lease expired")
            half = self.spec.dt / 2
            skipped = 0
            while self._queue and self._queue[0].t < now - half:
                if self._queue[0].first and len(self._queue) > 1:
                    self._queue[1].first = True
                self._queue.pop(0)
                skipped += 1
            due = self._queue.pop(0) if self._queue and self._queue[0].t <= now + half else None
            if due is not None and self.pending is not None and due.cmd_id == self.pending.cmd_id:
                self.active, self.pending = self.pending, None
                self.mode, self.hold_reason = Mode.RUNNING, ""
                self._ack(self.active.cmd_id, AckStatus.ACTIVATED, now, "first action applied")
            if due is not None and self.mode == Mode.RUNNING:
                action, violations = self.spec.limit(due.action.copy())
                source = "chunk"
                ticks = round(self.blend_s * self.spec.control_hz)
                if due.first and self.last is not None and ticks > 0:
                    self._blend = _Blend(np.where(self.spec.continuous, self.last - action, 0.0), ticks)
                self.last_valid_t = now
            else:
                action, violations = self.spec.limit(self.spec.hold(self.last))
                source = "hold"
                self._blend = None
            if self._blend is not None:
                self._blend.k += 1
                action = action + self._blend.offset * (1.0 - _smoothstep5(self._blend.k / self._blend.ticks))
                if self._blend.k >= self._blend.ticks:
                    self._blend = None
            if source == "hold":
                ticks = round(self.stop_s * self.spec.control_hz)
                if self._last_source == "chunk" and self.last is not None and ticks > 0:
                    self._decel = _Blend(np.where(self.spec.continuous, self.last - action, 0.0), ticks)
                if self._decel is not None:
                    self._decel.k += 1
                    action = action + self._decel.offset * (1.0 - _smoothstep5(self._decel.k / self._decel.ticks))
                    if self._decel.k >= self._decel.ticks:
                        self._decel = None
            else:
                self._decel = None
            self._last_source = source
            moving = self.active is not None and self.active.operation.moves_robot()
            starved = source == "hold" and moving and (
                self.mode == Mode.RUNNING or (self.mode == Mode.HOLD and self.hold_reason == "starved"))
            self.starved_ticks += starved
            reviewing = self.under_review()
            held = () if source == "chunk" and not reviewing else tuple(self.spec.irreversible_change(self.last, action))
            if held:
                action = self.spec.keep_switches(self.last, action, list(held))
            action = np.clip(action, self.spec.low, self.spec.high)
            self.last = action
            self.applied_index += 1
            return TickResult(now, action, self.mode, self.active.cmd_id if self.active else None, source,
                              self.applied_index, skipped, held, tuple(violations), self.hold_reason, reviewing,
                              starved)

    def _set_lease(self, cmd_id: int, lease) -> bool:
        """Apply `lease(current_expiry)` to the active or pending command `cmd_id`; False if it is neither or revoked."""
        if cmd_id in self._revoked:
            return False
        if self.active is not None and self.active.cmd_id == cmd_id:
            self.active = dataclasses.replace(self.active, expires_t=lease(self.active.expires_t))
        elif self.pending is not None and self.pending.cmd_id == cmd_id:
            self.pending = dataclasses.replace(self.pending, expires_t=lease(self.pending.expires_t))
        else:
            return False
        return True

    def _enter_hold(self, reason: str) -> None:
        self.mode, self.hold_reason = Mode.HOLD, reason

    def _drop_queue(self) -> None:
        self._queue.clear()
        self._blend = None

    def _ack(self, cmd_id: int | None, status: AckStatus, now: float, detail: str) -> None:
        self._seq += 1
        self.acks.append(ExecutionAck(self.epoch, cmd_id, self._seq, status, now, self.applied_index, detail))

    def drain_acks(self) -> list[ExecutionAck]:
        with self._lock:
            out, self.acks = self.acks, []
            return out
