"""S1 runner: keeps the command that needs actions supplied with chunks from an S1 policy, on its own thread.

It asks the policy (any `Policy`: local, or a remote GPU service) for a chunk on the newest observation whenever the
target command's queue reaches less than `refill_s` ahead of now, and places the chunk on that observation's time:
action k is due at `t_capture + k · dt`. Actions whose time passed during inference are skipped by the runtime, so a
slow policy costs its latency, never stale motion. A revoked or replaced command's late chunk is rejected by the
runtime. The policy sees the command's instruction (`AcceptedGoal.instruction`), so a changed instruction never
reaches S1 as stale task text. Policy errors are counted and retried; they never stop the loop (the guard holds
the robot meanwhile).

Liveness is derived from measured latency, not configured. Latency L = how old the observation is when its chunk is
submitted (robot clock; includes K-sample cost and the network), estimated as the largest of the last
`latency_window` requests. A chunk captured at c and committed for n steps covers [c, c + n·dt); the next request
starts when less than the lead time is queued and its chunk lands L later, so the robot runs without gaps only if the
lead time ≥ L and n·dt ≥ 2L (plus a tick of quantization). The runner therefore requests when less than
`max(refill_s, L + dt)` is queued and passes `floor = ⌈2L/dt⌉ + 1` to the commitment policy; before the first
measurement the floor is the whole chunk. Gaps that still happen (a slower request than any before) are counted by the
runtime (`ExecutionRuntime.starved_ticks`).

How much of each chunk is committed is the `CommitPolicy`'s decision (`commit.py`; default `FixedCommit` = the whole
chunk). With `speculative_predict` it asks for K candidates per observation and commits only their agreed prefix, so
the next request comes sooner where the candidates disagree.

`refill_now(cmd_id)` (the execution monitor's deviation) makes the next step request a chunk on the newest observation
even with actions still queued, if `cmd_id` is the command being supplied; the new chunk replaces the queue from its
first due time. A deviation observed under another command (an older one, or none) is not this command's and is ignored
(counted).

A derived floor above the chunk's length means even the whole chunk cannot cover two latencies: commitment cannot keep
up with the policy and the robot will starve between chunks. Each such request is counted (`floor_over_chunk`).

Each request has a deadline: the duration of the last chunk (its horizon × dt). A chunk's actions are placed at its
observation's time, so an answer older than that is entirely in the past and useless; waiting longer only starves the
robot (a stalled connection once held it for 105 s). On `TimeoutError` the request is abandoned, counted (`timeouts`)
and the next one is made at once on the newest observation, with no back-off. Before the first chunk the horizon is
unknown and the policy's own timeout applies (a cold first request is slow). With `deadline=False` (the robot clock
stands still while the policy thinks: lockstep) no answer can be stale, so requests carry no deadline.

Inference delay (real-time chunking): for a policy that `accepts_continuation`, each request after the first latency
measurement carries the command's actions already queued from the observation's time and `delay_steps` = ⌈L/dt⌉,
the actions that run before the chunk lands, so the policy can make its chunk agree with them (counted:
`continuations`). Placement and skipping are unchanged; continuing is the policy's part.

When the command leaves the subtask to S1 (`operation.subtask` None, the policy plans its own steps) and the policy
reports the subtask it chose (`ActionChunk.info["subtask"]`), a change from one chunk to the next under the same
command means S1 considers the previous subtask done: it is posted as a `StepEvent` (source `s1-planner`) through
`on_step`, so the owner records the progress and asks the planning lane. Every subtask S1's planner chooses (the first
under a command, and each change) is also reported as a `PlannedSubtask` through `on_plan`: a proposal a planning
decider may commit the robot to.
"""
import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from rojev.control import ExecutionRuntime
from rojev.protocol import AcceptedGoal, Capture, PlannedSubtask, StepEvent, TimedChunk

from .commit import CommitPolicy, FixedCommit
from .policy import Continuation, Policy


@dataclass
class S1Stats:
    requests: int = 0
    accepted: int = 0
    errors: int = 0
    last_error: str = ""
    infer_ms: list[float] = field(default_factory=list)
    obs_age_s: list[float] = field(default_factory=list)
    info: list[dict] = field(default_factory=list)
    committed: list[int] = field(default_factory=list)
    commit_reasons: dict[str, int] = field(default_factory=dict)
    forced: int = 0
    refill_ignored: int = 0
    floors: list[int] = field(default_factory=list)
    floor_over_chunk: int = 0
    planner_steps: int = 0
    timeouts: int = 0
    continuations: int = 0


class S1Runner:
    def __init__(self, runtime: ExecutionRuntime, policy: Policy, latest: Callable[[], Capture],
                 now: Callable[[], float], refill_s: float, poll_s: float = 0.005, error_backoff_s: float = 0.5,
                 commit: CommitPolicy | None = None, latency_window: int = 20,
                 on_step: Callable[[StepEvent], None] | None = None,
                 on_plan: Callable[[PlannedSubtask], None] | None = None, deadline: bool = True):
        self.runtime, self.policy, self.latest, self.now = runtime, policy, latest, now
        self.deadline = deadline
        self.refill_s, self.poll_s, self.error_backoff_s = refill_s, poll_s, error_backoff_s
        self.commit = commit or FixedCommit()
        self.on_step, self.on_plan = on_step, on_plan
        self.stats = S1Stats()
        self._planned: tuple[int, str] | None = None
        self._horizon_s: float | None = None
        self._latency: deque[float] = deque(maxlen=latency_window)
        self._refill_lock = threading.Lock()
        self._refill_cmd: int | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="s1-runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def latency_s(self) -> float | None:
        return max(self._latency) if self._latency else None

    def lead_s(self) -> float:
        latency = self.latency_s()
        return self.refill_s if latency is None else max(self.refill_s, latency + self.runtime.spec.dt)

    def floor(self, horizon: int) -> int:
        latency = self.latency_s()
        if latency is None:
            return horizon
        return math.ceil(2 * latency / self.runtime.spec.dt) + 1

    def deadline_s(self) -> float | None:
        return self._horizon_s if self.deadline else None

    def refill_now(self, cmd_id: int | None) -> bool:
        goal = self.runtime.target()
        with self._refill_lock:
            if cmd_id is None or goal is None or goal.cmd_id != cmd_id:
                self.stats.refill_ignored += 1
                return False
            self._refill_cmd = cmd_id
            return True

    def step(self) -> bool:
        goal = self.runtime.target()
        if goal is None:
            return False
        forced = self._refill_for(goal.cmd_id)
        until = self.runtime.queued_until(goal.cmd_id)
        if not forced and until is not None and until - self.now() > self.lead_s():
            return False
        with self._refill_lock:
            self._refill_cmd = None
        capture = self.latest()
        self.stats.requests += 1
        self.stats.forced += forced
        w0 = time.perf_counter()
        observation = capture.observation
        if goal.instruction:
            observation = dataclasses.replace(observation, task=goal.instruction)
        subtask = goal.operation.subtask or ""
        deadline = self.deadline_s()
        continuation = self._continuation(goal.cmd_id, capture.t)
        extra = (continuation,) if continuation is not None else ()
        self.stats.continuations += continuation is not None
        try:
            chunks = (self.policy.act_samples(observation, subtask, self.commit.samples, deadline, *extra)
                      if self.commit.samples > 1 else [self.policy.act(observation, subtask, deadline, *extra)])
        except TimeoutError as e:
            self.stats.timeouts += 1
            self.stats.last_error = f"TimeoutError: {e}"
            return True
        except Exception as e:
            self.stats.errors += 1
            self.stats.last_error = f"{type(e).__name__}: {e}"
            time.sleep(self.error_backoff_s)
            return True
        infer_ms = (time.perf_counter() - w0) * 1000
        horizon = min(len(c.actions) for c in chunks)
        self._horizon_s = horizon * self.runtime.spec.dt
        floor = self.floor(horizon)
        self.stats.floors.append(floor)
        self.stats.floor_over_chunk += floor > horizon
        commitment = self.commit.plan([c.actions for c in chunks], self.runtime.spec, floor, self.runtime.last_action())
        chunk = chunks[commitment.chosen]
        self.stats.committed.append(commitment.steps)
        self.stats.commit_reasons[commitment.reason] = self.stats.commit_reasons.get(commitment.reason, 0) + 1
        self.stats.infer_ms.append(infer_ms)
        timed = TimedChunk(goal.epoch, goal.cmd_id, capture.obs_id, capture.t, self.runtime.spec.dt,
                           commitment.actions, {**chunk.timings_ms, "s1_wall": round(infer_ms, 1)})
        if chunk.info:
            self.stats.info.append({"t": round(capture.t, 3), "cmd": goal.cmd_id, **chunk.info})
        if goal.operation.subtask is None:
            self._planner_moved_on(goal, capture, chunk.info.get("subtask"))
        now = self.now()
        if self.runtime.submit(timed, now):
            self.stats.accepted += 1
        self.stats.obs_age_s.append(now - capture.t)
        self._latency.append(now - capture.t)
        return True

    def _continuation(self, cmd_id: int, t: float) -> Continuation | None:
        latency = self.latency_s()
        if not self.policy.accepts_continuation or latency is None:
            return None
        queued = self.runtime.queued_from(cmd_id, t)
        if queued is None:
            return None
        return Continuation(queued, min(len(queued), math.ceil(latency / self.runtime.spec.dt)))

    def _planner_moved_on(self, goal: AcceptedGoal, capture: Capture, subtask: str | None) -> None:
        if not subtask:
            return
        previous, self._planned = self._planned, (goal.cmd_id, subtask)
        if previous is not None and previous == (goal.cmd_id, subtask):
            return
        if self.on_plan is not None:
            self.on_plan(PlannedSubtask(goal.epoch, goal.cmd_id, capture.t, subtask))
        if previous is None or previous[0] != goal.cmd_id:
            return
        self.stats.planner_steps += 1
        if self.on_step is not None:
            self.on_step(StepEvent(goal.epoch, goal.cmd_id, capture.t, "s1-planner", previous[1],
                                   {"next": subtask, "obs_id": capture.obs_id}))

    def _refill_for(self, cmd_id: int) -> bool:
        with self._refill_lock:
            if self._refill_cmd is not None and self._refill_cmd != cmd_id:
                self._refill_cmd = None
                self.stats.refill_ignored += 1
            return self._refill_cmd == cmd_id

    def _run(self) -> None:
        while self._running:
            if not self.step():
                time.sleep(self.poll_s)
