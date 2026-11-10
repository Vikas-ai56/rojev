"""State owner: the single writer of intent. Turns user instructions and decider proposals into commands for the
execution runtime, and keeps the record of what is active from the runtime's ACKs (docs/plan/v3-v1-spec.md §4).

One thread, one event queue; it never waits for a model. Events: `UserInstruction`, `DecisionProposal` and
`DecisionFailure` (from the S2 workers), `Deviation` (from the execution monitor), `StepEvent` (step-completion
evidence), and a periodic tick that drains ACKs, sends the guard its heartbeat, enforces the interaction deadline,
keeps the desired command leased and asks for a decision when an event or a timer calls for one. Out: commands to the
runtime, `Utterance`s to `say`, finished tasks to the episodic log.

Lanes: decisions go to the planning worker (start, step events, S1's planned subtasks, failed or expired commands, an
overdue step, the watchdog) or, when one is given,
the interaction worker (changes: `instruction`, `deviation`, `hold`). The interaction lane has priority: while its
request is pending, planning requests are deferred (the latest is sent once the interaction decision is applied) and
planning proposals are refused, so a routine next-step decision never overrides the response to what the user just
said. The pending request is released by its proposal, its `DecisionFailure`, or `interaction_timeout_s`. A change
nobody answered (failure, timeout, refused proposal) is never handed to the planning lane: the robot holds
(GUARDED_WAIT `UNANSWERED`, said to the user), so an unread "stop" can never move the robot. After a failure or a
refusal the interaction lane is asked again on the hold recheck; after a timeout the answer is still awaited: when it
comes it is applied to the situation it was decided on (the placeholder hold is not a new base; CONTINUE resumes what
was in force), and it is never asked again meanwhile. A newer interaction request ends the wait. Without an interaction
worker every trigger goes to the planning worker (one lane).

Planning is event-driven: the planner is asked at the start, on a step event, on a subtask S1's planner chose, when a
command ended without being revoked, when the running step has lasted `step_budget_s` without being moved past (trigger
`overdue`, once per budget), and by the watchdog after `watchdog_s` without any planning request. Nothing asks it on a
fixed beat: a planner that reasons for seconds is asked only when its answer can change.

Liveness: the lease is the owner's, not a decision. The owner renews the desired command (moving or holding) whenever
less than half a lease is left, except while it is under review (its answer is due) or while a planning request has
gone unanswered for more than `liveness_s` (the planner is gone): then the command lapses, the runtime expires it and
the robot holds. CONTINUE still renews, so an answer that keeps the step also keeps it alive.

Rules:
- A user instruction bumps `instruction_rev`, joins the dialogue, is acknowledged at once (`Utterance` `ACK`, before
  any model answers) and asks for a decision. It does not change the task text S1 is conditioned on (`instruction`):
  only an accepted decision does, by carrying new task text (`DecisionProposal.task`, e.g. a revision), so "wait" or
  "go on" never become the task. With `on_instruction_change="review"` (default) the moving command is put under
  review: it keeps its reversible motion, its irreversible switches freeze and its lease shrinks to the review window
  (the interaction deadline when an interaction lane exists, else `review_s`), so the robot never stops just because
  the user spoke and the command never expires before its answer is due; with `"revoke"` it is revoked at once.
- A proposal is accepted only if it answers a request this owner made, is from this epoch, for the current instruction
  revision, about the current desired command, newer than the last applied request, not expired, not a planning
  proposal while an interaction decision is pending (or an interaction proposal superseded by a newer one), and not a
  routine planning proposal while a hold stands; anything else is logged. Preconditions are checked when evidence
  exists to check them (step 6); until then a proposal that carries any is refused rather than trusted.
- CONTINUE (or the same operation again) renews the lease (`renewed` says whether there was a command to renew); if
  the decision adopted new task text it is reaffirmed instead: the same operation as a new command carrying the new
  task text, always `immediate` (no pause, S1 re-conditioned). COMPLETE marks the task done and holds;
  anything else becomes a new command with a new id. Between two moving commands the configured transition applies
  (default `immediate`); after a revoke, or into a hold, `pause_hold`. An operation with a template (an ASK question,
  a report) is said (`Utterance`) and joins the dialogue.
- Holds answer changes: ABORT, GUARDED_WAIT or ASK accepted in answer to a change becomes the standing `Hold`. It
  persists until a later change resolves it with a moving operation or COMPLETE: meanwhile no planning request is
  made and routine planning proposals are refused (so a planner never restarts a task the user stopped), the owner
  keeps the hold's command leased, and every `hold_recheck_s` (if set) asks the interaction lane again (trigger `hold`:
  the decider's wait budget). The hold remembers the moving step it interrupted, for a resume.
- Task memory (`TaskMemory`, the owner is its only writer; snapshots carry it): a checked step event about the running
  command records its step DONE at once, with the check's evidence, so the planner asked on that event already reads
  the step as done (the running subtask, or the event's step when S1 plans its own). When an accepted proposal moves
  past a step that was running (SWITCH_TO another subtask, or COMPLETE) and no checked event recorded it, the step is
  recorded DONE with the observation the decision used; its evidence is the step event about that command if one came,
  else the decider's judgement. An unchecked claim (e.g. S1's planner moving on) is never progress on its own. A proposal that
  switches to a step recorded DONE records it UNDONE first (the decider saw the scene contradict the memory). New task
  text becomes the memory's goal. ABORT, waits, asks and retries are not progress. A finished task (COMPLETE accepted,
  or the owner stopped first) is appended to the episodic log when one is given.
- A subtask S1's own planner chose for the active command (`PlannedSubtask`) is kept (`snapshot.planned`) and asks
  the planning lane at once (trigger `planned`); one about another command is stale.
- A deviation or step event about the active command is kept for the decider (`snapshot.deviation`,
  `snapshot.step_event`) and asks at once (triggers `deviation` → interaction, `step` → planning); one about any other
  command, or another epoch, is logged as stale. A newly activated command clears both.
- Snapshots carry the last `dialogue_turns` turns (user instructions, robot questions and reports), so a decider can
  read an instruction as the answer to its own question.
- What is active comes only from ACKs (newest `seq` wins); a command the runtime rejected, let expire or dropped (safety
  hold) triggers a new decision, and the desired command falls back to the one still running (if any), so the owner's
  record always matches what the robot does. A refused chunk (`chunk_rejected`) is logged only: the command stands.
"""
import dataclasses
import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from rojev.control import ExecutionRuntime, Guard
from rojev.protocol import (ACK, ROBOT, UNANSWERED, USER, AcceptedGoal, AckStatus, Capture, DecisionFailure,
                            DecisionProposal, Deviation, ExecutionAck, Hold, ObservationSnapshot, Operation, OpKind,
                            PlannedSubtask, Revoke, StepEvent, StepRecord, StepStatus, TaskMemory, Transition, Turn,
                            UserInstruction, Utterance)
from rojev.system2 import DecisionWorker

from .episodic import EpisodicLog

INSTRUCTION_CHANGE_MODES = ("review", "revoke")
PLANNING, INTERACTION = "planning", "interaction"
INTERACTION_TRIGGERS = ("instruction", "deviation", "hold")
STEP, PLANNED, START, OVERDUE, WATCHDOG = "step", "planned", "start", "overdue", "watchdog"
HOLD_KINDS = (OpKind.ABORT, OpKind.GUARDED_WAIT, OpKind.ASK)
ENDED = (AckStatus.REJECTED, AckStatus.EXPIRED, AckStatus.REVOKED, AckStatus.DROPPED)

Event = UserInstruction | DecisionProposal | DecisionFailure | Deviation | StepEvent | PlannedSubtask


@dataclass
class OwnerRecord:
    """Everything the owner decided or refused, in order, for logs and tests."""

    events: list[dict] = field(default_factory=list)

    def add(self, t: float, kind: str, **detail) -> None:
        self.events.append({"t": round(t, 3), "kind": kind, **detail})


@dataclass(frozen=True)
class _Request:
    lane: str
    snapshot: ObservationSnapshot


@dataclass(frozen=True)
class _Awaited:
    """An interaction request past its deadline whose answer may still come: the owner holds on `placeholder_cmd` (its
    own `UNANSWERED` hold) meanwhile."""

    request_id: int
    placeholder_cmd: int


class StateOwner:
    def __init__(self, runtime: ExecutionRuntime, guard: Guard, planner: DecisionWorker, latest: Callable[[], Capture],
                 now: Callable[[], float], instruction: str, lease_s: float = 3.0, watchdog_s: float = 10.0,
                 step_budget_s: float | None = None, liveness_s: float = 30.0,
                 transition: Transition = Transition.IMMEDIATE, period_s: float = 0.05,
                 on_instruction_change: str = "review", review_s: float = 1.0,
                 interaction: DecisionWorker | None = None, interaction_timeout_s: float = 1.0,
                 say: Callable[[Utterance], None] | None = None, dialogue_turns: int = 8,
                 hold_recheck_s: float | None = None, episodic: EpisodicLog | None = None):
        if on_instruction_change not in INSTRUCTION_CHANGE_MODES:
            raise ValueError(f"on_instruction_change {on_instruction_change!r} not in {INSTRUCTION_CHANGE_MODES}")
        self.runtime, self.guard, self.latest, self.now = runtime, guard, latest, now
        self.workers = {PLANNING: planner, INTERACTION: interaction}
        self.interaction_timeout_s = interaction_timeout_s
        self.on_instruction_change, self.review_s = on_instruction_change, review_s
        self.epoch = runtime.epoch
        self.lease_s, self.watchdog_s = lease_s, watchdog_s
        self.step_budget_s, self.liveness_s = step_budget_s, liveness_s
        self.episodic = episodic
        self.transition, self.period_s = transition, period_s
        self.say, self.hold_recheck_s = say, hold_recheck_s
        self.instruction, self.instruction_rev = instruction, 0
        self.desired: AcceptedGoal | None = None
        self.goals: dict[int, AcceptedGoal] = {}
        self.active_cmd: int | None = None
        self.active_since: float | None = None
        self.reviewing: int | None = None
        self.deviation: Deviation | None = None
        self.step_event: StepEvent | None = None
        self.planned: PlannedSubtask | None = None
        self.hold: Hold | None = None
        self.memory = TaskMemory(instruction)
        self.dialogue: deque[Turn] = deque(maxlen=dialogue_turns)
        self.complete = False
        self.record = OwnerRecord()
        self._cmd = 0
        self._request = 0
        self._requests: dict[int, _Request] = {}
        self._applied_request = -1
        self._interaction_pending: tuple[int, float] | None = None
        self._awaited: _Awaited | None = None
        self._deferred: str | None = None
        self._hold_checked_t: float | None = None
        self._last_ack_seq = 0
        self._last_request_t: float | None = None
        self._planning_unanswered_since: float | None = None
        self._overdue_asked_t: float | None = None
        self._liveness_lost = False
        self._ended = False
        self._events: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def interaction_pending(self) -> bool:
        return self._interaction_pending is not None

    def post(self, event: Event) -> None:
        self._events.put(event)

    @property
    def has_events(self) -> bool:
        """Posted events not handled yet."""
        return not self._events.empty()

    def step(self) -> None:
        """Handle every event posted so far, then tick: the owner driven from the caller's thread instead of `start`.

        Lockstep evaluation calls this before each control tick and ticks only when no lane is busy and, read after the
        lanes, `has_events` is false: a lane posts its answer before it stops being busy, so an answer is either still
        being computed or already queued here, never in between."""
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            self.handle(event)
        self.tick()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="state-owner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._end("stopped", self.now())

    def handle(self, event: Event) -> None:
        if isinstance(event, UserInstruction):
            self._on_instruction(event.text)
        elif isinstance(event, Deviation):
            self._on_deviation(event)
        elif isinstance(event, StepEvent):
            self._on_step(event)
        elif isinstance(event, PlannedSubtask):
            self._on_planned(event)
        elif isinstance(event, DecisionFailure):
            self._on_failure(event)
        else:
            self._on_proposal(event)

    def tick(self) -> None:
        now = self.now()
        self.guard.heartbeat(now)
        for ack in self.runtime.drain_acks():
            self._on_ack(ack)
        if self._interaction_pending is not None and now >= self._interaction_pending[1]:
            request_id = self._interaction_pending[0]
            self._interaction_pending = None
            self.record.add(now, "interaction timeout", request=request_id)
            self._unanswered(self._requests[request_id], "interaction timeout", awaiting=True)
        self._keep_leased(now)
        if self.hold is not None:
            self._recheck_hold(now)
            return
        if self.complete or self.interaction_pending or self.workers[PLANNING].busy:
            return
        trigger = self._due_trigger(now)
        if trigger is not None:
            self.request_decision(trigger)

    def _due_trigger(self, now: float) -> str | None:
        """The timer-driven planning triggers: `start` (never asked yet), `overdue` (the running step has lasted a step
        budget since it started or since it was last found overdue), `watchdog` (no planning request for `watchdog_s`)."""
        if self._last_request_t is None:
            return START
        goal = self.desired
        if self.step_budget_s is not None and goal is not None and goal.operation.moves_robot() \
                and self.active_cmd == goal.cmd_id and self.active_since is not None:
            since = max(self.active_since, self._overdue_asked_t or self.active_since)
            if now - since >= self.step_budget_s:
                self._overdue_asked_t = now
                return OVERDUE
        return WATCHDOG if now - self._last_request_t >= self.watchdog_s else None

    def _keep_leased(self, now: float) -> None:
        """Renew the desired command's lease while the system is alive (see module docstring: Liveness)."""
        goal = self.desired
        if goal is None or self.reviewing == goal.cmd_id or goal.expires_t - now >= self.lease_s / 2:
            return
        silent = self._planning_unanswered_since is not None and now - self._planning_unanswered_since > self.liveness_s
        if silent:
            if not self._liveness_lost:
                self._liveness_lost = True
                self.record.add(now, "liveness lapsed", cmd=goal.cmd_id,
                                unanswered_s=round(now - self._planning_unanswered_since, 3))
            return
        if self.runtime.renew(goal.cmd_id, now + self.lease_s, now):
            self.desired = self.goals[goal.cmd_id] = dataclasses.replace(goal, expires_t=now + self.lease_s)

    def request_decision(self, trigger: str, lane: str | None = None) -> None:
        """Ask a lane for a decision on the newest observation; the lane follows the trigger unless given."""
        lane = lane or (INTERACTION if trigger in INTERACTION_TRIGGERS and self.workers[INTERACTION] is not None
                        else PLANNING)
        now = self.now()
        if lane == PLANNING and self.interaction_pending:
            self._deferred = trigger
            self.record.add(now, "planning deferred", trigger=trigger, pending=self._interaction_pending[0])
            return
        if lane == PLANNING and self.hold is not None and trigger not in INTERACTION_TRIGGERS:
            self.record.add(now, "planning held", trigger=trigger, hold=self.hold.operation.kind.value)
            return
        self._request += 1
        self._last_request_t = now
        desired = self.desired
        snap = ObservationSnapshot(self.epoch, self._request, now, self.latest(), self.instruction_rev,
                                   self.instruction, desired.cmd_id if desired else None,
                                   desired.operation if desired else None,
                                   self.active_since if desired and self.active_cmd == desired.cmd_id else None,
                                   self.memory, trigger,
                                   review=desired is not None and self.reviewing == desired.cmd_id,
                                   deviation=self.deviation, step_event=self.step_event, planned=self.planned,
                                   hold=self.hold, dialogue=tuple(self.dialogue))
        self._requests[snap.request_id] = _Request(lane, snap)
        if lane == PLANNING and self._planning_unanswered_since is None:
            self._planning_unanswered_since = now
        if lane == INTERACTION:
            self._interaction_pending = (snap.request_id, now + self.interaction_timeout_s)
            self._awaited = None
        self.workers[lane].request(snap)

    def _run(self) -> None:
        while self._running:
            try:
                self.handle(self._events.get(timeout=self.period_s))
            except queue.Empty:
                pass
            self.tick()

    @property
    def review_window_s(self) -> float:
        """How long a moving command stays under review after the user speaks: until the answer is due, i.e. the
        interaction deadline when an interaction lane exists (one deadline, so the command never expires while its
        answer is still allowed to arrive), else `review_s`."""
        return self.interaction_timeout_s if self.workers[INTERACTION] is not None else self.review_s

    def _on_instruction(self, text: str) -> None:
        now = self.now()
        self.instruction_rev, self.complete = self.instruction_rev + 1, False
        self.dialogue.append(Turn(now, USER, self.instruction_rev, text=text))
        self.record.add(now, "instruction", rev=self.instruction_rev, text=text)
        self._utter(now, ACK, (), None)
        if self.desired is not None and self.desired.operation.moves_robot():
            cmd = self.desired.cmd_id
            until = now + self.review_window_s
            if self.on_instruction_change == "review" and self.runtime.review(cmd, until, now):
                self.reviewing = cmd
                self.desired = self.goals[cmd] = dataclasses.replace(
                    self.desired, expires_t=min(self.desired.expires_t, until))
                self.record.add(now, "review", cmd=cmd, until=round(until, 3))
            else:
                self.runtime.revoke(Revoke(self.epoch, cmd, "instruction changed"), now)
                self.record.add(now, "revoke", cmd=cmd, reason="instruction changed")
                self.desired = None
        self.request_decision("instruction")

    def _on_deviation(self, d: Deviation) -> None:
        now = self.now()
        if d.epoch != self.epoch or d.cmd_id is None or d.cmd_id != self.active_cmd:
            self.record.add(now, "stale deviation", cmd=d.cmd_id, active=self.active_cmd, score=d.score)
            return
        self.deviation = d
        self.record.add(now, "deviation", cmd=d.cmd_id, t_detect=d.t, score=d.score, threshold=d.threshold,
                        source=d.source)
        self.request_decision("deviation")

    def _on_step(self, e: StepEvent) -> None:
        now = self.now()
        if e.epoch != self.epoch or e.cmd_id is None or e.cmd_id != self.active_cmd:
            self.record.add(now, "stale step event", cmd=e.cmd_id, active=self.active_cmd, source=e.source)
            return
        self.step_event = e
        self.record.add(now, "step event", cmd=e.cmd_id, t_detect=e.t, source=e.source, step=e.step, detail=e.detail)
        running = self.goals.get(e.cmd_id)
        if e.checked and running is not None:
            step = running.operation.subtask or e.step
            if not self._recorded_done(step, running.cmd_id):
                self._remember(running, step, StepStatus.DONE, self.latest(), f"{e.source}: {e.detail}", True, now)
        self.request_decision(STEP)

    def _on_planned(self, e: PlannedSubtask) -> None:
        """S1's own planner chose a subtask for the active command: kept for the planning lane, which is asked at once
        (trigger `planned`); about any other command or epoch, logged as stale."""
        now = self.now()
        if e.epoch != self.epoch or e.cmd_id != self.active_cmd:
            self.record.add(now, "stale planned subtask", cmd=e.cmd_id, active=self.active_cmd, subtask=e.subtask)
            return
        self.planned = e
        self.record.add(now, "planned subtask", cmd=e.cmd_id, subtask=e.subtask)
        self.request_decision(PLANNED)

    def _on_failure(self, f: DecisionFailure) -> None:
        now = self.now()
        request = self._requests.pop(f.request_id, None)
        lane = request.lane if request is not None else None
        self._answered(lane)
        self.record.add(now, "decision failed", request=f.request_id, lane=lane, error=f.error)
        if self._awaited is not None and self._awaited.request_id == f.request_id:
            self._awaited = None
        if f.epoch != self.epoch or request is None or lane != INTERACTION:
            return
        if self._interaction_pending is not None and self._interaction_pending[0] == f.request_id:
            self._interaction_pending = None
            self._unanswered(request, f"decision failed: {f.error}")

    def _on_proposal(self, p: DecisionProposal) -> None:
        now = self.now()
        request = self._requests.get(p.request_id)
        lane = request.lane if request is not None else None
        self._answered(lane)
        pending = self._interaction_pending[0] if self._interaction_pending is not None else None
        base = self.desired.cmd_id if self.desired is not None else None
        late = self._awaited is not None and self._awaited.request_id == p.request_id
        if late and request is not None and base == self._awaited.placeholder_cmd:
            base = request.snapshot.active_cmd
        answers_change = request is not None and request.snapshot.trigger in INTERACTION_TRIGGERS
        problem = (f"epoch {p.epoch} != {self.epoch}" if p.epoch != self.epoch else
                   f"instruction rev {p.instruction_rev} != {self.instruction_rev}"
                   if p.instruction_rev != self.instruction_rev else
                   f"request {p.request_id} not newer than {self._applied_request}"
                   if p.request_id <= self._applied_request else
                   f"request {p.request_id} was not made by this owner" if request is None else
                   f"interaction decision {pending} pending" if lane == PLANNING and pending is not None else
                   f"interaction request {p.request_id} superseded by {pending}"
                   if lane == INTERACTION and pending is not None and p.request_id != pending else
                   f"based on cmd {p.base_active_cmd}, desired is {base}" if p.base_active_cmd != base else
                   f"held by {self.hold.operation.kind.value} until a change"
                   if self.hold is not None and not answers_change else
                   "expired" if p.expires_t <= now else
                   "preconditions cannot be checked yet" if p.preconditions else "")
        answers_pending = lane == INTERACTION and pending == p.request_id
        if answers_pending:
            self._interaction_pending = None
        if late:
            self._awaited = None
        if problem:
            op = p.operation
            self.record.add(now, "refused", request=p.request_id, lane=lane, op=op.kind.value, subtask=op.subtask,
                            why=problem)
            if answers_pending:
                self._unanswered(request, f"proposal refused: {problem}")
            return
        if late:
            p = self._as_decided(p, request)
            self.record.add(now, "late answer", request=p.request_id, op=p.operation.kind.value)
        self._apply(p, request, now)
        if answers_pending and self._deferred is not None:
            trigger, self._deferred = self._deferred, None
            self.request_decision(trigger)

    def _answered(self, lane: str | None) -> None:
        """The planning lane answered (a proposal, accepted or not, or a failure): it is alive."""
        if lane == PLANNING:
            self._planning_unanswered_since = None
            self._liveness_lost = False

    def _apply(self, p: DecisionProposal, request: _Request, now: float) -> None:
        self._applied_request = p.request_id
        for old in [r for r in self._requests if r <= p.request_id]:
            del self._requests[old]
        op = p.operation
        detail = {"request": p.request_id, "lane": request.lane, "trigger": request.snapshot.trigger,
                  "reason": p.evidence.get("reason", ""), "decider": p.evidence.get("decider")}
        desired = self.desired
        same = desired is not None and (op.kind == OpKind.CONTINUE or op == desired.operation)
        if p.task is not None and p.task != self.instruction:
            self.instruction = p.task
            self.memory = self.memory.with_goal(p.task)
            self.record.add(now, "task", text=p.task, request=p.request_id)
        if same and desired.instruction != self.instruction and desired.operation.moves_robot():
            self._issue(desired.operation, now, "reaffirm", detail, Transition.IMMEDIATE)
            return
        if same or op.kind == OpKind.CONTINUE:
            renewed = desired is not None and self.runtime.renew(desired.cmd_id, now + self.lease_s, now)
            if renewed:
                self.desired = self.goals[desired.cmd_id] = dataclasses.replace(desired, expires_t=now + self.lease_s)
                if self.reviewing == desired.cmd_id:
                    self.reviewing = None
            self.record.add(now, "continue", cmd=desired.cmd_id if desired else None, renewed=renewed, **detail)
            if same and op.template is not None and request.snapshot.trigger == "instruction":
                self.dialogue.append(Turn(now, ROBOT, self.instruction_rev, template=op.template, objects=op.objects))
                self._utter(now, op.template, op.objects, desired.cmd_id)
            return
        if op.kind == OpKind.COMPLETE:
            self.complete = True
        self._record_progress(p, request.snapshot, op, now)
        self._update_hold(op, request.snapshot.trigger in INTERACTION_TRIGGERS, now)
        self._issue(op, now, "command", detail)
        if op.kind == OpKind.COMPLETE:
            self._end("complete", now)

    def _as_decided(self, p: DecisionProposal, request: _Request) -> DecisionProposal:
        """A late answer is applied to the situation it was decided on, not to the owner's placeholder hold: CONTINUE
        ("keep what you are doing") becomes the operation in force when the change was decided (a plain hold if none
        was), so "keep going" resumes the motion the placeholder stopped."""
        if p.operation.kind != OpKind.CONTINUE:
            return p
        before = request.snapshot.active_operation
        return dataclasses.replace(p, operation=before if before is not None else Operation(OpKind.GUARDED_WAIT))

    def _unanswered(self, request: _Request, why: str, awaiting: bool = False) -> None:
        """No decision answered a change (the interaction lane failed, timed out, or its proposal was refused): the
        robot holds as if the answer were "wait" (`Hold`, interrupted step kept) and says `UNANSWERED`. After a timeout
        (`awaiting`) the answer may still come: it is applied when it does (`_as_decided`) and the hold is not
        rechecked meanwhile; otherwise the interaction lane is asked again on the hold recheck. A deferred routine
        planning request is dropped (planning is held)."""
        now = self.now()
        self._deferred = None
        op = Operation(OpKind.GUARDED_WAIT, template=UNANSWERED)
        self.record.add(now, "unanswered", request=request.snapshot.request_id, trigger=request.snapshot.trigger,
                        why=why)
        if self.desired is None or self.desired.operation != op:
            self._update_hold(op, True, now)
            self._issue(op, now, "command", {"request": request.snapshot.request_id, "lane": INTERACTION,
                                             "trigger": request.snapshot.trigger, "reason": f"unanswered: {why}",
                                             "decider": None})
        if awaiting:
            self._awaited = _Awaited(request.snapshot.request_id, self.desired.cmd_id)

    def _update_hold(self, op: Operation, answers_change: bool, now: float) -> None:
        """A hold that answers a change stands until a later change resolves it; keeps the step it first interrupted."""
        if answers_change and op.kind in HOLD_KINDS:
            moving = self.desired.operation if self.desired is not None and self.desired.operation.moves_robot() \
                else None
            interrupted = self.hold.interrupted if self.hold is not None else moving
            self.hold = Hold(op, now, interrupted)
            self._hold_checked_t = now
            self.record.add(now, "hold", op=op.kind.value, interrupted=interrupted.subtask if interrupted else None)
        elif self.hold is not None and (op.moves_robot() or op.kind == OpKind.COMPLETE):
            self.record.add(now, "hold released", op=op.kind.value, subtask=op.subtask)
            self.hold = None

    def _recheck_hold(self, now: float) -> None:
        """While a hold stands, ask the interaction lane again (trigger `hold`) every `hold_recheck_s` without a change,
        unless an answer is still awaited."""
        if self.hold_recheck_s is not None and not self.interaction_pending and self._awaited is None \
                and now - self._hold_checked_t >= self.hold_recheck_s:
            self._hold_checked_t = now
            self.request_decision("hold")

    def _record_progress(self, p: DecisionProposal, snap: ObservationSnapshot, op: Operation, now: float) -> None:
        """DONE when the proposal moves past a step that was running, UNDONE when it returns to a step recorded DONE
        (see module docstring)."""
        reason = f"decider {p.evidence.get('decider')}: {p.evidence.get('reason', '')}"
        if op.kind == OpKind.SWITCH_TO and op.subtask is not None \
                and self.memory.status(op.subtask) == StepStatus.DONE:
            self.memory = self.memory.with_record(StepRecord(op.subtask, StepStatus.UNDONE, snap.active_cmd,
                                                             self.instruction_rev, snap.capture, reason, False, now))
            self.record.add(now, "step undone", subtask=op.subtask, evidence=reason)
        step = self.desired.operation if self.desired is not None else None
        if step is None or step.subtask is None:
            return
        moves_on = op.kind == OpKind.COMPLETE or (op.kind == OpKind.SWITCH_TO and op.subtask != step.subtask)
        running = self.goals.get(self.active_cmd) if self.active_cmd is not None else None
        if not moves_on or running is None or running.operation != step \
                or self._recorded_done(step.subtask, running.cmd_id):
            return
        seen = snap.step_event
        about = seen is not None and seen.cmd_id == running.cmd_id
        evidence = f"{seen.source}: {seen.detail}" if about else reason
        self._remember(running, step.subtask, StepStatus.DONE, snap.capture, evidence, about and seen.checked, now)

    def _recorded_done(self, step: str, cmd_id: int) -> bool:
        latest = next((r for r in reversed(self.memory.records) if r.step == step), None)
        return latest is not None and latest.status == StepStatus.DONE and latest.cmd_id == cmd_id

    def _remember(self, running: AcceptedGoal, step: str, status: StepStatus, capture: Capture, evidence: str,
                  checked: bool, now: float) -> None:
        record = StepRecord(step, status, running.cmd_id, running.instruction_rev, capture, evidence, checked, now)
        self.memory = self.memory.with_record(record)
        self.record.add(now, "step " + status.value, cmd=record.cmd_id, subtask=step, obs_id=capture.obs_id,
                        evidence=evidence, checked=checked)

    def _end(self, outcome: str, now: float) -> None:
        """Append the finished task to the episodic log, once."""
        if self._ended or self.episodic is None or (outcome == "stopped" and self.complete):
            return
        self._ended = True
        self.episodic.append(self.memory, outcome, now, tuple(self.dialogue))

    def _issue(self, op: Operation, now: float, kind: str, detail: dict, transition: Transition | None = None) -> None:
        """A new command for `op` under the current instruction. Between moving commands the configured transition
        applies unless one is given; into a hold, or from a hold, `pause_hold`. An operation with a template is said."""
        moving_before = self.desired is not None and self.desired.operation.moves_robot()
        self._cmd += 1
        if transition is None:
            transition = self.transition if moving_before and op.moves_robot() else Transition.PAUSE_HOLD
        goal = AcceptedGoal(self.epoch, self._cmd, self.instruction_rev, op, transition, now + self.lease_s,
                            self.instruction)
        self.desired = self.goals[goal.cmd_id] = goal
        self.runtime.accept(goal, now)
        self.record.add(now, kind, cmd=goal.cmd_id, op=op.kind.value, subtask=op.subtask,
                        transition=transition.value, **detail)
        if op.template is not None:
            self.dialogue.append(Turn(now, ROBOT, self.instruction_rev, template=op.template, objects=op.objects))
            self._utter(now, op.template, op.objects, goal.cmd_id)

    def _utter(self, now: float, template: str, objects: tuple[str, ...], cmd_id: int | None) -> None:
        self.record.add(now, "say", template=template, objects=list(objects), cmd=cmd_id)
        if self.say is not None:
            self.say(Utterance(self.epoch, now, template, objects, cmd_id))

    def _on_ack(self, a: ExecutionAck) -> None:
        if a.epoch != self.epoch or a.seq <= self._last_ack_seq:
            return
        self._last_ack_seq = a.seq
        self.record.add(a.t, "ack", cmd=a.cmd_id, status=a.status.value, detail=a.detail)
        if a.status == AckStatus.ACTIVATED:
            previous = self.goals.get(self.active_cmd) if self.active_cmd is not None else None
            current = self.goals.get(a.cmd_id)
            if not (previous and current and previous.operation == current.operation):
                self.active_since = a.t
            if self.deviation is not None and self.deviation.cmd_id != a.cmd_id:
                self.deviation = None
            if self.step_event is not None and self.step_event.cmd_id != a.cmd_id:
                self.step_event = None
            if self.planned is not None and self.planned.cmd_id != a.cmd_id:
                self.planned = None
            self.active_cmd = a.cmd_id
            if self.reviewing is not None and self.reviewing != a.cmd_id:
                self.reviewing = None
        elif a.status in ENDED:
            if self.active_cmd == a.cmd_id:
                self.active_cmd = None
            if self.desired is not None and a.cmd_id == self.desired.cmd_id and a.status != AckStatus.REVOKED:
                running = self.goals.get(self.active_cmd) if self.active_cmd is not None else None
                self.desired = running
                self.request_decision(f"cmd {a.cmd_id} {a.status.value}")
