"""State owner: the single writer of intent. Turns user instructions and decider proposals into commands for the
execution runtime, and keeps the record of what is active from the runtime's ACKs (docs/plan/v3-v1-spec.md §4).

One thread, one event queue; it never waits for a model. Events: `UserInstruction`, `DecisionProposal` and
`DecisionFailure` (from the S2 workers), `Deviation` (from the execution monitor), `StepEvent` (step-completion
evidence), and a periodic tick that drains ACKs, sends the guard its heartbeat, enforces the interaction deadline,
keeps a hold alive and asks for a decision when one is due. Out: commands to the runtime, `Utterance`s to `say`.

Lanes: decisions go to the planning worker (heartbeat, step events, failed or expired commands) or, when one is given,
the interaction worker (changes: `instruction`, `deviation`, `hold`). The interaction lane has priority: while its
request is pending, planning requests are deferred (the latest is sent once the interaction decision is applied) and
planning proposals are refused, so a routine next-step decision never overrides the response to what the user just
said. The pending request is released by its proposal, its `DecisionFailure`, or `interaction_timeout_s`; if it fails,
times out or its proposal is refused, the same trigger goes to the planning lane, so a change is always answered.
Without an interaction worker every trigger goes to the planning worker (one lane).

Rules:
- A user instruction bumps `instruction_rev`, joins the dialogue, is acknowledged at once (`Utterance` `ACK`, before
  any model answers) and asks for a decision. With `on_instruction_change="review"` (default) the moving command is
  put under review: it keeps its reversible motion, its irreversible switches freeze and its lease shrinks to
  `review_s`, so the robot never stops just because the user spoke; with `"revoke"` it is revoked at once.
- A proposal is accepted only if it answers a request this owner made, is from this epoch, for the current instruction
  revision, about the current desired command, newer than the last applied request, not expired, not a planning
  proposal while an interaction decision is pending (or an interaction proposal superseded by a newer one), and not a
  routine planning proposal while a hold stands; anything else is logged. Preconditions are checked when evidence
  exists to check them (step 6); until then a proposal that carries any is refused rather than trusted.
- CONTINUE (or the same operation again) renews the lease (`renewed` says whether there was a command to renew); if
  the command predates the current instruction it is reaffirmed instead: the same operation as a new command carrying
  the new instruction, always `immediate` (no pause, S1 re-conditioned). COMPLETE marks the task done and holds;
  anything else becomes a new command with a new id. Between two moving commands the configured transition applies
  (default `immediate`); after a revoke, or into a hold, `pause_hold`. An operation with a template (an ASK question,
  a report) is said (`Utterance`) and joins the dialogue.
- Holds answer changes: ABORT, GUARDED_WAIT or ASK accepted in answer to a change becomes the standing `Hold`. It
  persists until a later change resolves it with a moving operation or COMPLETE: meanwhile no heartbeat is sent and
  routine planning proposals are refused (so a planner never restarts a task the user stopped), the owner keeps the
  hold's command leased, and every `hold_recheck_s` (if set) asks the interaction lane again (trigger `hold`: the
  decider's wait budget). The hold remembers the moving step it interrupted, for a resume.
- Progress keyframes: when an accepted proposal moves past a step that was running (SWITCH_TO another subtask, or
  COMPLETE), the step is recorded as a `Keyframe` with the observation the decision used; its evidence is the step
  event about that command if one came, else the decider's judgement. Snapshots carry all keyframes, so after a repair
  or a revision no decider redoes a done step. ABORT, waits, asks and retries are not progress.
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
from rojev.protocol import (ACK, ROBOT, USER, AcceptedGoal, AckStatus, Capture, DecisionFailure, DecisionProposal,
                            Deviation, ExecutionAck, Hold, Keyframe, ObservationSnapshot, Operation, OpKind, Revoke,
                            StepEvent, Transition, Turn, UserInstruction, Utterance)
from rojev.system2 import DecisionWorker

INSTRUCTION_CHANGE_MODES = ("review", "revoke")
PLANNING, INTERACTION = "planning", "interaction"
INTERACTION_TRIGGERS = ("instruction", "deviation", "hold")
STEP = "step"
HOLD_KINDS = (OpKind.ABORT, OpKind.GUARDED_WAIT, OpKind.ASK)
ENDED = (AckStatus.REJECTED, AckStatus.EXPIRED, AckStatus.REVOKED, AckStatus.DROPPED)

Event = UserInstruction | DecisionProposal | DecisionFailure | Deviation | StepEvent


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


class StateOwner:
    def __init__(self, runtime: ExecutionRuntime, guard: Guard, planner: DecisionWorker, latest: Callable[[], Capture],
                 now: Callable[[], float], instruction: str, lease_s: float = 3.0, decide_every_s: float = 1.0,
                 transition: Transition = Transition.IMMEDIATE, period_s: float = 0.05,
                 on_instruction_change: str = "review", review_s: float = 1.0,
                 interaction: DecisionWorker | None = None, interaction_timeout_s: float = 1.0,
                 say: Callable[[Utterance], None] | None = None, dialogue_turns: int = 8,
                 hold_recheck_s: float | None = None):
        if on_instruction_change not in INSTRUCTION_CHANGE_MODES:
            raise ValueError(f"on_instruction_change {on_instruction_change!r} not in {INSTRUCTION_CHANGE_MODES}")
        self.runtime, self.guard, self.latest, self.now = runtime, guard, latest, now
        self.workers = {PLANNING: planner, INTERACTION: interaction}
        self.interaction_timeout_s = interaction_timeout_s
        self.on_instruction_change, self.review_s = on_instruction_change, review_s
        self.epoch = runtime.epoch
        self.lease_s, self.decide_every_s = lease_s, decide_every_s
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
        self.hold: Hold | None = None
        self.keyframes: list[Keyframe] = []
        self.dialogue: deque[Turn] = deque(maxlen=dialogue_turns)
        self.complete = False
        self.record = OwnerRecord()
        self._cmd = 0
        self._request = 0
        self._requests: dict[int, _Request] = {}
        self._applied_request = -1
        self._interaction_pending: tuple[int, float] | None = None
        self._deferred: str | None = None
        self._hold_checked_t: float | None = None
        self._last_ack_seq = 0
        self._last_request_t: float | None = None
        self._events: queue.Queue = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def interaction_pending(self) -> bool:
        return self._interaction_pending is not None

    def post(self, event: Event) -> None:
        self._events.put(event)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, name="state-owner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def handle(self, event: Event) -> None:
        if isinstance(event, UserInstruction):
            self._on_instruction(event.text)
        elif isinstance(event, Deviation):
            self._on_deviation(event)
        elif isinstance(event, StepEvent):
            self._on_step(event)
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
            self._escalate(self._requests[request_id])
        if self.hold is not None:
            self._keep_hold(now)
            return
        due = self._last_request_t is None or now - self._last_request_t >= self.decide_every_s
        if not self.complete and due and not self.interaction_pending and not self.workers[PLANNING].busy:
            self.request_decision("heartbeat")

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
                                   tuple(self.keyframes), trigger,
                                   review=desired is not None and self.reviewing == desired.cmd_id,
                                   deviation=self.deviation, step_event=self.step_event, hold=self.hold,
                                   dialogue=tuple(self.dialogue))
        self._requests[snap.request_id] = _Request(lane, snap)
        if lane == INTERACTION:
            self._interaction_pending = (snap.request_id, now + self.interaction_timeout_s)
        self.workers[lane].request(snap)

    def _run(self) -> None:
        while self._running:
            try:
                self.handle(self._events.get(timeout=self.period_s))
            except queue.Empty:
                pass
            self.tick()

    def _on_instruction(self, text: str) -> None:
        now = self.now()
        self.instruction, self.instruction_rev, self.complete = text, self.instruction_rev + 1, False
        self.dialogue.append(Turn(now, USER, self.instruction_rev, text=text))
        self.record.add(now, "instruction", rev=self.instruction_rev, text=text)
        self._utter(now, ACK, (), None)
        if self.desired is not None and self.desired.operation.moves_robot():
            cmd = self.desired.cmd_id
            if self.on_instruction_change == "review" and self.runtime.review(cmd, now + self.review_s, now):
                self.reviewing = cmd
                self.desired = self.goals[cmd] = dataclasses.replace(
                    self.desired, expires_t=min(self.desired.expires_t, now + self.review_s))
                self.record.add(now, "review", cmd=cmd, until=round(now + self.review_s, 3))
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
        self.record.add(now, "step event", cmd=e.cmd_id, t_detect=e.t, source=e.source, detail=e.detail)
        self.request_decision(STEP)

    def _on_failure(self, f: DecisionFailure) -> None:
        now = self.now()
        request = self._requests.pop(f.request_id, None)
        lane = request.lane if request is not None else None
        self.record.add(now, "decision failed", request=f.request_id, lane=lane, error=f.error)
        if f.epoch != self.epoch or request is None or lane != INTERACTION:
            return
        if self._interaction_pending is not None and self._interaction_pending[0] == f.request_id:
            self._interaction_pending = None
            self._escalate(request)

    def _on_proposal(self, p: DecisionProposal) -> None:
        now = self.now()
        request = self._requests.get(p.request_id)
        lane = request.lane if request is not None else None
        pending = self._interaction_pending[0] if self._interaction_pending is not None else None
        base = self.desired.cmd_id if self.desired is not None else None
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
        if problem:
            op = p.operation
            self.record.add(now, "refused", request=p.request_id, lane=lane, op=op.kind.value, subtask=op.subtask,
                            why=problem)
            if answers_pending:
                self._escalate(request)
            return
        self._apply(p, request, now)
        if answers_pending and self._deferred is not None:
            trigger, self._deferred = self._deferred, None
            self.request_decision(trigger)

    def _apply(self, p: DecisionProposal, request: _Request, now: float) -> None:
        self._applied_request = p.request_id
        for old in [r for r in self._requests if r <= p.request_id]:
            del self._requests[old]
        op = p.operation
        detail = {"request": p.request_id, "lane": request.lane, "trigger": request.snapshot.trigger,
                  "reason": p.evidence.get("reason", ""), "decider": p.evidence.get("decider")}
        desired = self.desired
        same = desired is not None and (op.kind == OpKind.CONTINUE or op == desired.operation)
        if same and desired.instruction_rev != self.instruction_rev and desired.operation.moves_robot():
            self._issue(desired.operation, now, "reaffirm", detail, Transition.IMMEDIATE)
            return
        if same or op.kind == OpKind.CONTINUE:
            renewed = desired is not None and self.runtime.renew(desired.cmd_id, now + self.lease_s, now)
            if renewed:
                self.desired = self.goals[desired.cmd_id] = dataclasses.replace(desired, expires_t=now + self.lease_s)
                if self.reviewing == desired.cmd_id:
                    self.reviewing = None
            self.record.add(now, "continue", cmd=desired.cmd_id if desired else None, renewed=renewed, **detail)
            return
        if op.kind == OpKind.COMPLETE:
            self.complete = True
        self._record_progress(p, request.snapshot, op, now)
        self._update_hold(op, request.snapshot.trigger in INTERACTION_TRIGGERS, now)
        self._issue(op, now, "command", detail)

    def _escalate(self, request: _Request) -> None:
        """The interaction lane did not answer a change (failure, timeout, refusal): the planning lane answers it, and
        replaces any planning request deferred meanwhile."""
        self._deferred = None
        self.request_decision(request.snapshot.trigger, lane=PLANNING)

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

    def _keep_hold(self, now: float) -> None:
        """While a hold stands the owner keeps its command leased (holding needs no decision), and asks the
        interaction lane again (trigger `hold`) every `hold_recheck_s` without a change."""
        goal = self.desired
        if goal is not None and goal.operation == self.hold.operation and goal.expires_t - now < self.lease_s / 2 \
                and self.runtime.renew(goal.cmd_id, now + self.lease_s, now):
            self.desired = self.goals[goal.cmd_id] = dataclasses.replace(goal, expires_t=now + self.lease_s)
        if self.hold_recheck_s is not None and not self.interaction_pending \
                and now - self._hold_checked_t >= self.hold_recheck_s:
            self._hold_checked_t = now
            self.request_decision("hold")

    def _record_progress(self, p: DecisionProposal, snap: ObservationSnapshot, op: Operation, now: float) -> None:
        """A keyframe when the proposal moves past a step that was running (see module docstring)."""
        step = self.desired.operation if self.desired is not None else None
        if step is None or step.subtask is None:
            return
        moves_on = op.kind == OpKind.COMPLETE or (op.kind == OpKind.SWITCH_TO and op.subtask != step.subtask)
        running = self.goals.get(self.active_cmd) if self.active_cmd is not None else None
        if not moves_on or running is None or running.operation != step:
            return
        seen = snap.step_event
        evidence = (f"{seen.source}: {seen.detail}" if seen is not None and seen.cmd_id == running.cmd_id else
                    f"decider {p.evidence.get('decider')}: {p.evidence.get('reason', '')}")
        frame = Keyframe(running.cmd_id, step.subtask, running.instruction_rev, snap.capture, evidence)
        self.keyframes.append(frame)
        self.record.add(now, "keyframe", cmd=frame.cmd_id, subtask=frame.subtask, obs_id=frame.capture.obs_id,
                        evidence=frame.evidence)

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
