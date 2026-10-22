"""Jev v1 interaction decider (spec §9): answers a change with one calibrated act, inside budgets.

Per snapshot, in order:
1. `hold` trigger (the wait budget ran out, decision D7): a stop keeps holding; otherwise the robot reports once that
   it is waiting (GUARDED_WAIT with the `waiting_for_user` template) and then keeps holding. No model call. Except a
   hold the owner took because no decision answered the change (`UNANSWERED`): that change is decided now, as below.
2. Minimum dwell: a deviation about a command that started less than `min_dwell_s` ago is answered CONTINUE (the new
   command gets time to act before it is judged again; no flip-flopping). No model call.
3. The legal acts (`acts.candidates`) are scored by the chooser and calibrated (`Calibration`).
4. Abstain: if the calibrated margin between the two best acts is below `calibration.margin`, ask (if the question
   budget allows) else wait. A chosen ask that the budget forbids also becomes a wait.
5. The act becomes an operation (`acts.operation`); REVISE goes through the `Reviser`, which also gives the new task
   text (the only act that changes what S1 is conditioned on).
The decider holds no state of its own: holds, the dialogue and progress come from the snapshot (the owner's record).
"""
from rojev.protocol import ROBOT, UNANSWERED, ObservationSnapshot, Operation, OpKind

from ..decider import Verdict
from .acts import ASK, REVISE, WAIT, WAITING, Reviser, S1Replans, candidates, may_ask, operation
from .calibration import Calibration
from .chooser import Chooser


class JevInteraction:
    def __init__(self, chooser: Chooser, calibration: Calibration = Calibration(), reviser: Reviser = S1Replans(),
                 max_asks: int = 2, min_dwell_s: float = 1.0):
        self.chooser, self.calibration, self.reviser = chooser, calibration, reviser
        self.max_asks, self.min_dwell_s = max_asks, min_dwell_s
        self.name = f"jev-{chooser.name}"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        unanswered = s.hold is not None and s.hold.operation.template == UNANSWERED
        if s.trigger == "hold" and not unanswered:
            return self._wait_budget_spent(s)
        if s.trigger == "deviation" and s.active_since is not None and s.now - s.active_since < self.min_dwell_s:
            return Verdict(Operation(OpKind.CONTINUE), f"deviation within {self.min_dwell_s:g}s of the step's start")
        options = candidates(s, self.max_asks)
        raw = self.chooser.choose(s, options)
        probs = self.calibration.apply({o.label: raw.get(o.label, 0.0) for o in options})
        ranked = sorted(probs, key=probs.get, reverse=True)
        best = ranked[0]
        margin = probs[best] - (probs[ranked[1]] if len(ranked) > 1 else 0.0)
        evidence = {"options": {o.label: o.text for o in options}, "raw": raw, "margin": round(margin, 4),
                    "calibration": {"temperature": self.calibration.temperature, "fitted": self.calibration.fitted}}
        act, reason = best, f"{best} (p={probs[best]:.2f}, margin {margin:.2f})"
        if margin < self.calibration.margin:
            act = ASK if may_ask(s, self.max_asks) else WAIT
            reason = f"abstain: margin {margin:.2f} < {self.calibration.margin:g} between {best} and {ranked[1]}"
        if act == ASK and not may_ask(s, self.max_asks):
            act, reason = WAIT, reason + "; question budget spent"
        if act == WAIT and s.active_operation is not None and not s.active_operation.moves_robot():
            return Verdict(Operation(OpKind.CONTINUE), reason + "; already holding", probs, evidence=evidence)
        if act == REVISE:
            op, task = self.reviser.revise(s)
            return Verdict(op, reason, probs, evidence=evidence, task=task)
        return Verdict(operation(act, s), reason, probs, evidence=evidence)

    def _wait_budget_spent(self, s: ObservationSnapshot) -> Verdict:
        if s.hold is not None and s.hold.operation.kind == OpKind.ABORT:
            return Verdict(Operation(OpKind.CONTINUE), "stopped: hold until the user speaks")
        last = s.dialogue[-1] if s.dialogue else None
        if last is not None and last.speaker == ROBOT and last.template == WAITING:
            return Verdict(Operation(OpKind.CONTINUE), "already told the user it is waiting")
        return Verdict(Operation(OpKind.GUARDED_WAIT, template=WAITING), "wait budget spent: report and keep holding")
