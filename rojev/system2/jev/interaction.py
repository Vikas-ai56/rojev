"""Jev v1 interaction decider (spec §9): answers a change with one calibrated act, inside budgets.

Per snapshot, in order:
1. `hold` trigger (the wait budget ran out, decision D7): a stop keeps holding; otherwise the robot reports once that
   it is waiting (GUARDED_WAIT with the `waiting_for_user` template) and then keeps holding. No model call. Except a
   hold the owner took because no decision answered the change (`UNANSWERED`): that change is decided now, as below.
2. Minimum dwell: a deviation about a command that started less than `min_dwell_s` ago is answered CONTINUE (the new
   command gets time to act before it is judged again; no flip-flopping). No model call.
3. Question one, the act: the legal acts (`acts.candidates`) are scored by the chooser and calibrated (`calibration`).
4. Abstain on effects, not labels: the calibrated probabilities are summed per effect (`acts.effect`: move, hold,
   switch); if the margin between the two best effects is below `calibration.margin`, ask (if the question budget
   allows) else wait. Otherwise the act is the most probable one with the winning effect (live 2026-10-24: "stop"
   split 0.49 abort / 0.41 wait and asked, though both hold; "ruko" went to revise 0.45 against hold 0.38 + move 0.16).
   A chosen ask that the budget forbids also becomes a wait.
5. Question two, only for REVISE: the `Reviser` names the new task (default `CanonicalTask`: one of S1's trained task
   sentences, scored and calibrated on its own); it finding none is handled as an abstention (ask, else wait).
6. A wait while already holding keeps the hold; if the user just spoke, the robot says it is waiting (`WAITING`) so no
   instruction goes unanswered.
7. The act becomes an operation (`acts.operation`); only a revision changes the task text S1 is conditioned on.
The decider holds no state of its own: holds, the dialogue and progress come from the snapshot (the owner's record).
Evidence carries each question's options and scores (`act`, and `task` when asked), so both can be recorded as
calibration cases.
"""
from rojev.protocol import ROBOT, UNANSWERED, ObservationSnapshot, Operation, OpKind

from ..decider import Verdict
from .acts import ASK, REVISE, WAIT, WAITING, Reviser, candidates, effect, may_ask, operation, situation_of
from .calibration import Calibration
from .chooser import ACT_QUESTION, Chooser


class JevInteraction:
    def __init__(self, chooser: Chooser, reviser: Reviser, calibration: Calibration = Calibration(),
                 max_asks: int = 2, min_dwell_s: float = 1.0):
        self.chooser, self.reviser, self.calibration = chooser, reviser, calibration
        self.max_asks, self.min_dwell_s = max_asks, min_dwell_s
        self.name = f"jev-{chooser.name}"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        unanswered = s.hold is not None and s.hold.operation.template == UNANSWERED
        if s.trigger == "hold" and not unanswered:
            return self._wait_budget_spent(s)
        if s.trigger == "deviation" and s.active_since is not None and s.now - s.active_since < self.min_dwell_s:
            return Verdict(Operation(OpKind.CONTINUE), f"deviation within {self.min_dwell_s:g}s of the step's start")
        options = candidates(s, self.max_asks)
        raw = self.chooser.choose(s, options, ACT_QUESTION)
        probs = self.calibration.apply({o.label: raw.get(o.label, 0.0) for o in options})
        situation = situation_of(s)
        effects = {o.label: effect(o.label, situation) for o in options}
        by_effect: dict[str, float] = {}
        for label, p in probs.items():
            by_effect[effects[label]] = by_effect.get(effects[label], 0.0) + p
        ranked = sorted(by_effect, key=by_effect.get, reverse=True)
        margin = by_effect[ranked[0]] - (by_effect[ranked[1]] if len(ranked) > 1 else 0.0)
        best = max((label for label in probs if effects[label] == ranked[0]), key=probs.get)
        evidence = {"options": {o.label: o.text for o in options}, "raw": raw, "margin": round(margin, 4),
                    "effects": effects, "effect_scores": {k: round(v, 4) for k, v in by_effect.items()},
                    "question": ACT_QUESTION,
                    "calibration": {"temperature": self.calibration.temperature, "fitted": self.calibration.fitted}}
        act, reason = best, f"{best} (p={probs[best]:.2f}; {ranked[0]} {by_effect[ranked[0]]:.2f}, margin {margin:.2f})"
        if margin < self.calibration.margin:
            act = ASK if may_ask(s, self.max_asks) else WAIT
            reason = (f"abstain: effect margin {margin:.2f} < {self.calibration.margin:g} between {ranked[0]} "
                      f"and {ranked[1]}")
        if act == REVISE:
            revision = self.reviser.revise(s)
            evidence["task"] = revision.evidence
            if revision.operation is not None:
                return Verdict(revision.operation, f"{reason}; {revision.reason}", probs, evidence=evidence,
                               task=revision.task)
            act = ASK if may_ask(s, self.max_asks) else WAIT
            reason = f"{reason}; {revision.reason}"
        if act == ASK and not may_ask(s, self.max_asks):
            act, reason = WAIT, reason + "; question budget spent"
        if act == WAIT and s.active_operation is not None and not s.active_operation.moves_robot():
            if s.trigger == "instruction":
                return Verdict(Operation(OpKind.GUARDED_WAIT, template=WAITING), reason + "; holding, told the user",
                               probs, evidence=evidence)
            return Verdict(Operation(OpKind.CONTINUE), reason + "; already holding", probs, evidence=evidence)
        return Verdict(operation(act, s), reason, probs, evidence=evidence)

    def _wait_budget_spent(self, s: ObservationSnapshot) -> Verdict:
        if s.hold is not None and s.hold.operation.kind == OpKind.ABORT:
            return Verdict(Operation(OpKind.CONTINUE), "stopped: hold until the user speaks")
        last = s.dialogue[-1] if s.dialogue else None
        if last is not None and last.speaker == ROBOT and last.template == WAITING:
            return Verdict(Operation(OpKind.CONTINUE), "already told the user it is waiting")
        return Verdict(Operation(OpKind.GUARDED_WAIT, template=WAITING), "wait budget spent: report and keep holding")
