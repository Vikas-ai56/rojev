"""Decider stand-ins for the live scenarios (evaluation harness, not the system under test).

SPOOFS (declared):
  RuleBasedInteraction  keyword rules in place of Jev's calibrated interaction acts (step 6): stop → ABORT,
                        wait → GUARDED_WAIT, go on → resume, a vague reference → ASK, a new destination → REVISE
                        (SWITCH_TO the rewritten step), a deviation → RETRY, anything else → CONTINUE
  KeywordChooser        the same keyword rules as act probabilities, in place of OneJev's scores inside the real
                        `JevInteraction` (candidates, calibration, abstention and budgets are the system under test)
  RewriteReviser        REVISE = the interrupted or running place step rewritten to the destination the instruction
                        names, in place of π0.5²'s own planner re-planning (the scripted S1 needs a step text)
  RevisableSequence     the authored step list rewritten to the destination the current instruction names, advanced
                        on a time budget or on simulator truth; in place of a planner that re-plans from the
                        instruction (π0.5²'s own planner in v1)
  TruthVerifier         simulator truth (tasks.py completion checks) in place of MJev's end-state checks, so the
                        step watcher and step-completion events run without a GPU; P(done) is 1 or 0
  SlowDecider           adds latency to a decider (S2 delay fault)
  FailingDecider        raises on chosen triggers (S2 failure fault)
"""
import re
import time
from collections.abc import Callable

from rojev.eval import next_after_latest_done
from rojev.protocol import ObservationSnapshot, Operation, OpKind
from rojev.system2 import Verdict, jev

DESTINATIONS = (("middle shelf", "on the middle shelf"), ("top shelf", "on the top shelf"),
                ("bottom shelf", "on the bottom shelf"), ("microwave", "in the microwave"), ("plate", "on the plate"))
STOP = re.compile(r"\b(stop|abort|cancel)\b")
WAIT = re.compile(r"\b(wait|hold on|pause)\b")
RESUME = re.compile(r"\b(go on|carry on|continue|resume|go ahead)\b")
VAGUE = re.compile(r"\b(there|that one|over there|this one)\b")
PLACE = re.compile(r"^(place the .+?) (?:in|on|back in|back on) the .+$")


def destination(text: str) -> str | None:
    """The destination phrase an instruction names, if any."""
    lowered = text.lower()
    return next((phrase for key, phrase in DESTINATIONS if key in lowered), None)


def rewrite(subtask: str | None, phrase: str | None) -> str | None:
    """A place step sent to `phrase` instead; other steps unchanged."""
    match = PLACE.match(subtask or "")
    return f"{match.group(1)} {phrase}" if match and phrase else subtask


def _act(s: ObservationSnapshot) -> str:
    """The act the keyword rules pick for a change."""
    if s.trigger == "deviation":
        return jev.RETRY
    text = s.instruction.lower()
    if STOP.search(text):
        return jev.ABORT
    if WAIT.search(text):
        return jev.WAIT
    if VAGUE.search(text) and destination(text) is None:
        return jev.ASK
    if destination(text) is not None:
        return jev.REVISE
    if RESUME.search(text) and s.hold is not None and s.hold.interrupted is not None:
        return jev.RESUME
    return jev.CONTINUE


class RuleBasedInteraction:
    """Interaction-lane stand-in: the keyword act as an operation. Resumes the step a hold interrupted (the owner's
    `snapshot.hold`)."""

    name = "rules"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        act = _act(s)
        running = s.active_operation if s.active_operation is not None and s.active_operation.moves_robot() else None
        if act == jev.RETRY:
            if running is not None:
                return Verdict(Operation(OpKind.RETRY, subtask=running.subtask), "deviation: retry the step")
            return Verdict(Operation(OpKind.CONTINUE), "deviation while holding")
        if act == jev.ABORT:
            return Verdict(Operation(OpKind.ABORT), "user said stop")
        if act == jev.WAIT:
            return Verdict(Operation(OpKind.GUARDED_WAIT), "user asked to wait")
        if act == jev.ASK:
            return Verdict(Operation(OpKind.ASK, template=jev.CLARIFY), "vague reference: ask")
        if act == jev.RESUME:
            return Verdict(s.hold.interrupted, "resume the interrupted step")
        if act == jev.REVISE:
            revised = RewriteReviser().revise(s)
            if revised.kind == OpKind.CONTINUE:
                return Verdict(revised, "destination applies to a later step")
            return Verdict(revised, f"revise: {destination(s.instruction)}")
        return Verdict(Operation(OpKind.CONTINUE), "nothing to change")


class KeywordChooser:
    """`Chooser` stand-in for OneJev: the keyword act gets `confidence`, the other legal acts share the rest."""

    name = "keywords"

    def __init__(self, confidence: float = 0.8):
        self.confidence = confidence

    def choose(self, s: ObservationSnapshot, options: list[jev.Option]) -> dict[str, float]:
        labels = [o.label for o in options]
        act = _act(s)
        best = act if act in labels else jev.CONTINUE
        rest = (1.0 - self.confidence) / max(len(labels) - 1, 1)
        return {label: self.confidence if label == best else rest for label in labels}


class RewriteReviser:
    """`Reviser` stand-in: the step a hold interrupted (else the running one) rewritten to the destination the
    instruction names; CONTINUE when the destination concerns a later step."""

    def revise(self, s: ObservationSnapshot) -> Operation:
        step = s.hold.interrupted if s.hold is not None and s.hold.interrupted is not None else s.active_operation
        phrase = destination(s.instruction)
        if step is None or not step.subtask:
            return Operation(OpKind.CONTINUE)
        revised = rewrite(step.subtask, phrase)
        held = s.active_operation is not None and not s.active_operation.moves_robot()
        if revised == step.subtask and not held:
            return Operation(OpKind.CONTINUE)
        return Operation(OpKind.SWITCH_TO, subtask=revised)


class RevisableSequence:
    """Planning-lane stand-in. Steps advance after `budget_s` of activity, or when `is_done(i)` (simulator truth) holds
    when given; place steps follow the destination the current instruction names."""

    def __init__(self, sequence: list[str], budget_s: float | None = None,
                 is_done: Callable[[int], bool] | None = None):
        if (budget_s is None) == (is_done is None):
            raise ValueError("give exactly one of budget_s (time budget) or is_done (simulator truth)")
        self.sequence, self.budget_s, self.is_done, self.index = list(sequence), budget_s, is_done, 0
        self.name = "revisable-truth" if is_done is not None else "revisable-budget"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        phrase = destination(s.instruction) if s.instruction_rev > 0 else None
        steps = [rewrite(step, phrase) for step in self.sequence]
        current = s.active_operation.subtask if s.active_operation is not None else None
        if s.active_operation is not None and not s.active_operation.moves_robot() \
                and s.active_operation.kind != OpKind.COMPLETE:
            return Verdict(Operation(OpKind.CONTINUE), f"{s.active_operation.kind.value}: planning waits")
        if self.is_done is not None:
            self.index = next_after_latest_done([bool(self.is_done(i)) for i in range(len(steps))])
        elif current == steps[min(self.index, len(steps) - 1)] and s.active_since is not None \
                and s.now - s.active_since >= self.budget_s:
            self.index += 1
        if self.index >= len(steps):
            return Verdict(Operation(OpKind.COMPLETE), "sequence finished")
        target = steps[self.index]
        if current == target:
            return Verdict(Operation(OpKind.CONTINUE), f"step {self.index + 1}/{len(steps)} in progress")
        return Verdict(Operation(OpKind.SWITCH_TO, subtask=target),
                       "start" if current is None else f"step {self.index + 1}/{len(steps)}")


class TruthVerifier:
    """`Verifier` stand-in: `done(step)` reads the simulator's completion check of an authored step."""

    name = "truth"

    def __init__(self, done: Callable[[str], bool | None]):
        self.done = done

    def check(self, images, steps: list[str]) -> dict[str, dict[str, float]]:
        out = {}
        for step in steps:
            truth = self.done(step)
            p = 0.0 if truth is None else float(truth)
            out[step] = {"done": p, "possible": 1.0 - p, "blocked": 0.0}
        return out


class SlowDecider:
    def __init__(self, inner, delay_s: float):
        self.inner, self.delay_s, self.name = inner, delay_s, f"{inner.name}+delay{delay_s:g}s"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        time.sleep(self.delay_s)
        return self.inner.decide(s)


class FailingDecider:
    def __init__(self, inner, triggers: tuple[str, ...]):
        self.inner, self.triggers, self.name = inner, triggers, f"{inner.name}+fail"

    def decide(self, s: ObservationSnapshot) -> Verdict:
        if s.trigger in self.triggers:
            raise RuntimeError(f"injected failure on trigger {s.trigger!r}")
        return self.inner.decide(s)
