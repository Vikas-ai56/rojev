"""Postconditions: what must be true in the images after a step, as one yes / no question per step (spec §9: skill
templates give a step its postconditions; plan p. 8: completion needs explicit success conditions).

A step text is matched against skill templates (verb + object + destination, robot-agnostic language); the template
gives a concrete visual fact ("Is the steak inside the microwave?") instead of the abstract "is this step done?",
which both checkers answered at chance (done-check A/B 2026-10-22). A step that matches no template has no
postcondition here (`fact` returns None): it cannot be confirmed by a check, only released by its commitment bound.
"""
import re

PLACE = re.compile(r"^(?:place|put|move|load)\s+(?:the\s+)?(?P<obj>.+?)\s+(?P<rel>in|into|inside|on|onto)\s+"
                   r"(?:the\s+)?(?P<dest>.+)$", re.I)
TEMPLATES = (
    (PLACE,
     lambda m: f"Is the {m['obj']} {'on' if m['rel'].lower() in ('on', 'onto') else 'inside'} the {m['dest']}?"),
    (re.compile(r"^close\s+(?:the\s+)?(?P<what>.+)$", re.I), lambda m: f"Is the {m['what']} closed?"),
    (re.compile(r"^open\s+(?:the\s+)?(?P<what>.+)$", re.I), lambda m: f"Is the {m['what']} open?"),
    (re.compile(r"^(?:pick up|grab|grasp)\s+(?:the\s+)?(?P<obj>.+)$", re.I),
     lambda m: f"Is the robot holding the {m['obj']}?"),
    (re.compile(r"^(?:press|push)\s+(?:the\s+)?start button$|^turn on\s+(?:the\s+)?(?P<what>.+)$", re.I),
     lambda m: f"Is the {m['what'] or 'appliance'} switched on?"),
)


def fact(step: str) -> str | None:
    """The yes / no postcondition question for `step`, or None if no template matches."""
    text = step.strip().rstrip(".")
    for pattern, render in TEMPLATES:
        match = pattern.match(text)
        if match:
            return render(match)
    return None


def placement(step: str) -> tuple[str, str, str] | None:
    """(object, "inside" | "on", destination) of a placing step, or None for any other step."""
    match = PLACE.match(step.strip().rstrip("."))
    if match is None:
        return None
    return match["obj"], "on" if match["rel"].lower() in ("on", "onto") else "inside", match["dest"]
