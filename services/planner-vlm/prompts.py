"""The questions Embodied-R1.5 is asked and how its answers are read in the bake-off; the wording comes from
`config/s2/embodied-r1.5.yaml`, the same file the live client (`rojev.system2.models.chat_vlm`) reads, so the bench
and the robot ask the same questions.

  next_step  RoboVQA form `current goal is: <goal>. last steps: 1- a 2- b. what is the immediate next step?` (no
             `last steps` when the case shows no ledger), plus the allowed steps (S1 follows only its trained steps).
  status     the SFT process-detection form `<step>. is it satisfied?` (yes / no); R1.5's Corrector prompt is not public.
Every question goes inside R1.5's official wrapper (think, then `<answer>…</answer>`). Answers are read strictly:
anything outside the asked format is a format failure, never guessed at.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

PROMPTS = yaml.safe_load((Path(__file__).resolve().parents[2] / "config/s2/embodied-r1.5.yaml").read_text())


@dataclass(frozen=True)
class Parsed:
    value: object
    format_ok: bool


def next_step(goal: str, record: list[str] | None, options: list[str]) -> str:
    steps = " ".join(PROMPTS["history_step"].format(index=i, step=s) for i, s in enumerate(record or [], 1))
    history = PROMPTS["history"].format(steps=steps) if record else ""
    question = PROMPTS["next_step"].format(goal=goal, history=history,
                                           options=PROMPTS["options_separator"].join(options))
    return PROMPTS["wrapper"].format(question=question)


def status(step: str) -> str:
    return PROMPTS["wrapper"].format(question=PROMPTS["status"].format(step=step))


def final_answer(text: str) -> str | None:
    found = re.findall(PROMPTS["answer"], text, flags=re.DOTALL)
    return found[-1].strip() if found else None


def read_next_step(text: str, options: list[str]) -> Parsed:
    answer = final_answer(text)
    if answer is None:
        return Parsed(None, False)
    normal = answer.strip().strip(".").lower()
    match = [o for o in options if o.lower() == normal]
    return Parsed(match[0], True) if match else Parsed(answer, False)


def read_status(text: str) -> Parsed:
    answer = final_answer(text)
    if answer is None:
        return Parsed(None, False)
    word = answer.strip().strip(".").lower()
    known = tuple(PROMPTS["yes_words"]) + tuple(PROMPTS["no_words"])
    return Parsed(word in PROMPTS["yes_words"], True) if word in known else Parsed(answer, False)
