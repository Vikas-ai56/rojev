"""A vision-language model behind an OpenAI-compatible chat endpoint (`POST /v1/chat/completions`, e.g. vLLM's own
`vllm serve`), and the two S2 slots it fills: the planner's next step (`ChatNextStep`, a `NextStepModel`) and the
done checker (`ChatVerifier`, a `Verifier`).

No model is named here. How the model is asked and read comes from its prompt config (`config/s2/<model>.yaml`,
`PromptConfig`): the served model name, a wrapper around every question, the next-step and status questions, how the
final answer is cut out of the reply (a regex, last match), the yes / no words (`yes_words`, `no_words`: YAML reads a
bare `yes:` key as a boolean), and the sampling settings. Images go as
lossless PNG data URLs, the newest frame of each camera; what came before reaches the planner as text (its step
record), as the S2 bake-off measured it. The done checker asks once per camera and calls a step done when any camera
answers yes (the bake-off's "either camera" figure: the wrist sees grasps the top camera misses). Answers are read
strictly: a next step must equal one of the allowed steps (case and a trailing full stop aside), a status answer must
be a yes or a no word; anything else is reported as no step / not done, so a malformed answer never moves the robot
and never confirms a step. One kept HTTP connection (`rojev.transport.HttpTransport`), bearer token.
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from rojev.transport import HttpTransport

from ..planning import NextStep
from .images import png_b64

JSON = "application/json"


@dataclass(frozen=True)
class PromptConfig:
    model: str
    wrapper: str
    next_step: str
    history: str
    history_step: str
    options_separator: str
    status: str
    answer: str
    yes_words: tuple[str, ...]
    no_words: tuple[str, ...]
    sampling: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "PromptConfig":
        values = yaml.safe_load(Path(path).read_text())
        values["yes_words"], values["no_words"] = tuple(values["yes_words"]), tuple(values["no_words"])
        return cls(**values)

    def next_step_question(self, goal: str, record: list[str] | None, options: list[str]) -> str:
        steps = " ".join(self.history_step.format(index=i, step=s) for i, s in enumerate(record or [], 1))
        history = self.history.format(steps=steps) if record else ""
        question = self.next_step.format(goal=goal, history=history, options=self.options_separator.join(options))
        return self.wrapper.format(question=question)

    def status_question(self, step: str) -> str:
        return self.wrapper.format(question=self.status.format(step=step))

    def final_answer(self, reply: str) -> str | None:
        found = re.findall(self.answer, reply, flags=re.DOTALL)
        return found[-1].strip() if found else None


class ChatVLM:
    def __init__(self, url: str, token: str, prompts: PromptConfig, timeout_s: float = 120.0):
        self.prompts, self.timeout_s = prompts, timeout_s
        self.transport = HttpTransport(url, token, timeout_s)
        self.log: list[dict] = []

    def ask(self, images: list[np.ndarray], text: str) -> str:
        content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64(im)}"}}
                   for im in images] + [{"type": "text", "text": text}]
        body = {"model": self.prompts.model, "messages": [{"role": "user", "content": content}],
                **self.prompts.sampling}
        w0 = time.perf_counter()
        raw = self.transport.request("POST", "/v1/chat/completions", json.dumps(body).encode(), self.timeout_s, JSON)
        reply = json.loads(raw)["choices"][0]["message"]["content"] or ""
        self.log.append({"images": len(images), "wall_ms": round((time.perf_counter() - w0) * 1000, 1),
                         "question": text[-200:], "reply": reply[-400:]})
        return reply


class ChatNextStep:
    def __init__(self, vlm: ChatVLM):
        self.vlm = vlm
        self.name = vlm.prompts.model

    def next_step(self, goal: str, record: list[str], options: list[str],
                  images: dict[str, np.ndarray]) -> NextStep:
        reply = self.vlm.ask(list(images.values()), self.vlm.prompts.next_step_question(goal, record, options))
        answer = self.vlm.prompts.final_answer(reply)
        normal = (answer or "").strip().strip(".").lower()
        chosen = next((o for o in options if o.lower() == normal), None)
        return NextStep(chosen, reply[-300:] if chosen is None else f"chose {chosen}")


class ChatVerifier:
    def __init__(self, vlm: ChatVLM):
        self.vlm = vlm
        self.name = vlm.prompts.model

    def check(self, images: list[np.ndarray], steps: list[str]) -> dict[str, dict[str, float]]:
        out = {}
        for step in steps:
            done = 1.0 if any(self._says_yes(image, step) for image in images) else 0.0
            out[step] = {"done": done, "possible": 1.0 - done, "blocked": 0.0}
        return out

    def _says_yes(self, image: np.ndarray, step: str) -> bool:
        answer = self.vlm.prompts.final_answer(self.vlm.ask([image], self.vlm.prompts.status_question(step)))
        return (answer or "").strip().strip(".").lower() in self.vlm.prompts.yes_words
