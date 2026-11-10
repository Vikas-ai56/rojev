"""The bench without vLLM: a stand-in engine answers in R1.5's format, so the questions built from cases (one per
next-step case, one per camera for status), the official prompt forms, the strict answer reading and the written
answers and timing can be checked on a tiny case set."""
import json

from PIL import Image

import prompts
from bench import run

OPTIONS = ["open the drawer", "pick up the red cube", "the task is complete"]


class Request:
    def __init__(self, images, text):
        self.images, self.text = images, text


class Reply:
    def __init__(self, text):
        self.text, self.new_tokens = text, len(text.split())


class StandIn:
    def __init__(self):
        self.seen = []

    def generate(self, requests):
        self.seen += requests
        out = []
        for r in requests:
            if "immediate next step" in r.text:
                out.append(Reply("<think>the lid is up</think><answer>Pick up the red cube.</answer>"))
            elif len(r.images) == 2:
                out.append(Reply("<answer>yes</answer>"))
            else:
                out.append(Reply("it looks satisfied to me"))
        return out

    def latency_ms(self, request):
        return 5.0


def cases_dir(tmp_path):
    for name in ("a", "b", "c"):
        Image.new("RGB", (32, 18), "red").save(tmp_path / f"{name}.jpg")
    cases = [{"id": "n1", "kind": "next_step", "goal": "g", "record": ["open the drawer"], "options": OPTIONS,
              "images": {"top": ["a.jpg"]}},
             {"id": "s1", "kind": "status", "step": "open the drawer",
              "images": {"top": ["a.jpg", "b.jpg"], "wrist": ["c.jpg"]}}]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases))
    return tmp_path


def test_answers_are_read_strictly_one_line_per_question(tmp_path):
    engine = StandIn()
    timing = run(engine, Request, cases_dir(tmp_path), tmp_path / "out")
    answers = [json.loads(line) for line in (tmp_path / "out/answers.jsonl").read_text().splitlines()]
    assert [(a["id"], a["camera"]) for a in answers] == [("n1", "top"), ("s1", "top"), ("s1", "wrist")]
    assert answers[0]["answer"] == "pick up the red cube" and answers[0]["format_ok"]
    assert answers[1]["answer"] is True and answers[1]["format_ok"]
    assert answers[2]["answer"] is None and not answers[2]["format_ok"]
    assert timing["questions"] == 3 and timing["single_question_ms"]["status"] == [5.0, 5.0]
    assert "1- open the drawer" in engine.seen[0].text and engine.seen[0].text.endswith("</answer>\n")


def test_prompt_forms_follow_the_official_sources():
    assert "last steps" not in prompts.next_step("g", None, OPTIONS)
    assert "current goal is: g. last steps: 1- open the drawer. what is the immediate next step?" in \
        prompts.next_step("g", ["open the drawer"], OPTIONS)
    assert prompts.status("open the drawer").startswith("open the drawer. is it satisfied?\n")
    assert prompts.read_next_step("<answer>stack it</answer>", OPTIONS) == prompts.Parsed("stack it", False)
    assert prompts.read_status("<answer>No.</answer>") == prompts.Parsed(False, True)
