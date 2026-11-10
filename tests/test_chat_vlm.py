"""The chat client against a stand-in OpenAI-compatible server: it sends the served model name, the images as PNG data
URLs before the question, the bearer token and the sampling settings from the prompt config (R1.5's real config);
the planner slot accepts only an allowed step and the done-checker slot only a yes word, so a malformed or off-list
answer never moves the robot and never confirms a step; the done checker asks each camera alone and a yes from any
camera is done (the bake-off's rule)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from rojev.system2.models.chat_vlm import ChatNextStep, ChatVerifier, ChatVLM, PromptConfig

PROMPTS = PromptConfig.load(Path(__file__).resolve().parents[1] / "config/s2/embodied-r1.5.yaml")
OPTIONS = ["open the middle drawer", "turn on the stove", "the task is complete"]
IMAGE = np.zeros((4, 4, 3), np.uint8)


@pytest.fixture
def server():
    seen, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "auth": self.headers["Authorization"],
                         "type": self.headers["Content-Type"], "body": body})
            data = json.dumps({"choices": [{"message": {"content": replies.pop(0)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", seen, replies
    httpd.shutdown()


def test_the_request_carries_model_images_question_token_and_sampling(server):
    url, seen, replies = server
    replies.append("<think>the drawer is shut</think><answer>Open the middle drawer.</answer>")
    planner = ChatNextStep(ChatVLM(url, "secret", PROMPTS))
    answer = planner.next_step("open the drawer and turn on the stove", ["turn on the stove"], OPTIONS,
                               {"top": IMAGE, "wrist": IMAGE})
    assert answer.step == "open the middle drawer"
    request = seen[0]
    assert request["path"] == "/v1/chat/completions" and request["auth"] == "Bearer secret"
    assert request["type"] == "application/json" and request["body"]["model"] == "Embodied-R1.5"
    content = request["body"]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image_url", "image_url", "text"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "last steps: 1- turn on the stove." in content[2]["text"] and content[2]["text"].endswith("</answer>\n")
    assert request["body"]["temperature"] == 1.0 and request["body"]["max_tokens"] == 1024


def test_off_list_and_malformed_answers_never_move_or_confirm(server):
    url, _, replies = server
    replies.extend(["<answer>open the oven</answer>", "no answer tag", "<answer>Yes.</answer>",
                    "<answer>maybe</answer>"])
    vlm = ChatVLM(url, "secret", PROMPTS)
    assert ChatNextStep(vlm).next_step("g", [], OPTIONS, {"top": IMAGE}).step is None
    assert ChatNextStep(vlm).next_step("g", [], OPTIONS, {"top": IMAGE}).step is None
    states = ChatVerifier(vlm).check([IMAGE], ["open the middle drawer", "turn on the stove"])
    assert states["open the middle drawer"]["done"] == 1.0 and states["turn on the stove"]["done"] == 0.0


def test_done_check_asks_each_camera_alone_and_any_yes_is_done(server):
    url, requests, replies = server
    replies.extend(["<answer>no</answer>", "<answer>yes</answer>", "<answer>no</answer>", "<answer>no</answer>"])
    verifier = ChatVerifier(ChatVLM(url, "secret", PROMPTS))
    states = verifier.check([IMAGE, IMAGE], ["put the bowl on the plate", "turn on the stove"])
    assert states["put the bowl on the plate"]["done"] == 1.0 and states["turn on the stove"]["done"] == 0.0
    contents = [r["body"]["messages"][0]["content"] for r in requests]
    assert len(contents) == 4 and all([c["type"] for c in content] == ["image_url", "text"] for content in contents)
