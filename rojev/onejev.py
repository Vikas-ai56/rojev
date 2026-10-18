"""OneJev System One calls, each recorded as an OpenInference LLM span holding the exact image the model received."""
import json
from typing import Any

import numpy as np
from opentelemetry import trace
from PIL import Image
from qev import Client
from qev.client import Response
from qev.media import data_uri

_tracer = trace.get_tracer("rojev.onejev")
_MSG = "llm.input_messages.0.message"


Img = np.ndarray | Image.Image


def system_one(client: Client, image: Img | list[Img], state: dict[str, Any], questions: dict[str, Any],
               span_name: str = "onejev", session_id: str | None = None, **params: Any) -> Response:
    """Ask OneJev `questions` about `state`, which refers to the images as `<image:1>`, `<image:2>`, … in order.

    The span records the data URI sent, the state and questions, the answers, qev's debug diagnostics (raw slot
    logits, timing, prompt hash) and each probability as a numeric `onejev.<question>[.<option>]` attribute.
    `params` are qev request options, e.g. `debias=3` (average over cyclic shifts of the option order).
    """
    images = image if isinstance(image, list) else [image]
    uris = [data_uri(i if isinstance(i, Image.Image) else Image.fromarray(i)) for i in images]
    wire_questions = {k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in questions.items()}
    with _tracer.start_as_current_span(span_name) as span:
        span.set_attribute("openinference.span.kind", "LLM")
        if session_id:
            span.set_attribute("session.id", session_id)
        span.set_attribute(f"{_MSG}.role", "user")
        span.set_attribute(f"{_MSG}.contents.0.message_content.type", "text")
        span.set_attribute(f"{_MSG}.contents.0.message_content.text",
                           json.dumps({"state": state, "questions": wire_questions}, indent=1))
        for n, uri in enumerate(uris, start=1):
            span.set_attribute(f"{_MSG}.contents.{n}.message_content.type", "image")
            span.set_attribute(f"{_MSG}.contents.{n}.message_content.image.image.url", uri)
        span.set_attribute("input.value", json.dumps(state))
        span.set_attribute("input.mime_type", "application/json")
        span.set_attribute("onejev.params", json.dumps(params))

        r = client.system_one(state=state, questions=questions, media=[{"type": "image", "data": u} for u in uris],
                              **{"debug": 1, **params})

        span.set_attribute("llm.model_name", r.model)
        span.set_attribute("output.value", json.dumps({k: v.raw for k, v in r.answers.items()}))
        span.set_attribute("output.mime_type", "application/json")
        span.set_attribute("metadata", json.dumps(r.meta))
        for qid, a in r.answers.items():
            if a.raw["type"] == "noul":
                span.set_attribute(f"onejev.{qid}", a.noul)
            elif a.raw["type"] == "choice":
                for opt, p in a.probabilities.items():
                    span.set_attribute(f"onejev.{qid}.{opt}", p)
            elif a.raw["type"] == "score":
                span.set_attribute(f"onejev.{qid}", a.score)
    return r
