"""OneJev System One calls, each recorded as an OpenInference LLM span holding the exact image the model received.

`KeptClient` is qev's `Client` over one kept HTTP connection. qev's own client opens a connection per request
(`urllib.request.urlopen`); over the SSH tunnel to a remote box every new connection costs an extra wide-area round
trip (measured 2026-10-24, three 256 x 256 cameras: median 1269 ms per decision with a new connection, 604 ms kept;
server compute 75-140 ms). It overrides the client's two transport methods (`_post`, `_get`, qev pinned in uv.lock);
requests are serialised on one connection, and a broken connection is reopened and the request sent once more
(inference is read-only, so a repeat is safe).
"""
import http.client
import json
import threading
import urllib.error
from typing import Any
from urllib.parse import urlparse

import numpy as np
from opentelemetry import trace
from PIL import Image
from qev import Client
from qev.client import Response
from qev.media import data_uri

_tracer = trace.get_tracer("rojev.onejev")
_MSG = "llm.input_messages.0.message"


Img = np.ndarray | Image.Image


class KeptClient(Client):
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0):
        super().__init__(base_url, api_key=api_key, timeout=timeout)
        url = urlparse(self.base_url)
        self._https, self._host, self._port, self._prefix = url.scheme == "https", url.hostname, url.port, url.path
        self._conn: http.client.HTTPConnection | None = None
        self._lock = threading.Lock()

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            kind = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            self._conn = kind(self._host, self._port, timeout=self.timeout)
        return self._conn

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _request(self, method: str, path: str, body: bytes | None) -> tuple[bytes, dict[str, str]]:
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": "qev-client"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        with self._lock:
            for attempt in range(2):
                try:
                    conn = self._connection()
                    conn.request(method, self._prefix + path, body=body, headers=headers)
                    resp = conn.getresponse()
                    payload, resp_headers = resp.read(), dict(resp.getheaders())
                except (http.client.HTTPException, OSError):
                    self._close()
                    if attempt:
                        raise
                    continue
                if resp.status >= 400:
                    raise urllib.error.HTTPError(self.base_url + path, resp.status, payload.decode(errors="replace"),
                                                 resp.msg, None)
                return payload, resp_headers
        raise RuntimeError("unreachable")

    def _post(self, path: str, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        payload, headers = self._request("POST", path, json.dumps(body).encode())
        return json.loads(payload), headers

    def _get(self, path: str) -> dict[str, Any]:
        return json.loads(self._request("GET", path, None)[0])

    def close(self) -> None:
        with self._lock:
            self._close()


def camera_views(names: list[str] | None, count: int) -> str:
    """How the state names the images of one moment: each camera is its own image (`<image:k>`), never views glued
    side by side, so the model sees where one view ends and the next begins; all are taken at the same time."""
    labels = names if names is not None else [f"camera {k}" for k in range(1, count + 1)]
    views = "; ".join(f"<image:{k}> = {label}" for k, label in enumerate(labels, start=1))
    return f"{views} (separate views of the same moment)"


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
