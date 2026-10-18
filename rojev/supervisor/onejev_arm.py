"""Arm D: OneJev picks the next subtask from the current images; MJev-Spatial judges the candidates.

One generic rule, no per-subtask or per-object code (docs/features/v2-arm-d.md):
  1. candidates  shortlist (`CandidateBuilder`, K) given the task and the subtasks so far
  2. evidence    MJev, same template for every judged subtask s (current + recent + top M candidates):
                 "Robot step: '<s>'. Which is true right now?"  → done / possible / blocked
  3. decision    one OneJev Choice over {continue current, candidates, recent (redo), task complete} on the
                 composite image, with the evidence table and the task-progress summary in its state
Task-progress summary (task-level memory, no generated text): one entry per subtask the arm moved past, with its
step range, the outcome at the switch (MJev: judged done / left unfinished; redo → "undone, redoing") and its latest
re-checked P(done). Recent entries are re-judged every decision, so a stale record shows its age. The images, not the
summary, are the truth.
Logged alongside: the SayCan-style product P_OneJev(option) × P_MJev(possible) [ahn2022saycan] (no extra calls).
Option order is shuffled per decision (seeded, logged) against position bias [PriDe].
"""
import random
import time

import numpy as np
from qev import Choice, Client

from rojev import onejev
from rojev.perception import RemoteMJev

from .base import Decision, DecisionContext, SubtaskSource
from .candidates import CandidateBuilder

STATES = {"done": "done: its goal is already achieved",
          "possible": "possible: it can be started now; what it needs is present and reachable",
          "blocked": "blocked: it cannot be done now; something it needs is missing, hidden, or not ready"}
COMPLETE = "the whole task is complete; stop"


def evidence_question(i: int, subtask: str) -> dict:
    return {"id": f"s{i}", "question": f"Robot step: '{subtask}'. Which is true right now?",
            "answer_space": {"kind": "choice", "options": [{"id": k, "text": v} for k, v in STATES.items()]}}


class OneJevSupervisor(SubtaskSource):
    name = "D-onejev"

    def __init__(self, client: Client, candidates: CandidateBuilder, mjev: RemoteMJev | None,
                 cameras: tuple[str, ...], k: int = 20, m: int = 5, recent: int = 2, seed: int = 0,
                 session_id: str | None = None):
        self.client, self.candidates, self.mjev, self.cameras = client, candidates, mjev, cameras
        self.k, self.m, self.recent, self.seed, self.session_id = k, m, recent, seed, session_id
        self.progress: list[dict] = []  # task-progress summary, oldest first
        self.started = 0  # step at which the current subtask began

    def _evidence(self, images: list[np.ndarray], judged: list[str]) -> tuple[dict, dict]:
        """{subtask: {done, possible, blocked}} from one batched MJev request."""
        if self.mjev is None or not judged:
            return {}, {}
        out = self.mjev.ask(images, [evidence_question(i, s) for i, s in enumerate(judged)])
        table = {s: out["answers"][f"s{i}"]["probabilities"] for i, s in enumerate(judged)}
        return table, out["timings_ms"]

    def decide(self, ctx: DecisionContext) -> Decision:
        obs, current, history = ctx.observation, ctx.current, list(ctx.history)
        images = [obs.images[c] for c in self.cameras]
        # Next steps assume the current one is finished; "continue current" stays an option either way.
        shortlist = [s for s in self.candidates.shortlist(obs.task, history + ([current] if current else []), self.k)
                     if s != current]
        recent = [h for h in history[-self.recent:] if h != current]
        judged = list(dict.fromkeys(([current] if current else []) + recent + shortlist[: self.m]))
        evidence, mjev_ms = self._evidence(images, judged)

        texts = list(dict.fromkeys(shortlist + recent))
        random.Random(f"{self.seed}:{ctx.chunk}").shuffle(texts)
        options = {f"o{i}": t for i, t in enumerate(texts)}
        if current:
            options["continue"] = f"continue the current subtask: {current}"
        options["complete"] = COMPLETE
        for entry in self.progress:  # re-checked entries get fresh evidence
            if entry["subtask"] in evidence:
                entry.update(p_done_now=round(evidence[entry["subtask"]]["done"], 2), checked_at_step=ctx.step)
        state = {"image": "<image:1> = robot cameras side by side: " + ", ".join(self.cameras),
                 "task": obs.task, "task_progress": self.progress,
                 "task_progress_note": "a record of earlier subtasks; trust the current images over it",
                 "current_subtask": current, "chunks_spent_on_current": ctx.chunks_on_current}
        if evidence:
            state["evidence_per_subtask"] = {s: {k: round(p, 2) for k, p in v.items()} for s, v in evidence.items()}

        t0 = time.perf_counter()
        r = onejev.system_one(self.client, np.concatenate(images, axis=1), state, {"next": Choice(
            "Which should the robot do now to complete the task, judging by the images now?", options)},
            span_name="onejev.next", session_id=self.session_id)
        onejev_ms = round((time.perf_counter() - t0) * 1000, 1)
        choice, probs = r.answers["next"].choice, r.answers["next"].probabilities

        subtask_of = {k: (current if k == "continue" else v) for k, v in options.items() if k != "complete"}
        saycan = sorted(((subtask_of[k], round(probs[k] * evidence[subtask_of[k]]["possible"], 4))
                         for k in subtask_of if subtask_of[k] in evidence), key=lambda x: -x[1])[:3]
        top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
        detail = {"evidence": evidence, "judged": judged, "options": list(options.values()), "chosen": options[choice],
                  "top5": [(options[k], round(p, 4)) for k, p in top], "saycan_top3": saycan,
                  "timings_ms": {"mjev": mjev_ms, "onejev": onejev_ms}}
        p = probs[choice]
        if choice != "continue":
            self._record(current, ctx.step, evidence, redo=None if choice == "complete" else options[choice])
        detail["task_progress"] = [dict(e) for e in self.progress]
        if choice == "complete":
            return Decision(None, f"task complete (p={p:.2f})", detail)
        if choice == "continue":
            return Decision(current, f"continue (p={p:.2f})", detail)
        return Decision(options[choice], f"{'start' if current is None else 'switch'} (p={p:.2f})", detail)

    def _record(self, left: str | None, step: int, evidence: dict, redo: str | None) -> None:
        """Close the entry for the subtask being left; mark a chosen earlier subtask as being redone."""
        if left is not None:
            e = evidence.get(left)
            outcome = ("no evidence" if e is None else
                       "judged done" if max(e, key=e.get) == "done" else "left unfinished")
            self.progress.append({"subtask": left, "steps": [self.started, step], "outcome": outcome,
                                  "p_done_now": None if e is None else round(e["done"], 2), "checked_at_step": step})
        for entry in self.progress:
            if redo is not None and entry["subtask"] == redo:
                entry["outcome"] = "undone, redoing"
        self.started = step
