"""Deterministic candidate builder: a short list of subtasks for the decision model to choose from.

Fitted once from ordered subtask timelines of the executor's training episodes (no model calls at runtime):
  A. transition graph   P(next subtask | last completed subtask), with a START state for the first step
  B. similar tasks      BM25 between the instruction and training task strings -> subtasks those tasks used
  C. object overlap     words shared between the instruction (and optional scene evidence) and a subtask
  D. canonical wording  one most-frequent surface form per normalised meaning
Scores from A, B, C are each scaled to [0, 1] and summed; the top-k canonical subtasks are returned.
See docs/paper/method.md §3.

`SYNONYMS` is an explicit, reviewed table (normalisation only; extend deliberately, never automatically).
"""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

START = "<start>"
ARTICLES = {"a", "an", "the"}
STOPWORDS = ARTICLES | {"to", "of", "and", "in", "on", "into", "onto", "from", "with", "it", "its", "then", "for", "at",
                        "by", "up", "down", "is", "are", "that", "this", "please", "make", "sure"}
SYNONYMS = {"fridge": "refrigerator"}


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and articles, apply SYNONYMS; the key that identifies one meaning."""
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    return " ".join(SYNONYMS.get(w, w) for w in words if w not in ARTICLES)


def content_words(text: str) -> set[str]:
    return {w for w in normalize(text).split() if w not in STOPWORDS}


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / max(1, len(docs))
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for w in query:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


@dataclass
class CandidateBuilder:
    surface: dict[str, str]  
    transitions: dict[str, Counter]  
    task_subtasks: dict[str, Counter]  
    task_names: list[str]
    bm25: BM25
    words: dict[str, set[str]] = field(default_factory=dict)  
    similar_tasks: int = 10
    _task_scores: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)  

    @classmethod
    def fit(cls, episodes: list[tuple[str, list[str]]], similar_tasks: int = 10) -> "CandidateBuilder":
        """`episodes` = [(task string, ordered subtask list), ...] from the executor's training data."""
        forms: dict[str, Counter] = defaultdict(Counter)
        transitions: dict[str, Counter] = defaultdict(Counter)
        task_subtasks: dict[str, Counter] = defaultdict(Counter)
        for task, subtasks in episodes:
            prev = START
            for s in subtasks:
                k = normalize(s)
                forms[k][s] += 1
                transitions[prev][k] += 1
                task_subtasks[task][k] += 1
                prev = k
        surface = {k: c.most_common(1)[0][0] for k, c in forms.items()}
        names = sorted(task_subtasks)
        return cls(surface, dict(transitions), dict(task_subtasks), names,
                   BM25([sorted(content_words(t)) for t in names]),
                   {k: content_words(k) for k in surface}, similar_tasks)

    def similar_tasks_ranked(self, task: str) -> list[tuple[str, float]]:
        """Up to `similar_tasks` training task strings most similar to `task` (BM25 > 0), best first."""
        sims = self.bm25.scores(sorted(content_words(task)))
        ranked = sorted(range(len(sims)), key=lambda i: (-sims[i], i))[: self.similar_tasks]
        return [(self.task_names[i], sims[i]) for i in ranked if sims[i] > 0]

    def _similar_task_scores(self, task: str) -> dict[str, float]:
        """Source B for one instruction, scaled to [0, 1]; cached because it does not change between steps."""
        if task not in self._task_scores:
            b: dict[str, float] = defaultdict(float)
            for name, sim in self.similar_tasks_ranked(task):
                for k, c in self.task_subtasks[name].items():
                    b[k] += sim * c
            top = max(b.values(), default=0.0)
            self._task_scores[task] = {k: v / top for k, v in b.items()} if top > 0 else {}
        return self._task_scores[task]

    def scores(self, task: str, history: list[str], evidence_words: set[str] = frozenset(),
               weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> dict[str, float]:
        """Per-key score = wA·A + wB·B + wC·C, each source scaled to [0, 1]: A = transitions from the last subtask,
        B = similar tasks, C = word overlap with the task and evidence (only re-ranks keys proposed by A or B)."""
        wa, wb, wc = weights
        total: dict[str, float] = defaultdict(float)
        prev = normalize(history[-1]) if history else START
        nxt = self.transitions.get(prev)
        if nxt and wa:
            top = max(nxt.values())
            for k, c in nxt.items():
                total[k] += wa * c / top
        if wb:
            for k, v in self._similar_task_scores(task).items():
                total[k] += wb * v
        query = content_words(task) | set(evidence_words)
        if wc and query:
            for k in list(total):
                w = self.words.get(k, set())
                if w:
                    total[k] += wc * len(w & query) / len(w)
        return total

    def shortlist(self, task: str, history: list[str], k: int, evidence_words: set[str] = frozenset(),
                  weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> list[str]:
        """Top-k subtasks (canonical phrasing), best first; ties broken alphabetically for determinism."""
        s = self.scores(task, history, evidence_words, weights)
        keys = sorted(s, key=lambda key: (-s[key], key))[:k]
        return [self.surface[key] for key in keys]
