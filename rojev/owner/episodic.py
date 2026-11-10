"""Episodic memory: an append-only log of finished tasks, one JSON object per line (docs/target-system.md, Memory).

Written by the state owner once per task (COMPLETE accepted, or the owner stopped first), read later for feasibility
statistics and training. Each line: time, outcome, goal, the step ledger (step, status, command, instruction revision,
evidence, checked, time, and the observation id and time it was judged on; images stay with the run's recording) and
the dialogue. Everything is kept, failures included: a task that stopped short is as informative as one that finished.
"""
import json
import threading
from pathlib import Path

from rojev.protocol import TaskMemory, Turn


class EpisodicLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, memory: TaskMemory, outcome: str, t: float, dialogue: tuple[Turn, ...] = ()) -> None:
        line = {"t": round(t, 3), "outcome": outcome, "goal": memory.goal,
                "steps": [{"step": r.step, "status": r.status.value, "cmd": r.cmd_id, "rev": r.instruction_rev,
                           "evidence": r.evidence, "checked": r.checked, "t": round(r.t, 3),
                           "obs_id": r.capture.obs_id, "obs_t": round(r.capture.t, 3)} for r in memory.records],
                "dialogue": [{"t": round(d.t, 3), "speaker": d.speaker, "rev": d.instruction_rev, "text": d.text,
                              "template": d.template, "objects": list(d.objects)} for d in dialogue]}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(line) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
