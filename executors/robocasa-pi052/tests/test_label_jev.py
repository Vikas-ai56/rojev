"""The labelling CLI writes the chosen act into labels.jsonl and leaves skipped cases unlabelled."""
import json

import numpy as np
from PIL import Image

import label_jev


def make_case(folder, name):
    (folder / name).mkdir(parents=True)
    for cam in ("left", "right"):
        Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(folder / name / f"{cam}.png")
    (folder / name / "case.json").write_text(json.dumps({
        "case": name, "trigger": "instruction", "cameras": ["left", "right"],
        "situation": {"user_just_said": "go on", "task": "t", "current_step": "guarded_wait"},
        "options": {"resume": "go on with it", "continue": "keep holding"}}))
    return {"case": name, "predicted": "continue", "label": None}


def test_a_typed_number_becomes_the_label_and_skip_leaves_it_empty(tmp_path, monkeypatch):
    table = [make_case(tmp_path, "c1"), make_case(tmp_path, "c2")]
    label_jev.save(tmp_path, table)
    answers = iter(["x", "1", "s"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    monkeypatch.setattr(label_jev.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr("sys.argv", ["label_jev.py", str(tmp_path)])
    assert label_jev.main() == 0
    assert [r["label"] for r in label_jev.rows(tmp_path)] == ["resume", None]
    assert (tmp_path / "c1" / "view.png").exists()
