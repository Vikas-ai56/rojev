"""The S2 bake-off case set, cut from the pinned SO-101 recording (`config/s2_dataset.txt`) at checked step ends.

  uv run python cases.py label --episodes 0 4 8 --labels ../../data/s2-labels/<name>.json --sheets <dir>
      step ends per episode (`steps.step_ends`) into the labels file, plus one contact sheet per episode to check by
      eye; the file's `checked` note is written by the person who looked, and `build` refuses a file without one
  uv run python cases.py build --labels ../../data/s2-labels/<name>.json --out ../../data/s2-cases/so101-steps
      `cases.jsonl` + frames (JPEG) under --out

Moments per episode, from the step ends: start (frame 0), lid (mid lid lift), open (drawer open + 0.5 s), approach
(mid reach for the red cube), carry (mid carry of the red cube), after (release + 0.8 s, when both cameras show red on
blue; later the arm hides it from the top camera).

Case contract (one JSON object per line), shared with every other case set (e.g. `rollouts.py`):
  id, source, kind (`next_step` | `status`), episode, frame, goal
  next_step: record (the step ledger shown to the planner: a list of labels, or null when none is shown), variant,
             options (steps + COMPLETE), truth (the step the scene needs next); images {top: [the frame]}
  status:    step (a step label or the goal), truth (`done` | `in_progress` | `failed`); images {camera: [the frame]}
  images:    per camera, the path (relative to the case dir) of `frame` alone: what the live planner and done checker
             see, the newest frame; what came before reaches the planner as text, its step record
Next-step variants: `record` (the true ledger: the deployed case), `no_record` (images only: does it read the scene),
`contradiction` (the ledger claims a step the scene shows undone, as after a silent failure: the right answer is that
step again). Status asks the same frame about two steps where it matters (carry: red cube picked = done, stacked = in
progress), so a model that reads the image but not the step cannot score well.
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from recording import Recording, read_pin
from steps import step_ends

PIN_FILE = Path(__file__).resolve().parents[2] / "config/s2_dataset.txt"
CAMERAS = ("top", "wrist")
GOAL = "open the drawer, pick up the red cube, then stack it on the blue cube"
OPEN, PICK, STACK = "open the drawer", "pick up the red cube", "stack the red cube on the blue cube"
STEPS = [OPEN, PICK, STACK]
COMPLETE = "the task is complete"
SETTLE_OPEN = 15
SETTLE_AFTER = 25
JPEG_QUALITY = 95

SOURCE = "so101-steps"
NEXT_STEP = [("start", "record", [], OPEN), ("open", "record", [OPEN], PICK),
             ("carry", "record", [OPEN, PICK], STACK), ("after", "record", [OPEN, PICK, STACK], COMPLETE),
             ("start", "no_record", None, OPEN), ("open", "no_record", None, PICK),
             ("carry", "no_record", None, STACK), ("after", "no_record", None, COMPLETE),
             ("start", "contradiction", [OPEN], OPEN), ("open", "contradiction", [OPEN, PICK], PICK)]
STATUS = [(OPEN, "lid", "in_progress"), (OPEN, "open", "done"), (OPEN, "after", "done"),
          (PICK, "approach", "in_progress"), (PICK, "carry", "done"),
          (STACK, "carry", "in_progress"), (STACK, "after", "done"),
          (GOAL, "carry", "in_progress"), (GOAL, "after", "done")]


def moments(ends: dict) -> dict[str, int]:
    return {"start": 0, "lid": (ends["lid_closed"] + ends["drawer_open"]) // 2,
            "open": ends["drawer_open"] + SETTLE_OPEN, "approach": (ends["drawer_open"] + ends["grasped"]) // 2,
            "carry": (ends["grasped"] + ends["released"]) // 2, "after": ends["released"] + SETTLE_AFTER}


def plan(labels: dict) -> list[dict]:
    cases = []
    for episode, ends in sorted(labels["episodes"].items(), key=lambda kv: int(kv[0])):
        at = moments(ends)
        base = {"source": SOURCE, "episode": int(episode), "goal": GOAL}
        for n, (moment, variant, record, truth) in enumerate(NEXT_STEP):
            cases.append({**base, "id": f"ep{int(episode):02d}-next-{n}", "kind": "next_step", "frame": at[moment],
                          "record": record, "variant": variant, "options": STEPS + [COMPLETE], "truth": truth,
                          "frames": {"top": [at[moment]]}})
        for n, (step, moment, truth) in enumerate(STATUS):
            cases.append({**base, "id": f"ep{int(episode):02d}-status-{n}", "kind": "status", "frame": at[moment],
                          "step": step, "truth": truth,
                          "frames": {camera: [at[moment]] for camera in CAMERAS}})
    return cases


def frame_path(episode: int, camera: str, frame: int) -> str:
    return f"frames/ep{episode:02d}/{camera}/f{frame:04d}.jpg"


def build(labels_file: Path, out: Path) -> int:
    labels = json.loads(labels_file.read_text())
    if not labels.get("checked"):
        raise SystemExit(f"{labels_file}: no `checked` note; look at the contact sheets first")
    if labels["dataset"] != " @ ".join(read_pin(PIN_FILE)):
        raise SystemExit(f"{labels_file} labels {labels['dataset']}, but {PIN_FILE} pins another revision")
    rec = Recording.fetch(PIN_FILE, CAMERAS)
    cases = plan(labels)
    wanted: dict[tuple[int, str], set[int]] = {}
    for case in cases:
        for camera, frames in case["frames"].items():
            wanted.setdefault((case["episode"], camera), set()).update(frames)
    for (episode, camera), frames in sorted(wanted.items()):
        ordered = sorted(frames)
        for frame, image in zip(ordered, rec.frames(episode, ordered, camera)):
            path = out / frame_path(episode, camera, frame)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(path, quality=JPEG_QUALITY)
    with (out / "cases.jsonl").open("w") as f:
        for case in cases:
            images = {camera: [frame_path(case["episode"], camera, i) for i in frames]
                      for camera, frames in case.pop("frames").items()}
            f.write(json.dumps({**case, "images": images}) + "\n")
    (out / "source.json").write_text(json.dumps({"dataset": labels["dataset"], "labels": labels_file.name,
                                                 "checked": labels["checked"], "cases": len(cases)}, indent=1))
    return len(cases)


def label(episodes: list[int], labels_file: Path, sheets: Path) -> None:
    rec = Recording.fetch(PIN_FILE, CAMERAS)
    entries, skipped = {}, {}
    sheets.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        try:
            ends = step_ends(rec.states(episode)[:, 5])
        except ValueError as e:
            skipped[str(episode)] = str(e)
            continue
        entries[str(episode)] = {k: getattr(ends, k) for k in ("lid_closed", "drawer_open", "grasped", "released",
                                                               "length")}
        at = moments(entries[str(episode)])
        shown = ["start", "lid", "open", "approach", "carry", "after"]
        sheet = Image.new("RGB", (len(shown) * 320, 180 + 240), "white")
        for column, name in enumerate(shown):
            for camera, top, size in (("top", 0, (320, 180)), ("wrist", 180, (320, 240))):
                image = Image.fromarray(rec.frames(episode, [at[name]], camera)[0]).resize(size)
                ImageDraw.Draw(image).text((4, 4), f"ep{episode} {name} f{at[name]}", fill=(255, 255, 0))
                sheet.paste(image, (column * 320, top))
        sheet.save(sheets / f"ep{episode:02d}.png")
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    labels_file.write_text(json.dumps({"dataset": " @ ".join(read_pin(PIN_FILE)), "checked": None,
                                       "skipped": skipped, "episodes": entries}, indent=1) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    a = sub.add_parser("label")
    a.add_argument("--episodes", type=int, nargs="+", required=True)
    a.add_argument("--labels", type=Path, required=True)
    a.add_argument("--sheets", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--labels", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.what == "label":
        label(args.episodes, args.labels, args.sheets)
    else:
        print(f"{build(args.labels, args.out)} cases in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
