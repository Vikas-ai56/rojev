"""SAM 2.1 against simulator truth: can object masks / tracks give the "placed inside" completion signal? (2026-10-22)

frames  still frames. Frames and truth come from `done_check_ab.py build` (its `masks.npz`: the simulator's
per-camera pixels of the task's object, its container fixture and the gripper). Each stage is scored on its own, so a failure has one owner:
  1. masks from a correct point: SAM 2.1 prompted at the truth mask's most central pixel → IoU with truth, per
     object and container (upper bound of grounding);
  2. MJev pointing: "Point to the <object>." per camera → hit rate (the point lies on the truth object);
  3. masks from MJev's point → IoU with truth (the full grounding chain);
  4. "placed inside" in 2D: the share of object pixels inside the container's filled outline, ≥ `INSIDE` on any camera
     that sees the object → done; scored against the simulator's step truth with truth masks (what 2D geometry can
     do at all), then with SAM masks from correct points. A frame where no camera sees the object has no 2D answer
     and is counted as unseen.
track   a live run recorded with `realtime.py --record-frames S`. In this fresh process each recorded simulator state
        is restored and its truth masks rendered; the recorded images are streamed through one SAM 2.1 tracking
        session in order, started on the first frame where a camera sees the object (object and container prompted
        at correct points). Scored per camera and role: IoU while the object is visible; P(visible) while visible vs
        while hidden (identity through occlusion); re-acquisition after a hidden stretch; the 2D inside rule against
        the simulator's truth of the task's placing step; time per track step.
Correct points are taken from the simulator: a declared spoof that stands in for grounding, never available on a robot.

  uv run python sam_check.py frames results/sam-check/<stamp> --service ~/.config/rojev/services/lambda.json
  uv run python sam_check.py track <realtime run dir> --service ~/.config/rojev/services/lambda.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from harness.tracking import MIN_PIXELS, TRACKED, central_pixel, inside_share, iou, truth_masks
from realtime import RoboCasaBackend
from rojev.system2.models import PointPrompt, RemoteMJev, RemoteSam2
from rojev.system2.verify import placement
from tasks import TASKS
from workflow import CAMS, service_addresses

INSIDE = 0.5


def balanced(rows: list[dict], key: str) -> dict:
    seen = [r for r in rows if r[key] is not None]
    pos = [r for r in seen if r["truth"]]
    neg = [r for r in seen if not r["truth"]]
    recall = sum(r[key] for r in pos) / len(pos) if pos else None
    false_done = sum(r[key] for r in neg) / len(neg) if neg else None
    bal = None if recall is None or false_done is None else round((recall + 1 - false_done) / 2, 3)
    return {"balanced_accuracy": bal, "recall": recall and round(recall, 3), "false_done": false_done and
            round(false_done, 3), "pos": len(pos), "neg": len(neg), "unseen": len(rows) - len(seen)}


def decide(shares: list[float | None]) -> bool | None:
    seen = [s for s in shares if s is not None]
    return max(seen) >= INSIDE if seen else None


def score(folder: Path, service: Path) -> None:
    labels = [json.loads(line) for line in (folder / "labels.jsonl").read_text().splitlines()]
    place = [r for r in labels if placement(r["step"]) is not None]
    addresses = service_addresses(str(service), "serve")
    sam = RemoteSam2(**addresses["sam2"])
    mjev = RemoteMJev(**addresses["mjev"])
    by_frame = defaultdict(list)
    for r in place:
        by_frame[r["frame"]].append(r)
    cams, steps, sam_ms, mjev_ms = [], [], [], []
    for frame, rows in by_frame.items():
        images = [np.asarray(Image.open(folder / frame / f"{c}.png").convert("RGB")) for c in CAMS]
        truth = np.load(folder / frame / "masks.npz")
        obj_name = placement(rows[0]["step"])[0]
        prompts = [PointPrompt(f"{i}:{role}", i, (central_pixel(truth[f"{c}:{role}"]),), (1,))
                   for i, c in enumerate(CAMS) for role in ("object", "container")
                   if truth[f"{c}:{role}"].sum() >= MIN_PIXELS]
        pointed = {}
        for i, c in enumerate(CAMS):
            reply = mjev.ask([images[i]], [{"id": "p", "question": f"Point to the {obj_name}.",
                                            "answer_space": {"kind": "point", "coordinate_system": "normalized_xy",
                                                             "num_points": 1}}])
            mjev_ms.append(reply["timings_ms"]["round_trip"])
            x, y = reply["answers"]["p"]["prediction"]
            h, w = images[i].shape[:2]
            pointed[i] = (min(w - 1.0, x * w), min(h - 1.0, y * h))
        prompts += [PointPrompt(f"{i}:mjev", i, (pointed[i],), (1,)) for i in range(len(CAMS))]
        masks = sam.segment(images, prompts)
        sam_ms.append(sam.last_timings_ms["round_trip"])
        shares_truth, shares_sam = [], []
        for i, c in enumerate(CAMS):
            obj_t, box_t = truth[f"{c}:object"], truth[f"{c}:container"]
            visible = obj_t.sum() >= MIN_PIXELS
            px, py = pointed[i]
            row = {"frame": frame, "camera": c, "object_pixels": int(obj_t.sum()),
                   "container_pixels": int(box_t.sum()), "mjev_point": [round(px, 1), round(py, 1)],
                   "mjev_hit": bool(obj_t[int(py), int(px)]) if visible else None}
            for role, t in (("object", obj_t), ("container", box_t)):
                m = masks.get(f"{i}:{role}")
                row[f"{role}_iou"] = None if m is None else round(iou(m.mask, t), 3)
                row[f"{role}_score"] = None if m is None else m.score
            row["mjev_object_iou"] = round(iou(masks[f"{i}:mjev"].mask, obj_t), 3) if visible else None
            cams.append(row)
            shares_truth.append(inside_share(obj_t, box_t) if visible else None)
            sam_obj, sam_box = masks.get(f"{i}:object"), masks.get(f"{i}:container")
            shares_sam.append(inside_share(sam_obj.mask, sam_box.mask)
                              if visible and sam_obj is not None and sam_box is not None else None)
        for r in rows:
            steps.append({**r, "shares_truth": shares_truth, "shares_sam": shares_sam,
                          "inside_truth_masks": decide(shares_truth), "inside_sam_masks": decide(shares_sam)})
        print(f"[sam] {frame}: {len(rows)} place labels", flush=True)

    def mean(key, rows):
        values = [r[key] for r in rows if r[key] is not None]
        return {"mean": round(float(np.mean(values)), 3), "n": len(values)} if values else None

    visible = [r for r in cams if r["object_pixels"] >= MIN_PIXELS]
    report = {
        "masks_from_correct_point": {"object_iou": mean("object_iou", visible),
                                     "object_iou_ge_100px": mean("object_iou", [r for r in visible
                                                                                if r["object_pixels"] >= 100]),
                                     "container_iou": mean("container_iou", cams)},
        "mjev_pointing": {"hit_rate": mean("mjev_hit", visible),
                          "hit_rate_ge_100px": mean("mjev_hit", [r for r in visible if r["object_pixels"] >= 100])},
        "masks_from_mjev_point": {"object_iou": mean("mjev_object_iou", visible)},
        "placed_inside_2d": {"truth_masks": balanced(steps, "inside_truth_masks"),
                             "sam_masks_correct_points": balanced(steps, "inside_sam_masks"),
                             "per_task": {t: {"truth_masks": balanced([r for r in steps if r["task"] == t],
                                                                      "inside_truth_masks"),
                                              "sam_masks": balanced([r for r in steps if r["task"] == t],
                                                                    "inside_sam_masks")}
                                          for t in sorted({r["task"] for r in steps})}},
        "ms": {"sam2_round_trip_p50_3_images": float(np.percentile(sam_ms, 50)),
               "mjev_point_round_trip_p50": float(np.percentile(mjev_ms, 50))},
        "inside_threshold": INSIDE, "min_object_pixels": MIN_PIXELS,
    }
    (folder / "sam_cameras.jsonl").write_text("".join(json.dumps(r) + "\n" for r in cams))
    (folder / "sam_steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in steps))
    (folder / "sam_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def track(run: Path, service: Path) -> None:
    config = json.loads((run / "config.json").read_text())
    task = config["task"]
    data = np.load(run / "frames" / "frames.npz")
    cameras = [str(c) for c in data["cameras"]]
    backend = RoboCasaBackend(task, config["seed"], horizon=100)
    kitchen = backend.kitchen
    spec = TASKS.get(task)
    place = next((s for s in spec.steps(kitchen) if placement(s.subtask) is not None), None) if spec else None
    sam = RemoteSam2(**service_addresses(str(service), "serve")["sam2"])
    session, frame, rows = None, 0, []
    for t, state, images in zip(data["t"], data["state"], data["images"], strict=True):
        kitchen.sim.set_state_from_flattened(state)
        kitchen.sim.forward()
        truth = truth_masks(kitchen, task, cameras, images.shape[1:3])
        row = {"t": round(float(t), 3), "placed_truth": None if place is None else bool(place.done(kitchen)),
               "truth_px": {c: int(truth[f"{c}:object"].sum()) for c in cameras}}
        if session is None:
            prompts = [PointPrompt(f"{i}:{role}", i, (central_pixel(truth[f"{c}:{role}"]),), (1,))
                       for i, c in enumerate(cameras) for role in TRACKED if truth[f"{c}:{role}"].sum() >= MIN_PIXELS]
            if not any(p.id.endswith(":object") for p in prompts):
                rows.append({**row, "tracking": False})
                continue
            result = sam.track_start(list(images), prompts)
            session = result.session
        else:
            frame += 1
            result = sam.track_step(session, frame, list(images))
        row |= {"tracking": True, "frame": frame, "ms": sam.last_timings_ms["round_trip"], "cameras": {}}
        for i, c in enumerate(cameras):
            cam = {}
            for role in TRACKED:
                m = result.masks.get(f"{i}:{role}")
                cam[role] = {"truth_px": int(truth[f"{c}:{role}"].sum()),
                             "tracked_px": None if m is None else int(m.mask.sum()),
                             "p_visible": None if m is None else m.score,
                             "iou": None if m is None else round(iou(m.mask, truth[f"{c}:{role}"]), 3)}
            obj, box = result.masks.get(f"{i}:object"), result.masks.get(f"{i}:container")
            cam["inside_share"] = (round(inside_share(obj.mask, box.mask), 3)
                                   if obj is not None and box is not None and obj.mask.sum() else None)
            row["cameras"][c] = cam
        rows.append(row)
    if session is not None:
        sam.track_end(session)
    (run / "frames" / "tracks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = track_report(rows, cameras)
    (run / "frames" / "track_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def track_report(rows: list[dict], cameras: list[str]) -> dict:
    """Per camera: object IoU while visible; P(visible) while visible vs hidden; tracked pixels while hidden (a track
    that keeps a mask on a hidden object is on something else); re-acquisitions; inside rule vs truth."""
    tracked = [r for r in rows if r["tracking"]]
    out = {"frames": len(rows), "frames_before_first_sight": len(rows) - len(tracked),
           "placed_truth_frames": sum(bool(r["placed_truth"]) for r in rows),
           "ms_p50": float(np.percentile([r["ms"] for r in tracked], 50)) if tracked else None, "cameras": {}}
    for c in cameras:
        o = [r["cameras"][c]["object"] for r in tracked if r["cameras"][c]["object"]["p_visible"] is not None]
        seen = [x for x in o if x["truth_px"] >= MIN_PIXELS]
        hidden = [x for x in o if x["truth_px"] < MIN_PIXELS]
        reacquired = sum(1 for prev, cur in zip(o, o[1:])
                         if prev["truth_px"] < MIN_PIXELS <= cur["truth_px"] and cur["iou"] >= 0.5)
        returns = sum(1 for prev, cur in zip(o, o[1:]) if prev["truth_px"] < MIN_PIXELS <= cur["truth_px"])
        inside = [(r["cameras"][c]["inside_share"], r["placed_truth"]) for r in tracked
                  if r["cameras"][c]["inside_share"] is not None and r["placed_truth"] is not None]
        out["cameras"][c] = {
            "tracked_frames": len(o), "visible_frames": len(seen), "hidden_frames": len(hidden),
            "object_iou_visible": round(float(np.mean([x["iou"] for x in seen])), 3) if seen else None,
            "p_visible_when_visible": round(float(np.mean([x["p_visible"] for x in seen])), 3) if seen else None,
            "p_visible_when_hidden": round(float(np.mean([x["p_visible"] for x in hidden])), 3) if hidden else None,
            "mask_on_hidden_object": sum((x["tracked_px"] or 0) > 0 for x in hidden),
            "reacquired": f"{reacquired}/{returns}",
            "inside_rule_vs_truth": {"inside_and_placed": sum(s >= INSIDE and p for s, p in inside),
                                     "inside_not_placed": sum(s >= INSIDE and not p for s, p in inside),
                                     "outside_placed": sum(s < INSIDE and p for s, p in inside),
                                     "outside_not_placed": sum(s < INSIDE and not p for s, p in inside)}}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("frames", "track"):
        s = sub.add_parser(name)
        s.add_argument("folder", type=Path)
        s.add_argument("--service", type=Path, required=True)
    a = p.parse_args()
    (score if a.command == "frames" else track)(a.folder, a.service.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
