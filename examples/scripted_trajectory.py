"""Fixed joint-space trajectory for the SO-101, driven through the robot interface (`rojev.robot`), traced to Phoenix.

No teleop and no model control: the arm follows WAYPOINTS at the robot's control rate, every command passes the
safety filter, and the same loop would drive the real arm. With --onejev, the overhead camera frame at each checkpoint
is sent to OneJev for observation only; a separate evaluator checks the answers against simulator ground truth.

  uv run python examples/scripted_trajectory.py                    # headless
  uv run mjpython examples/scripted_trajectory.py --viewer         # watch it (macOS needs mjpython for the viewer)
  uv run python examples/scripted_trajectory.py --onejev           # also query OneJev (needs `make qev`)
  uv run python examples/scripted_trajectory.py --onejev --hide-box --instruction "Describe the scene"
"""
import argparse
import json
import math
import time
import uuid

import mujoco.viewer
import numpy as np
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from qev import Choice, Client, Noul

from rojev import onejev, tracing
from rojev.robot import CameraSpec, Rate, so101

HOME = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 0.0}
# ctrl of the "pickup" keyframe in Menagerie's robotstudio_so101/scene_box.xml
PICKUP = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.4732, "wrist_flex": 1.17718, "wrist_roll": 1.58438,
          "gripper": 0.727665}
# (name, {joint: target}, move seconds, checkpoint). Unlisted joints keep their previous target.
WAYPOINTS = [
    ("home", HOME, 1.0, True),
    ("pan_left", {"shoulder_pan": 0.8}, 1.5, False),
    ("pan_right", {"shoulder_pan": -0.8}, 2.5, False),
    ("pickup", PICKUP, 2.0, True),
    ("gripper_to_0", {"gripper": 0.0}, 1.0, False),
    ("gripper_back", {"gripper": PICKUP["gripper"]}, 1.0, False),
    ("home_again", HOME, 2.0, True),
]
HOLD_S = 0.5  # settle time after each move, included in the final-error measurement
JOINT_TOL_RAD = 0.01  # a segment fails if any joint ends further than this from its target
CAMERA = "overhead"
QUESTIONS = {
    "box_visible": Noul("A green box is visible in the image"),
    "next": Choice("Which action is most appropriate for the visible scene?", {
        "approach_box": "Move the gripper towards the visible box",
        "observe": "Observe again because evidence is inadequate",
        "stop": "Do not move"}),
}


def fail(span, problems: list[str]) -> None:
    """Mark a span ERROR with every problem, or OK."""
    span.set_status(Status(StatusCode.ERROR, "; ".join(problems)) if problems else Status(StatusCode.OK))


def report(problems: list[str]) -> str:
    return ("  FAIL: " + "; ".join(problems)) if problems else ""


def evaluate(robot, r, tracer, run_id: str, label: str) -> list[str]:
    """Separate evaluator: checks OneJev's answers against simulator ground truth, which is never sent to the model.

    Ground truth is the box's pixel count in a segmentation render from the camera OneJev saw. Rules:
    `box_visible` must be on the right side of 0.5; `next` must be `approach_box` if and only if the box is visible.
    """
    with tracer.start_as_current_span(f"eval:{label}") as span:
        span.set_attribute("openinference.span.kind", "EVALUATOR")
        span.set_attribute("session.id", run_id)
        px = robot.ground_truth.visible_pixels("box", CAMERA)
        visible, p_vis, pick = px > 0, r.answers["box_visible"].noul, r.answers["next"].raw["choice"]
        problems = []
        if (p_vis > 0.5) != visible:
            problems.append(f"box_visible={p_vis:.3f} but ground truth box pixels={px}")
        if (pick == "approach_box") != visible:
            problems.append(f"next={pick} but box {'visible' if visible else 'not visible'} ({px} px)")
        span.set_attribute("input.value", json.dumps({"box_pixels": px, "box_visible": visible}))
        span.set_attribute("output.value", json.dumps({"p_box_visible": p_vis, "next": pick, "problems": problems}))
        span.set_attribute("eval.passed", not problems)
        fail(span, problems)
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--viewer", action="store_true", help="open the passive MuJoCo viewer (run under mjpython)")
    p.add_argument("--onejev", action="store_true", help="ask OneJev about each checkpoint frame (observe only)")
    p.add_argument("--hide-box", action="store_true", help="make the green box invisible (physics unchanged)")
    p.add_argument("--instruction", default="Approach the visible green box")
    p.add_argument("--qev-url", default="http://localhost:8000")
    p.add_argument("--project", default="sim-demo", help="Phoenix project name")
    args = p.parse_args()

    provider = tracing.setup(args.project)
    tracer = trace.get_tracer("examples.scripted_trajectory")
    robot = so101.simulated(cameras=(CameraSpec(CAMERA),), realtime=args.viewer)
    if args.hide_box:  # scenario setup on the simulated world, not something the pipeline can do
        robot.model.geom_rgba[robot.model.geom("box").id, 3] = 0.0
    names = robot.joint_names
    client = Client(args.qev_url) if args.onejev else None
    viewer = mujoco.viewer.launch_passive(robot.model, robot.data) if args.viewer else None

    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    print(f"run {run_id}  (Phoenix project '{args.project}', session {run_id}, {robot.control_rate_hz:g} Hz control)")
    with robot, tracer.start_as_current_span("episode") as ep:
        if viewer is not None:
            robot.attach_viewer(viewer)
        ep.set_attribute("openinference.span.kind", "CHAIN")
        ep.set_attribute("session.id", run_id)
        ep.set_attribute("input.value", json.dumps({"scene": "pick_place.xml", "waypoints": [w[0] for w in WAYPOINTS],
                                                    "hide_box": args.hide_box, "instruction": args.instruction,
                                                    "control_rate_hz": robot.control_rate_hz}))
        rate, obs = Rate(robot.clock, robot.control_rate_hz), robot.read()
        target, wall_t0 = obs.joints.position.copy(), time.perf_counter()
        worst, failures, paused = 0.0, [], 0.0
        for name, spec, move_s, checkpoint in WAYPOINTS:
            start = target.copy()
            target = start.copy()
            for joint, value in spec.items():
                target[names.index(joint)] = value
            with tracer.start_as_current_span(f"segment:{name}") as seg:
                seg.set_attribute("openinference.span.kind", "TOOL")
                seg.set_attribute("session.id", run_id)
                seg.set_attribute("input.value", json.dumps(dict(zip(names, target.round(4).tolist()))))
                seg.set_attribute("input.mime_type", "application/json")
                problems = robot.limits.position_violations(target)
                peak, floor_hits, clamps = 0.0, 0, []
                if not problems:  # never send a waypoint that is outside the joint limits
                    n_move, n_hold = round(move_s * robot.control_rate_hz), round(HOLD_S * robot.control_rate_hz)
                    for k in range(n_move + n_hold):
                        s = min(1.0, (k + 1) / n_move)
                        cmd = start + (target - start) * (1 - math.cos(math.pi * s)) / 2
                        clamps += robot.write(robot.command(cmd)).violations
                        rate.sleep()
                        obs = robot.read()
                        peak = max(peak, float(np.abs(obs.joints.position - cmd).max()))
                        floor_hits = max(floor_hits, robot.ground_truth.contacts_with("floor", ignore_bodies=("box",)))
                q = obs.joints.position
                err = np.abs(q - target)
                worst = max(worst, float(err.max()))
                if err.max() > JOINT_TOL_RAD:
                    problems.append(f"final joint error {err.max():.4f} rad on {names[int(err.argmax())]} > {JOINT_TOL_RAD}")
                if floor_hits:
                    problems.append(f"{floor_hits} arm-floor contacts")
                if clamps:
                    problems.append(f"safety filter clamped {len(clamps)} commands, first: {clamps[0]}")
                result = {"reached": dict(zip(names, q.round(4).tolist())),
                          "final_error_rad": dict(zip(names, err.round(4).tolist())),
                          "max_final_error_rad": round(float(err.max()), 4), "peak_tracking_error_rad": round(peak, 4),
                          "robot_floor_contacts_max": floor_hits, "safety_clamps": len(clamps),
                          "sim_time_s": round(robot.clock.now(), 3)}
                seg.set_attribute("output.value", json.dumps(result))
                seg.set_attribute("output.mime_type", "application/json")
                seg.set_attribute("sim.max_final_error_rad", result["max_final_error_rad"])
                seg.set_attribute("sim.robot_floor_contacts_max", floor_hits)
                seg.set_attribute("sim.safety_clamps", len(clamps))
                fail(seg, problems)
                failures += problems
                print(f"  {name:13s} final err max {err.max():.4f} rad ({names[int(err.argmax())]})  peak {peak:.3f}  "
                      f"floor contacts {floor_hits}  clamps {len(clamps)}  t={robot.clock.now():.2f}s" + report(problems))
            if checkpoint and client is not None:
                t_pause = time.perf_counter()
                with robot.clock.paused(), tracer.start_as_current_span(f"checkpoint:{name}") as cp:
                    cp.set_attribute("openinference.span.kind", "CHAIN")
                    cp.set_attribute("session.id", run_id)
                    frame = obs.cameras[CAMERA]
                    cp.set_attribute("camera.frame", json.dumps({"camera": CAMERA, "seq": frame.stamp.seq,
                                                                 "acquired": frame.stamp.acquired,
                                                                 "age_s": robot.clock.now() - frame.stamp.acquired}))
                    state = {"instruction": args.instruction, "screen": "<image:1>", "gripper": "unknown",
                             "previous_skill_result": "none"}
                    r = onejev.system_one(client, frame.rgb, state, QUESTIONS, span_name=f"onejev:{name}", session_id=run_id)
                    problems = evaluate(robot, r, tracer, run_id, name)
                    fail(cp, problems)
                    failures += problems
                    print(f"    OneJev @ {name}: P(box visible)={r.answers['box_visible'].noul:.3f}  "
                          f"next={ {k: round(v, 3) for k, v in r.answers['next'].probabilities.items()} }" + report(problems))
                paused += time.perf_counter() - t_pause
        wall = time.perf_counter() - wall_t0 - paused
        ep.set_attribute("output.value", json.dumps({"max_final_error_rad": round(worst, 4), "sim_time_s": round(robot.clock.now(), 3),
                                                     "motion_wall_time_s": round(wall, 3), "failures": failures}))
        fail(ep, failures)
    print(f"done: sim {robot.clock.now():.2f}s, motion wall time (excl. OneJev waits) {wall:.2f}s, "
          f"worst final error {worst:.4f} rad, {len(failures)} failure(s)")
    if viewer is not None:
        viewer.close()
    provider.shutdown()


if __name__ == "__main__":
    main()
