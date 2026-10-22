# Observability (Phoenix): runtime run export + scripted sim demo

Code: `rojev/observability/tracing.py` (`setup`), `rojev/observability/run_trace.py` (`export_run`), `rojev/system2/models/onejev.py`, `examples/scripted_trajectory.py`, `executors/robocasa-pi052/harness/recording.py` (per-call records).

## Run it (one terminal each)

| Command | What |
|---|---|
| `make phoenix` | Phoenix UI + trace collector, http://localhost:6006. Traces persist in `data/phoenix/` (gitignored) |
| `make qev` | OneJev 1.5 0.8B selector, http://127.0.0.1:8000 |
| `make sim` / `make sim-viewer` | Fixed joint trajectory (no teleop, no model control), headless / in the MuJoCo viewer |
| `make sim ARGS="--onejev"` | Also ask OneJev about the frame at each checkpoint (observe only) |
| `make sim ARGS="--onejev --hide-box --instruction '…'"` | Same, box invisible / custom instruction |

## Runtime runs: post-run export (`run_trace.export_run`)

- `realtime.py --trace PROJECT` / `scenarios.py --trace PROJECT` (each run); start `make phoenix` first. Exported after the run from plain records: nothing is traced on the control path.
- Inputs (dicts, any executor): owner events (`OwnerRecord.events`), S1 requests (`RecordingPolicy.log`), commitments (`RecordingCommit.log`), S2 decisions (`RecordingDecider.log`), run config and summary. Times = robot clock from the run start; `t0_ns` (wall time at start) maps them.

```
run <out dir name>             config.* + summary.* attributes
├── command <id> <op> <subtask>  issue → next command (or run end); events: ack <status>, review, keyframe, deviation, stale deviation
├── s1.request                   policy call (start → return): subtask, samples, instruction, info; commit.* (floor, steps, reason, chosen, detail); ERROR status on a policy error
├── s2.<lane>.decide             request → proposal; outcome applied: command / reaffirm / continue, refused + why, or not delivered; ERROR on decider error
└── user                         each user instruction (zero length; text, instruction_rev)
```

- Commitments are matched to successful requests by order (one `plan` per served request). Non-scalar attributes are JSON strings; `None` values dropped.
- Not tested: no unit test covers `run_trace`.

## Scripted sim demo: what lands in Phoenix (project `sim-demo`, one session per run)

```
episode (CHAIN)                    status ERROR if anything below failed, reasons listed
├─ segment:<name> (TOOL)           motion only: targets, joints reached, per-joint error, peak error, arm–floor contacts
└─ checkpoint:<name> (CHAIN)       after a segment; physics paused (excluded from viewer pacing)
   ├─ onejev:<name> (LLM)          exact image sent, state, questions, answers, debug logits; `onejev.<q>[.<opt>]`
   └─ eval:<name> (EVALUATOR)      simulator ground truth vs the answers; status ERROR + reason on mismatch
```

- A segment fails if any joint ends more than `JOINT_TOL_RAD` = 0.01 rad from its target, or the arm touches the floor.
- Evaluator ground truth is the box's pixel count in a segmentation render from the same camera (`rojev.robot.sim.visible_pixels`). It is never sent to OneJev. Rules: `box_visible` must be on the correct side of 0.5; `next` must be `approach_box` if and only if the box is visible.
- Failures are also printed as `FAIL: <reason>`. In Phoenix, filter with `status_code == 'ERROR'`.
- Reading durations: a segment span in the viewer lasts as long as its real-time motion (move + 0.5 s hold). OneJev time is in the `onejev:*` span. Median OneJev call: 145 ms headless vs 311 ms with the viewer open. The viewer is untested but suspected of competing with llama.cpp for the GPU.

Trajectory: `home → pan ±0.8 → Menagerie "pickup" keyframe ctrl → gripper 0 → back → home`, cosine-interpolated and sent at 30 Hz through `rojev.robot` (safety filter on every command), with a 0.5 s hold each. Waypoints outside the joint limits are refused. OneJev sees the fixed `overhead` sensor camera in `scenes/pick_place.xml`. The measurements below were taken on the earlier free-camera view of `scene_box.xml`.

## Measured (2026-10-16)

- Tracking: final error ≤ 0.0006 rad on every segment; 0 arm–floor contacts; the viewer runs in real time (14.50 s sim = 14.50 s loop wall time, recorded as `motion_wall_time_s` on the episode span, excluding OneJev waits; the episode span duration is the true total).
- OneJev, same trajectory, box hidden, instruction varied (one deterministic sample per checkpoint):

  | Checkpoint | Instruction "Approach the visible green box": P(box) / P(approach) | "Decide the next action from what the image shows": P(box) / P(approach) |
  |---|---|---|
  | home | 0.138 / 0.569 | 0.003 / 0.309 |
  | pickup | 0.064 / 0.228 | 0.001 / 0.058 |
  | home_again | 0.118 / 0.502 | 0.002 / 0.289 |

  With the box visible: P(box) 0.980–0.983, P(approach) 0.815–0.868.
  The instruction raises **both** the action and the visibility answer. A visibility check asked in the same state as the instruction is therefore biased too. This is small-sample evidence, not a benchmark.

## Gotchas

- OpenInference replaces base64 images longer than 32k chars with `__REDACTED__`. `rojev/observability/tracing.py` raises the limit to 1M chars.
- The UI binds to 127.0.0.1 (`PHOENIX_HOST` in the Makefile). Phoenix's gRPC collector (:4317) is hardcoded to bind all interfaces (`phoenix/server/grpc_server.py:109`). Other machines on the network can push spans to it but cannot read them.
- The Phoenix server (`arize-phoenix`) is in the uv `dev` group. The SDK itself depends only on `arize-phoenix-otel` and `openinference-instrumentation`. Phoenix pulls `websockets` down from 17.2 to 16.1.1; G0 was re-run and still passes.
