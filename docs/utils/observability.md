# Observability (Phoenix): post-run export of runtime runs + the SO-101 scripted demo

Code: `rojev/observability/tracing.py` (`setup`), `rojev/observability/run_trace.py` (`export_run`),
`rojev/system2/models/onejev.py` (`system_one`: one LLM span per OneJev call), `executors/robocasa-pi052/harness/
recording.py` (per-call records), `examples/scripted_trajectory.py` (demo). Decision: Phoenix (local) via
OpenTelemetry (decisions 2026-10-16).

## Run it
| Command | What |
|---|---|
| `make phoenix` | Phoenix UI + collector at http://localhost:6006; traces persist in `data/phoenix/` (gitignored) |
| `realtime.py --trace PROJECT` / `scenarios.py --trace PROJECT` | export each runtime run after it ends |
| `make sim` / `make sim-viewer` | SO-101 scripted trajectory, headless / in the MuJoCo viewer (project `sim-demo`) |
| `make qev` + `make sim ARGS="--onejev [--hide-box] [--instruction '…']"` | demo also asks a local OneJev about each checkpoint frame |

## Runtime runs: `run_trace.export_run`
- Exported after the run from plain records, so nothing is traced on the control path and any executor can export.
- Inputs: owner events (`OwnerRecord.events`), S1 requests (`RecordingPolicy.log`), commitments
  (`RecordingCommit.log`), S2 decisions of both lanes (`RecordingDecider.log`), run config and summary. Times are the
  robot clock from the run start; `t0_ns` (wall time at start) maps them.
```
run <out dir name>               config.* + summary.* attributes
├── command <id> <op> <subtask>  issue → next command (or run end); events: ack <status>, review, step done / undone,
│                                deviation, stale deviation
├── s1.request                   policy call: subtask, samples, instruction, continuation, info; commit.* (floor,
│                                steps, horizon, reason, chosen, detail); ERROR status on an error or timeout
├── s2.<lane>.decide             request → verdict; outcome: applied (command / reaffirm / continue), refused + why,
│                                or not delivered; ERROR on a decider error
└── user                         each user instruction (zero length; text, instruction_rev)
```
- Commitments are matched to successful requests by order (one `plan` per served request); non-scalar attributes are
  JSON strings; `None` values are dropped.
- Not exported as span events: owner `hold`, `unanswered`, `late answer`, `step event`, `say` (they stay in
  `owner.jsonl`).
- OneJev calls inside a live run produce no spans: `tracing.setup` registers the provider only after the run, so the
  `system_one` spans of the interaction lane are non-recording.
- Not tested: no unit test covers `run_trace`.

## SO-101 scripted demo (`examples/scripted_trajectory.py`; Phase-0 era)
```
episode (CHAIN)                    ERROR if anything below failed, reasons listed
├─ segment:<name> (TOOL)           targets, joints reached, per-joint error, peak error, arm–floor contacts
└─ checkpoint:<name> (CHAIN)       after a segment; physics paused
   ├─ onejev:<name> (LLM)          exact image sent, state, questions, answers, debug logits
   └─ eval:<name> (EVALUATOR)      simulator ground truth vs the answers; ERROR + reason on mismatch
```
- Trajectory: home → pan ±0.8 → Menagerie "pickup" keyframe → gripper 0 → back → home, cosine-interpolated at 30 Hz
  through `rojev.robot` (safety filter on every command), 0.5 s hold each.
- A segment fails if any joint ends more than 0.01 rad from its target or the arm touches the floor.
- Evaluator truth = the box's pixel count in a segmentation render from the same camera (never sent to OneJev):
  `box_visible` must sit on the correct side of 0.5; `next` must be `approach_box` iff the box is visible.
- `make qev` serves OneJev 1.5 0.8B (Makefile `GGUF`), not the 1.6 4B the v1 loop uses; demo answers are evidence
  for E1–E4 only, all superseded (`docs/paper/experiments.md`). The 1.6 4B GGUF (`mradermacher/OneJev1.6-4B-GGUF`)
  does not load on the local llama.cpp (build 11429: `tensor 'blk.32.attn_norm.weight' not found`, 2026-10-25); the
  v1 model runs only on the GPU service (bf16, `services/onejev`).
- Measured 2026-10-16: final error ≤ 0.0006 rad per segment, 0 floor contacts, viewer in real time.

## Gotchas
- OpenInference replaces base64 images longer than 32k chars with `__REDACTED__`; `tracing.MAX_IMAGE_CHARS` raises
  the limit to 1M.
- If Phoenix is not reachable at `PHOENIX_URL`, `setup` warns and spans are dropped.
- The UI binds to 127.0.0.1 (`PHOENIX_HOST` in the Makefile); Phoenix's gRPC collector (:4317) binds all interfaces
  (`phoenix/server/grpc_server.py:109`, `[::]`): other machines can push spans, not read them.
- The Phoenix server (`arize-phoenix`) is in the uv `dev` group; the SDK needs only `arize-phoenix-otel` and
  `openinference-instrumentation`.
