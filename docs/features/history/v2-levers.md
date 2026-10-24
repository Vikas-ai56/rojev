# v2 levers registry (historical)

Historical (v2 design, superseded 2026-10-21 by the v3 runtime and v1 = Route A; see [`../levers.md`](../levers.md) for the current knobs and [`../v3-runtime.md`](../v3-runtime.md)).

## What it was
- One table of every value that shaped the v2 output: chunk-synchronous loop, one decision per 50-step chunk (2.5 s), sim paused while models answered.
- Pipeline: MJev-Spatial evidence (done / possible / blocked per candidate, 3 cameras at 256²) → candidate filter over π0.5²'s 21,326 training subtasks (`rojev/legacy_v2/candidates.py`, E8–E9) → one OneJev Choice (continue / K candidates / redo / done) → π0.5² executes the chosen subtask.
- Arms: A π0.5²'s own planner · B fixed sequence · C generative VLM · D OneJev · E oracle.

## What replaced it
- Jev as next-step selector measured 40% (E13); decision 2026-10-21: π0.5² (now XR-1) plans its own steps, Jev answers only changes ([`../jev-interaction.md`](../jev-interaction.md)).
- Per-chunk decisions → asynchronous runtime with leases, review, commitment and two decision lanes ([`../v3-runtime.md`](../v3-runtime.md)).
- Code left: `rojev/legacy_v2/` (quarantined), `executors/robocasa-pi052/workflow.py`.

## §7 Simulation and evaluation (still read by `tasks.py`)
- `max_steps` per authored task ≈ 1.5 × p90 training-demo length: 2320 / 1520 / 2370 (MicrowaveThawingFridge / MoveFridgeToFreezer / LoadCondimentsInFridge); used by `workflow.py` only.
- Arm B budget `chunks_per_subtask` = p75 demo subtask duration ÷ 50-step chunks: 5 / 5 / 4. Still used by `realtime.py --arm scripted|revisable` ([`../levers.md`](../levers.md)).
- Subtask wording = most frequent training phrasing (E8); ground truth = RoboCasa's own success primitives (`tasks.py`).

## Lessons that still apply
- Attribute a failure to one stage before changing anything: wrong evidence → candidate missing → wrong choice → failed execution (an oracle arm isolates the executor).
- Change a config value before swapping a model, a model before training; training is for wrong knowledge, not a wrong setting.
- Ask evidence questions without the instruction in the state (E2: the instruction biased what the model "saw").
- Shuffle option order per query and log it (position bias).
