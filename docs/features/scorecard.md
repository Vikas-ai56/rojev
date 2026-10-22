# Revision-benchmark scorecard (T8, 2026-10-22)

Code: `rojev/eval/scorecard.py` (`Injection`, `Case`, `RunRecords`, `score_run`, `aggregate`, `render_markdown`, `render_json`). Tests: `tests/test_eval_scorecard.py` (27, synthetic records). Spec: `docs/plan/v3-v1-spec.md` §11 T8.

## Inputs
- One run = `Case(name, arm, injections)` + `RunRecords(summary, events, ticks, decisions)`: the dicts of `summary.json`, `owner.jsonl`, `ticks.jsonl`, `decisions.jsonl` (`docs/features/live-harness.md`), loaded by the caller.
- `Injection(t, kind, expected, goal, old_goal, phase)`: kind ∈ goal_swap · object_swap · stop · ambiguous · harmless · deviation; expected act ∈ continue · revise · wait · ask · abort; `goal` / `old_goal` = text naming the goal after / before a swap as it appears in subtask text (given together); `phase` = injection point label.
- Robot-agnostic: no file IO, no executor or robot imports (`eval` → `protocol`, `system2` only).

## Window and answer
- Window of an injection: change time → next injection or run end.
- Change time: speech = the owner's first `instruction` event at or after `t`; deviation = `t` (fault onset: detection time counts).
- Applied answer: first owner `command` / `reaffirm` / `continue` in the window whose `trigger` is the change's (`instruction` / `deviation`). Heartbeat decisions after the change are not answers.

## Metrics (per injection unless noted)
| Metric | Definition |
|---|---|
| success (run) | summary `task_success`; None if absent |
| answer latency | applied answer t − change time |
| execution latency | first tick under the answer's new command − change time; None for `continue` (no new command) |
| act | `continue` / `reaffirm` → continue · switch_to / retry → revise · guarded_wait → wait · ask · abort · other ops keep their name (never correct) |
| act correct | act == expected; no answer → False, or None if < `answer_window_s` (2 s) of the window remained |
| unanswered | owner `unanswered` events in the window (its fallback hold scores as a wait) |
| wrong-goal commitments | moving `command` / `reaffirm` in the window whose subtask contains `old_goal` and not `goal` |
| wasted motion | chunk ticks in the window under a command naming `old_goal` and not `goal` (incl. the one running at the change); ticks and s (÷ `control_hz`) |
| pause | hold ticks in [change, change + `pause_window_s` (3 s)) ∩ window, and their share of the ticks there |
| ask precision (arm) | asks in windows expecting ask ÷ all asks (asks outside any window are misses); n = asks |
| act ECE (arm) | first scored interaction-lane decision per window (`scores`: label → calibrated p, sum 1); class p = sum of its labels (revise, retry, resume → revise); confidence = top class p; correct = top class == expected; 10 equal-width bins, pooled over the arm's runs |
- Aggregate per arm (first-seen order): runs, changes reached, missed (judged, no answer), each metric as `Stat(value, n)` = mean over the values it could judge.
- Every None carries its reason in `InjectionScore.notes`.

## Why ticks, not path length, for wasted motion
- Ticks carry commanded actions (OSC deltas on PandaOmron), not measured poses; which dims are arm motion is the robot's `ActionSpec`, which `eval` may not import. A path length would be a commanded-delta sum standing in for motion.

## Not validated / open
- No live scored run yet; checked by hand on `results/scenarios/20261022-155022` (11 changes, scripted S1).
- `decisions.jsonl` has no `scores` today (`RecordingDecider` drops `Verdict.scores`): act ECE is None until it records them.
- Arms whose S1 plans its own subtasks (native, Jev with `S1Replans`) record no subtask text → goal metrics None; needs the planner's subtask in the records.
- Success is the simulator's check of the starting task: after a goal swap it measures the old task.
- A resume of the interrupted step is recorded as switch_to → scored revise (records do not carry the act label).
- Goal matching is containment of the case author's text; the text must be unambiguous within the task's subtasks.
