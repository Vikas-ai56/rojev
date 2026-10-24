# Revision-benchmark scorecard (T8, 2026-10-22)

Code: `rojev/eval/scorecard.py` (`Injection`, `Case`, `RunRecords`, `score_run`, `aggregate`, `render_markdown`, `render_json`). Tests: `tests/test_eval_scorecard.py` (27, synthetic records). Spec: `docs/plan/v3-v1-spec.md` §11 T8.

## Inputs
- One run = `Case(name, arm, injections)` + `RunRecords(summary, events, ticks, decisions)`: the dicts of `summary.json`, `owner.jsonl`, `ticks.jsonl`, `decisions.jsonl` (`docs/features/live-harness.md`), loaded by the caller.
- `Injection(t, kind, expected, goal, old_goal, phase)`: kind ∈ goal_swap · object_swap · stop · pause · resume (release a standing hold) · ambiguous · harmless · deviation; expected act ∈ continue · revise · wait · ask · abort; `goal` / `old_goal` = text naming the goal after / before a swap as it appears in subtask text (given together); `phase` = injection point label.
- Robot-agnostic: no file IO, no executor or robot imports (`eval` → `protocol`, `system2` only).

## Fed by the scenario matrix (2026-10-22)
- `executors/robocasa-pi052/scenarios.py`: each case declares its changes (`Scenario.changes`); times come from its `--say` / `--block-at`; arm = its `--interaction`, else `--arm`. Table of declarations: [`live-harness.md`](live-harness.md) § Scorecard.
- The matrix loads each run with `harness/invariants.py` `Run.load` → `RunRecords`, scores it, writes `scorecard.md` / `scorecard.json` beside `report.json`; `--score-only RUN_DIR` re-scores a past matrix.
- A run is scored only if its summary `say` / `blocked` equals the case's schedule.

## Window and answer
- Window of an injection: change time → next injection or run end.
- Change time: speech = the owner's first `instruction` event at or after `t`; deviation = `t` (fault onset: detection time counts).
- Applied answer: first owner `command` / `reaffirm` / `continue` in the window whose `trigger` is the change's (`instruction` / `deviation`) and that a decider made; the owner's unanswered hold counts only if no decider's answer follows it (a late answer is applied since 2026-10-24, noted). Heartbeat decisions after the change are not answers.

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
- Scored by the matrix (`--score-only`) on `results/scenarios/20261022-155022` (14 cases, 19 changes) and `20261022-162842` (9 cases, 12 changes); scripted S1, rules / jev deciders with keyword spoofs, so the acts measure the plumbing, not Jev.
- "wait" is kind `pause`, "go on" after it kind `resume`; a resume is logged as switch_to, so its expected act is revise (the records carry no act label to tell resume from revise).
- No case declares a `phase`: the matrix injects by time, not at pre-grasp / contact / transport / pre-commit.
- Arm pools faults and baselines with the same decider (`rules` = harmless, revise, stop, s2-slow, s2-fail, revoke-baseline, blocked, …).
- `decisions.jsonl` has no `scores` today (`RecordingDecider` drops `Verdict.scores`): act ECE is None until it records them.
- Arms whose S1 plans its own subtasks (native, Jev with `CanonicalTask`) record no subtask text → goal metrics None; needs the planner's subtask in the records.
- Success is the simulator's check of the starting task: after a goal swap it measures the old task.
- A resume of the interrupted step is recorded as switch_to → scored revise (records do not carry the act label).
- Goal matching is containment of the case author's text; the text must be unambiguous within the task's subtasks.

## Scoring rules added 2026-10-22
- The owner's safety hold for an unanswered change is no decision: act `unanswered`, never correct; per-arm `unanswered` rate.
- Pause split: `pause when moving` (changes expecting continue / revise: any pause is a freeze) and `hold when asked` (changes expecting wait / ask / abort).
- Arms carry conditions: `rules`, `rules+revoke`, `rules+s2-slow`, `rules+s2-fail`, `jev`, `revisable` (one lane).
- Answer latency is on the robot clock (one control tick of resolution).
