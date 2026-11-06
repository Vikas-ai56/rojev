# Revision-benchmark scorecard (T8)

Code: `rojev/eval/scorecard.py` (`Injection`, `Case`, `RunRecords`, `score_run`, `aggregate`, `render_markdown`,
`render_json`). Tests: `tests/test_eval_scorecard.py` (28 passed 2026-10-24, synthetic records). Spec:
`docs/plan/v3-v1-spec.md` §11 T8. Fed by the scenario matrix ([`live-harness.md`](live-harness.md)); user campaigns
use their own per-line judge (`executors/robocasa-pi052/campaign.py`).

## Inputs
- One run = `Case(name, arm, injections)` + `RunRecords(summary, events, ticks, decisions)`: the dicts of
  `summary.json`, `owner.jsonl`, `ticks.jsonl`, `decisions.jsonl`, loaded by the caller.
- `Injection(t, kind, expected, goal, old_goal, phase)`: kind ∈ goal_swap · object_swap · stop · pause · resume ·
  ambiguous · harmless · deviation; expected ∈ continue · revise · wait · ask · abort; `goal` / `old_goal` = text naming
  the goal after / before a swap as subtask text spells it (given together); `phase` = injection-point label.
- Robot-agnostic: no file IO, no executor or robot imports (`eval` → `protocol`, `system2` only).

## Window and applied answer
- Window: change time → next injection or run end. Change time: speech = the owner's first `instruction` event at or
  after `t`; deviation = `t` (fault onset, so detection time counts).
- Answers = owner `command` / `reaffirm` / `continue` events in the window with the change's trigger (`instruction` /
  `deviation`). Heartbeat decisions are never answers.
- Applied answer = the first answer a decider made. A late decider answer (after the owner's `UNANSWERED` hold, applied
  as decided since 2026-10-24) is scored as the answer, noted "answered late".
- No decider answer, only the owner's safety hold after an `unanswered` event → act `unanswered`, never correct (a safe
  fallback is not a right answer).
- No answer at all → incorrect, or not judged if < `answer_window_s` (2 s) of the window remained.

## Metrics (per injection unless noted)
| Metric | Definition |
|---|---|
| success (run) | summary `task_success` (the task the run started with); None if absent |
| answer latency | applied answer t − change time (robot clock, one tick resolution) |
| execution latency | first tick under the answer's new command − change time; None for `continue` |
| act | `continue` / `reaffirm` → continue · switch_to / retry → revise (a resume is a switch_to) · guarded_wait → wait · ask · abort · any other op keeps its name (never correct) |
| act correct | act == expected |
| unanswered | owner `unanswered` events in the window with the change's trigger |
| wrong-goal commitments | moving `command` / `reaffirm` in the window whose subtask contains `old_goal` and not `goal` |
| wasted motion | chunk ticks in the window under a command naming `old_goal` and not `goal` (incl. the one running at the change); ticks and s (÷ `control_hz`) |
| pause | hold ticks in [change, change + `pause_window_s` (3 s)) ∩ window, and their share |
| asks | `ask` commands in the window; ask precision (arm) = asks in windows expecting ask ÷ all asks of the run |
| act ECE (arm) | first interaction-lane decision in the window with the change's trigger and `scores` (calibrated p, sum 1); labels → classes as their ops (resume → revise; `RECORDED_RETRY` "retry" in runs before 2026-10-24 → revise); confidence = top class p; 10 equal-width bins, pooled per arm |

- Aggregate per arm (first-seen order): runs, changes reached, missed (judged, no answer); each metric a
  `Stat(value, n)` = mean over the values it could judge; pause split into `pause when moving` (expected continue /
  revise: any pause is a freeze) and `hold when asked` (expected wait / ask / abort); unanswered rate.
- Every None carries its reason in `InjectionScore.notes`.
- Wasted motion counts ticks, not path length: ticks carry commanded OSC deltas, not measured poses, and which dims are
  arm motion is the robot's `ActionSpec`, which `eval` may not import.

## Matrix feed
- `executors/robocasa-pi052/scenarios.py`: each case declares its changes; times come from its `--say` / `--block-at`;
  arm = `--interaction` (else `--arm`) + `+revoke` / `+s2-slow` / `+s2-fail`. Declarations:
  [`live-harness.md`](live-harness.md) matrix table.
- Scored only if the run's summary schedule equals the case's and its frames are valid; `scorecard.md` /
  `scorecard.json` beside `report.json`; `--score-only DIR` re-scores a past matrix.

## Measured (`results/scenarios/20261024-125433/scorecard.md`)
- 13 cases, 18 changes; scripted S1 and `rules` / keyword-scored `jev` deciders, so acts measure the plumbing, not Jev.
- `rules` 9/9 and `jev` 5/5 acts correct; `jev` act ECE 0.180 (n 5, of keyword scores).
- `revisable` (one lane): "wait" answered continue by the planning lane (0/1, the baseline's design).
- `rules+s2-slow`: wait answered late at 1.6 s, scored correct with 1 unanswered; `rules+s2-fail`: `unanswered`.

## Limits / open
- No case declares a `phase`: the matrix injects by time, not at pre-grasp / contact / transport / pre-commit.
- A resume is logged as switch_to, so it scores as revise (records carry no act label).
- An S1 that plans its own subtasks (native, Jev's `CanonicalTask`) records no subtask text → goal metrics None.
- Success checks the starting task: after a goal swap it measures the old task.
- Goal matching is containment of the case author's text; it must be unambiguous within the task's subtasks.
- Model runs (XR-1 + Jev/OneJev) have not been scored here; they are judged per line by the campaign.
