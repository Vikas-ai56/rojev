# Jev v1: the interaction lane (spec §9)

Code: `rojev/system2/jev/` (`acts`, `chooser`, `interaction`, `task_choice`, `calibration`); owner support in
`rojev/owner/state_owner.py`; executor tools `executors/robocasa-pi052/{calibrate_jev.py, label_jev.py}`,
`harness/recording.py` `InteractionCases`. Line-level map: [`system2-map.md`](system2-map.md) §6.

## Roles (decisions 2026-10-21 / 22)
- S1 (XR-1 in v1) picks its own steps and moves; Jev answers only changes (user speech, a motion deviation).
- Interaction lane = `JevInteraction` over OneJev; planning lane = `NativePlanner` or a baseline. 
- Jev is stateless: holds, dialogue and progress come from the snapshot (the owner's record).

## Flow of one change
```
user speaks → owner: rev+1, user turn, ACK said, moving command under review until the interaction deadline
            → JevInteraction.decide(snapshot)
                 question 1 (act):  legal acts in code → OneJev Choice → calibration → sum per effect → act | abstain
                 question 2 (task, revise only): trained sentences + none → OneJev Choice → calibration → sentence | abstain
            → Operation (+ task text = the chosen sentence, never the user's words)
            → owner: command (+ standing hold for ABORT / WAIT / ASK) (+ utterance if templated)
```

## Question one: the act (`interaction.py`, `acts.py`)
| Step | Rule | Model |
|---|---|---|
| wait budget | trigger `hold` (not an `UNANSWERED` hold): a stop keeps holding; else report `waiting_for_user` once, then keep holding (D7) | no |
| dwell | a deviation < `min_dwell_s` (1 s) after the step started → CONTINUE | no |
| candidates | legal acts for the situation (table), one per distinct effect | — |
| score | `Chooser.choose(snapshot, options, ACT_QUESTION)` → `Calibration.apply` (temperature) | one Choice |
| abstain | probabilities summed per effect; top-two effect margin < `margin` → ask if allowed, else wait | — |
| pick | the most probable act with the winning effect | — |
| wait while holding | active operation not moving: on an instruction → GUARDED_WAIT `waiting_for_user` (said); else CONTINUE | — |
| act → op | continue → CONTINUE · resume → the interrupted operation · wait → GUARDED_WAIT · ask → ASK(`clarify`) · abort → ABORT · revise → question two | — |

| Situation | Acts (effect) |
|---|---|
| moving | continue (move) · revise (switch) · wait · ask · abort (hold) |
| idle | continue · revise · ask · abort |
| held | resume (move; only if a step was interrupted) · continue = keep holding (hold) · revise · ask · abort (not if already stopped) |
| deviation | continue = motion fine (move) · wait = hold until the user decides (hold) |

- Effects (`acts.effect`): move · hold · switch. Acts with one effect differ only later (stop vs wait on resume), so
  Jev never hesitates between them (2026-10-24, E27 "stop" split abort 0.49 / wait 0.41 and asked).
- No retry (2026-10-24): XR-1 takes only a task sentence, so a retry is the same sentence = continue.
- Wording: options say what the user wants; keep-the-task options name the task; revise = a *complete* new task (what
  and with which objects); ask = unclear, or a change without the full task.
- Budgets (`may_ask`): ≤ `max_asks` (2) questions in the dialogue, never two in a row.
- Model inputs (`chooser.situation`): task in force, `user_just_said`, trigger, current step, steps done, conversation
  (robot turns rendered from `TEMPLATES`), hold, motion check; every camera its own image in one request; neutral
  option ids `o1, o2, …`, never act names. Client: `onejev.KeptClient` (one kept connection, E29).

## Question two: which trained task (`task_choice.CanonicalTask`)
- Options: the trained task sentences (the run's instruction + `--skills FILE`, verbatim) + `none`; `TASK_QUESTION`;
  own calibration.
- `none` wins or margin < `margin` → no revision → ask if allowed, else wait. Else SWITCH_TO(None) with the sentence
  as the new task text (`Verdict.task`).
- The user's words are evidence only, never S1's prompt (`S1Replans` removed 2026-10-24). A task off the list cannot
  start: the list is the repertoire (v1: trained tasks only).
- Evidence keeps both questions' options, raw and calibrated scores, margins, effects (`act` top level, `task` nested).

## Calibration (`calibration.py`, `calibrate_jev.py`)
- `Calibration{temperature, margin, fitted, cases, target_accuracy, coverage}`; only `apply` runs live. One per
  question (`act`, `task`); `realtime.py --jev-calibration C.json` loads `{act, task}`, a missing question keeps defaults.
- Cases: `realtime.py --interaction jev --record-interaction DIR` saves each question asked (cameras, situation,
  question, options, raw scores, effects) + a `labels.jsonl` line with `"label": null`; `label_jev.py DIR…` labels them
  in the terminal.
- `calibrate_jev.py --question act|task --fit DIR… --held-out DIR… --service S.json --out C.json
  [--target-accuracy 0.9]`: re-scores every labelled case with OneJev; T by NLL over the options; margin = smallest
  whose decided cases reach the target, over effects for the act question; written only if the held-out decided
  accuracy reaches the target (else exit 1, nothing written). Fit and held-out folders must differ (split by run / scene).
- **Now: T = 1, margin 0.2 for both questions, declared.** The 2026-10-23 fit (T 0.441, margin 0.103, 30 cases, one
  scene, old act set, no held-out set) is withdrawn (E26).

## Owner support ([`system2-map.md`](system2-map.md) §3)
- A hold answering a change stands until a later change releases it; `hold_recheck_s` asks Jev again (wait budget).
- Unanswered change (failure, timeout, refused answer) → GUARDED_WAIT `UNANSWERED`, said; never handed to planning.
- After a timeout the answer is awaited and applied as decided (`_as_decided`; CONTINUE re-issues the operation then in
  force); after a failure or refusal Jev decides again on the recheck.
- Dialogue: last 8 turns per snapshot; ACK at once; templated operations said, a repeated one said again.

## End-state checks (opt-in, off in v1)
- `StepWatcher` + `OneJevVerifier` / `MJevVerifier` (`--verify`); 2 checks with P(done) ≥ 0.7 → one `StepEvent`.
  Done-check A/B: neither model tells a finished step (E17). Details: [`system2-map.md`](system2-map.md) §7.

## Executor
- Flags: `--interaction jev --chooser onejev|keywords --reviser canonical|rewrite --skills FILE --max-asks N
  --jev-calibration FILE --hold-recheck S --record-interaction DIR` ([`live-harness.md`](../utils/live-harness.md)).
- Service JSON: `onejev: {url}` (qev over the tunnel). Spoofs (`harness/deciders.py`): `KeywordChooser` (scores; a
  deviation reads as wait), `RewriteReviser` (the scripted S1 needs step text).

## Not validated / open
- A fitted calibration on held-out runs (needs labelled runs in ≥ 2 scenes).
- Completion signal (resume after success), capability signal (is a listed task feasible now), an act for status
  questions, ASK with object ids (needs perception), OneJev server keep-alive.
