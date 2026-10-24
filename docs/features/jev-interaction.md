# Jev v1 interaction lane + end-state checks (spec §9, step 6)

Code: `rojev/system2/jev/` (acts, task_choice, calibration, chooser, interaction), `rojev/system2/verify/` (checks, watcher),
owner support in `rojev/owner/state_owner.py` (holds, dialogue, utterances, step events). Built 2026-10-22; live OneJev
/ MJev runs and the calibration fit need the GPU.

## Roles (decided 2026-10-21)
- π0.5² picks the next step and moves; Jev answers only changes; MJev checks end states.
- Interaction lane = `JevInteraction`; planning lane = π0.5²'s own planner or a baseline decider.

## Flow of one change
```
user speaks → owner: rev+1, user Turn, ACK utterance, moving command under review
            → interaction lane: JevInteraction.decide(snapshot)
                 question 1 (act): candidates in code → Chooser (OneJev Choice) → Calibration → margin / budgets
                 question 2 (task, revise only): trained task sentences + none → Chooser → Calibration → sentence | ask
                 → Operation (+ task text = the chosen sentence, never the user's words)
            → owner accepts → command (+ Hold if ABORT / WAIT / ASK) (+ utterance if templated)
```

## `JevInteraction` (stateless; everything comes from the snapshot)
| Step | Rule | Model call |
|---|---|---|
| `hold` trigger | stop → keep holding; else report once (`waiting_for_user`), then keep holding (D7) | no |
| dwell | deviation < `min_dwell_s` (1 s) after the step started → CONTINUE | no |
| candidates | legal acts per situation (table below; one act per distinct effect) | — |
| score | `Chooser.choose(snapshot, options, ACT_QUESTION)` → probabilities; `Calibration.apply` (temperature) | one OneJev Choice |
| abstain | probabilities summed per effect (`acts.effect`: move · hold · switch); top-two effect margin < `margin` (0.2) → ask if allowed, else wait; else the most probable act of the winning effect (2026-10-24: never hesitate between two acts that do the same thing) | — |
| revise | `Reviser.revise` → `Revision`; `CanonicalTask`: Choice over the trained task sentences + `none` (`TASK_QUESTION`, own calibration); none / low margin → ask if allowed, else wait | one more OneJev Choice |
| budgets | `max_asks` (2) questions in the dialogue; never two questions in a row | — |
| act → op | continue → CONTINUE · resume → interrupted operation · revise → the `Revision` · wait → GUARDED_WAIT · ask → ASK(`clarify`) · abort → ABORT | — |

| Situation | Legal acts |
|---|---|
| instruction, moving | continue · revise · wait · ask · abort |
| instruction, idle | continue · revise · ask · abort |
| instruction, held | resume (if a step was interrupted) · continue (keep holding) · revise · ask · abort (not if already stopped) |
| deviation | continue (motion fine) · wait (motion wrong: hold until the user decides) |

- No retry (2026-10-24): XR-1 takes only a task sentence, so a retry is the same sentence again (= continue).
- `Reviser`: `CanonicalTask(chooser, sentences, calibration)` (default) = SWITCH_TO(None) under the chosen trained task
  sentence; `sentences` come from the robot side (`realtime.py`: the run's instruction + `--skills FILE`). A task not
  on the list cannot be started: the list is the repertoire (v1 scope: trained tasks only). `S1Replans` (the user's
  raw words as the task) was removed 2026-10-24: remarks, speed requests and typos became S1's prompt.
- Wait while already holding → CONTINUE (keeps the hold, no new command).
- `OneJevChooser` state: task in force, what the user just said, trigger, current step, steps done (keyframes), conversation (robot turns rendered from `TEMPLATES`), hold, motion check; images = every camera as its own `<image:k>` in the one request (`onejev.camera_views`; glued side-by-side views gave the model no boundaries, changed 2026-10-22).
- Only REVISE changes the task text: the `Revision` carries the operation and the chosen sentence (`Verdict.task`);
  evidence keeps both questions' options and scores (`act` at top level, `task` nested).
- The model sees neutral option ids (`o1`, `o2`, …) and options worded as what the user wants; act names are never shown ("continue" in a held situation meant "keep holding" while the user said "go on").

## Calibration (D5)
- `Calibration{temperature, margin, fitted, cases, target_accuracy, coverage}`; offline only, `apply` at runtime.
- `fit_temperature` = NLL grid search; `fit_margin` = smallest margin whose decided cases reach `target_accuracy` (else 1.0, always abstain); `fit_calibration` = both + coverage.
- One calibration per question (`act`, `task`).
- Cases: `realtime.py --interaction jev --record-interaction DIR` saves each question asked (cameras, situation, `question`, options, raw scores) + a `labels.jsonl` line (`"label": null`) for a person to fill.
- `label_jev.py DIR…`: label cases in the terminal (cameras opened, options numbered, written to `labels.jsonl` at once).
- `calibrate_jev.py --question act|task --fit DIR… --held-out DIR… --service S.json --out C.json [--target-accuracy 0.9]`: fit on the fit set; written under its question only if the held-out decided accuracy reaches the target (else exit 1, nothing written). `--jev-calibration C.json` loads `{act, task}`; a missing question keeps the defaults.
- Fit and held-out sets: different runs and scenes, clean frames only (`frame_check` valid), recorded after the act set is frozen.
- **Today T = 1, margin 0.2: declared tuned constants.** The 2026-10-23 fit (T 0.441, margin 0.103, 30 cases) was on the old act set, one scene, no held-out set: withdrawn.

## Owner support
- Holds persist until a change resolves them (D4 fix); `hold_recheck_s` asks the interaction lane again (wait budget).
- An unanswered change (interaction failure, timeout, refused answer) holds the robot (`UNANSWERED`, said); the planning lane never answers a change, so an unread "stop" cannot move the robot.
- After a failure or refusal the change is decided again on the recheck. After a timeout the answer is awaited (no recheck): when it comes it is applied to the situation it was decided on (`_as_decided`: the placeholder hold is not a new base; CONTINUE re-issues the operation in force then). A newer interaction request ends the wait.
- Dialogue: last 8 turns in every snapshot; `Utterance` out through `say` (`ACK` at once, ≤ one owner tick; templated ops).
- `StepEvent` → planning lane (trigger `step`) + keyframe evidence.

## End-state checks (`system2/verify`)
- `Verifier.check(images, steps) → {step: {done, possible, blocked}}`; `MJevVerifier` = one MJev request, the arm-D question per step (no per-skill parser).
- `StepWatcher` thread: every `every_s` (1 s) checks the runtime's active command (if it moves with a subtask); `confirmations` (2) consecutive P(done) ≥ `done_threshold` (0.7) → one `StepEvent` per command; failures counted, never posted.
- Threshold and confirmations = declared tuned constants until T6 (MJev vs simulator truth).
- `RemoteMJev` uses `rojev.transport` (one kept connection).

## Executor (`executors/robocasa-pi052/realtime.py`)
- `--interaction jev --chooser onejev|keywords --reviser canonical|rewrite --skills FILE --max-asks N --jev-calibration FILE`
- `--hold-recheck S` · `--verify mjev|truth --verify-every --done-threshold --confirmations`
- Service JSON entries: `onejev: {url}` (qev, tunnel only), `mjev: {url, token}`.
- `utterances.jsonl`; summary `holds`, `said`, `step_events`, `verify`.
- Spoofs (declared, `harness/deciders.py`): `KeywordChooser` (OneJev's scores; a deviation reads as wait), `RewriteReviser` (the trained-task choice; the scripted S1 needs step text), `TruthVerifier` (MJev's checks from simulator truth).

## Tests
| File | Pins |
|---|---|
| `tests/test_system2_jev.py` | candidates per situation, budgets, each act's operation, resume, a bad motion holds, abstention, wait-while-held, dwell, wait budget (D7), the trained sentence (never the user's words) on revise, none / low margin → ask then wait, temperature fit, model inputs, owner + Jev stop → heartbeats → go on |
| `tests/test_system2_verify.py` | confirmation streak, one event per command, no check without a step, failures counted, one MJev request |
| `tests/test_owner.py` | holds, dialogue, utterances, step events, D8 ordering, late answers applied as decided, no recheck while awaited |
| `executors/robocasa-pi052/tests/test_harness.py` | Jev with keyword scores answers like the rules; both questions recorded as cases; held-out decided accuracy |

## Live (GPU suite 2026-10-22, before the owner fixes)
- OneJev decision 1.1–1.5 s over the WAN; deviation → retry 8/8; "put the steak on the plate" while picking → continue 8/8 on a fixed snapshot, revise p 0.87 in the gate run.
- Two owner bugs found and fixed: review lease (1 s) expired before the answer; utterances replaced the task text.
- Improvement order: wording / inputs → calibration → T9 (evidence vs compute errors) → adaptive depth / LEV only for compute errors (decisions 2026-10-22).

## Campaign 2026-10-24 (XR-1 + Jev, declared calibration; `results/campaign/20261024-134629/analysis.md`)
- 82 user lines (27 scripted runs, 8 fuzz, 2 live): 55 right, 2 unsafe ("ruko" → revise to the faucet task; "go on"
  after success → resumed the finished task), 24 safe misses, 1 unanswered by design.
- Main misses: "stop" split abort / wait → ask (4 / 9); listed new tasks right only 6 / 14 (abstained, or kept holding
  / stopped); remarks with fillers asked; a status question → abort; task question tied on shared words ("sink").
- Never revised on unlisted, gibberish, negated or vague lines; never sent the user's words to S1.

## Not validated
- OneJev on these questions (accuracy, latency vs `interaction_timeout_s` 1 s, D6); calibration fit on a held-out set.
- OneJev on the task question (choosing among task sentences) has never run.
- MJev P(done) against truth (T6); π0.5²'s subtask switch as a step event is built but not yet run with π0.5².
- ASK has one generic template until perception supplies object ids.
