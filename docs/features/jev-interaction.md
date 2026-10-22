# Jev v1 interaction lane + end-state checks (spec §9, step 6)

Code: `rojev/system2/jev/` (acts, calibration, chooser, interaction), `rojev/system2/verify/` (checks, watcher),
owner support in `rojev/owner/state_owner.py` (holds, dialogue, utterances, step events). Built 2026-10-22; live OneJev
/ MJev runs and the calibration fit need the GPU.

## Roles (decided 2026-10-21)
- π0.5² picks the next step and moves; Jev answers only changes; MJev checks end states.
- Interaction lane = `JevInteraction`; planning lane = π0.5²'s own planner or a baseline decider.

## Flow of one change
```
user speaks → owner: rev+1, user Turn, ACK utterance, moving command under review
            → interaction lane: JevInteraction.decide(snapshot)
                 candidates in code → Chooser (OneJev Choice) → Calibration → margin / budgets → Operation
            → owner accepts → command (+ Hold if ABORT / WAIT / ASK) (+ utterance if templated)
```

## `JevInteraction` (stateless; everything comes from the snapshot)
| Step | Rule | Model call |
|---|---|---|
| `hold` trigger | stop → keep holding; else report once (`waiting_for_user`), then keep holding (D7) | no |
| dwell | deviation < `min_dwell_s` (1 s) after the step started → CONTINUE | no |
| candidates | legal acts per situation (table below) | — |
| score | `Chooser.choose` → probabilities; `Calibration.apply` (temperature) | one OneJev Choice |
| abstain | top-two margin < `margin` (0.2) → ask if allowed, else wait | — |
| budgets | `max_asks` (2) questions in the dialogue; never two questions in a row | — |
| act → op | continue → CONTINUE · resume → interrupted step · retry → RETRY step · revise → `Reviser` · wait → GUARDED_WAIT · ask → ASK(`clarify`) · abort → ABORT | — |

| Situation | Legal acts |
|---|---|
| instruction, moving | continue · revise · wait · ask · abort |
| instruction, idle | continue · revise · ask · abort |
| instruction, held | resume (if a step was interrupted) · continue (keep holding) · revise · ask · abort (not if already stopped) |
| deviation | continue · retry (if a step runs) · revise · ask |

- `Reviser`: `S1Replans` (default) = SWITCH_TO(None), π0.5²'s planner re-plans under the new instruction (D1).
- Wait while already holding → CONTINUE (keeps the hold, no new command).
- `OneJevChooser` state: instruction, trigger, current step, steps done (keyframes), conversation (robot turns rendered from `TEMPLATES`), hold, motion check; image = cameras side by side.

## Calibration (D5)
- `Calibration{temperature, margin, fitted, cases}`; `fit_temperature(rows, labels, margin)` = NLL grid search.
- **Today T = 1, margin 0.2: declared tuned constants.** Fit needs a labelled set (utterance × situation × scene → act) scored by OneJev on the GPU.

## Owner support
- Holds persist until a change resolves them (D4 fix); `hold_recheck_s` asks the interaction lane again (wait budget).
- Dialogue: last 8 turns in every snapshot; `Utterance` out through `say` (`ACK` at once, ≤ one owner tick; templated ops).
- `StepEvent` → planning lane (trigger `step`) + keyframe evidence.

## End-state checks (`system2/verify`)
- `Verifier.check(images, steps) → {step: {done, possible, blocked}}`; `MJevVerifier` = one MJev request, the arm-D question per step (no per-skill parser).
- `StepWatcher` thread: every `every_s` (1 s) checks the runtime's active command (if it moves with a subtask); `confirmations` (2) consecutive P(done) ≥ `done_threshold` (0.7) → one `StepEvent` per command; failures counted, never posted.
- Threshold and confirmations = declared tuned constants until T6 (MJev vs simulator truth).
- `RemoteMJev` uses `rojev.transport` (one kept connection).

## Executor (`executors/robocasa-pi052/realtime.py`)
- `--interaction jev --chooser onejev|keywords --reviser s1|rewrite --max-asks N --jev-calibration FILE`
- `--hold-recheck S` · `--verify mjev|truth --verify-every --done-threshold --confirmations`
- Service JSON entries: `onejev: {url}` (qev, tunnel only), `mjev: {url, token}`.
- `utterances.jsonl`; summary `holds`, `said`, `step_events`, `verify`.
- Spoofs (declared, `harness/deciders.py`): `KeywordChooser` (OneJev's scores), `RewriteReviser` (π0.5²'s re-planning; the scripted S1 needs step text), `TruthVerifier` (MJev's checks from simulator truth).

## Tests
| File | Pins |
|---|---|
| `tests/test_system2_jev.py` | candidates per situation, budgets, each act's operation, resume / retry, abstention, wait-while-held, dwell, wait budget (D7), temperature fit, model inputs, owner + Jev stop → heartbeats → go on |
| `tests/test_system2_verify.py` | confirmation streak, one event per command, no check without a step, failures counted, one MJev request |
| `tests/test_owner.py` | holds, dialogue, utterances, step events, D8 ordering |
| `executors/robocasa-pi052/tests/test_harness.py` | Jev with keyword scores answers like the rules |

## Not validated
- OneJev on these questions (accuracy, latency vs `interaction_timeout_s` 1 s, D6); calibration fit.
- MJev P(done) against truth (T6); step events from π0.5²'s own subtask switch and phase boundaries not built.
- ASK has one generic template until perception supplies object ids.
