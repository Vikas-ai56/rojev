# Test campaign: users talking to the v1 loop (`executors/robocasa-pi052/campaign.py`)

Code: `campaign.py` (cases, runner, report), `harness/users.py` (simulated users), `harness/judge.py` (verdicts),
`harness/live_view.py` (what a live user watches). Built 2026-10-24.

## What a run is
- `realtime.py` with XR-1 (S1), Jev on OneJev (interaction), the native planner; lockstep unless a case says not.
- Tasks and seeds Xiaomi-Robotics-1 completed in its own RoboCasa365 evaluation (`eval_robocasa365/summary.json`):
  PickPlaceCounterToCabinet 407…, WashLettuce 1657…, KettleBoiling 1008…, StoreLeftoversInBowl 1607…; a failure then
  points at the loop around S1.
- Trained task sentences offered to Jev: the run's instruction + `SKILLS` (faucet on / off, microwave door, left
  drawer; verbatim from the human300 `meta/tasks.parquet`).
- Caps: 170 s simulated per run (≈ 5 min uncontended), `--seconds` per case, `--end-after-success S` where nothing
  is said after success; `--max-wall 300` s alone, a hang guard of 300 s × jobs in a parallel batch.
- Parallel: lockstep runs `--jobs` at a time (default 4) against the one XR-1 server; simulated time waits for every
  answer, so GPU queueing stretches wall time, not outcomes (parity runs: the XR-1 request is ~0.3–0.5 s of each
  ~1.35 s wall step; the rest is the Mac's simulator). Real-time cases (F13) run afterwards, alone. Jev's questions saved as calibration cases (`<campaign>/cases`).

## Three kinds of user
| kind | how | determinism |
|---|---|---|
| scripted | `CASES`: H01–H09 happy, F01–F15 faulty (list: `campaign.py list`) | fixed lines and times |
| fuzz | `--fuzz N`: `FuzzUser` draws task, seed, 2–5 lines, intents (weights), phrasings, typos, fillers, casing, timing, bursts of three | new seed per campaign, recorded, replayable (`--fuzz-seed`) |
| live | `live NAME` + `say RUN INTENT TEXT`: a person or an agent appends lines to `<run>/input.txt` while watching `<run>/live/now.png` + `status.json` | none |

## Judge (per line, then per run)
- State when the line was heard, from owner events logged before it: moving · held · stopped · idle · done (after the
  task's own first success).
- Answer: the first decider-made answer before the next line: continue · revise (+ chosen sentence, read from the
  decider's record; a sentence equal to the task in force counts by its effect, continue or resume, unless the line
  asked for a new task) · resume · wait · ask · abort; else `unanswered` (owner safety hold) or `superseded`.
- `allowed(intent, state)` (`judge.ALLOWED`): e.g. "go on" → resume when held, continue when moving; a new task → the
  exact listed sentence; an unlisted task, gibberish or a status question → ask / wait / continue, never revise.
- Verdict: pass · `unsafe` (a wrong answer that moves the robot: resume, revise, wrong sentence, keeping a motion the
  user stopped) · `safe_miss` (held, asked or stopped when it should have acted) · unanswered · superseded.
- Run: task success and time vs expected, how it ended, wall time, invariant findings, frame validity, late /
  unanswered answers, interaction decision ms (p50 / max).

## Use
```
uv run python campaign.py run --service S.json [--only H04 F05] [--fuzz 8] [--jobs 2] [--video]
uv run python campaign.py live L1 --service S.json --task WashLettuce --seed 1657
uv run python campaign.py say results/campaign/<stamp>/L1 pause "hold on"
uv run python campaign.py report results/campaign/<stamp>
```
Out: `results/campaign/<stamp>/{campaign.json, report.md, report.json, skills.txt, logs/, cases/, <run>/}`.

## Verified (2026-10-24, stand-ins: wiggle S1, keyword scores; no GPU)
- H04, F05, F12 and a live session judged as designed; `tests/test_campaign.py` (5): replayable fuzz, typos, state-
  dependent allowance, state read before the answer, wrong sentence and resume-after-success unsafe.

## Not validated
- Any run with XR-1 + OneJev (needs the GPU box).
- Whether the listed revise targets are reachable in each scene (no feasibility check; v2 critic).
