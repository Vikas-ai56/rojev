# S1 supply and commitment: runner, latency-derived floor, deadline, continuation, commitment policies

Code: `rojev/system1/runner.py` (`S1Runner`, `S1Stats`), `rojev/system1/commit.py` (`FixedCommit`, `PhaseCommit`,
`ConsensusCommit`, `make_commit_policy`), `rojev/system1/policy.py` (`Policy`, `ActionChunk`, `Continuation`),
`rojev/control/execution.py` (`target`, `queued_until`, `queued_from`, `submit`, `starved_ticks`).
Tests: `tests/test_system1_runner.py` (18 + 2 `timing`), `tests/test_system1_commit.py` (43). Spec:
`docs/plan/v3-v1-spec.md` §14. Line-by-line walkthrough: [`system1-map.md`](../features/system1-map.md) §5–6. Remote policy and
services: [`remote-models.md`](../features/remote-models.md).

Terms: a **chunk** is the H actions one policy call returns; the **commitment** is the prefix the robot is committed
to; L is the request **latency** (observation age when its chunk is submitted, robot clock).

## Cycle (one `S1Runner.step`; own thread, or once per tick in lockstep)
1. `runtime.target()`: the pending command, else the active one while it moves (or is held "starved"); none → no request.
2. Request if `refill_now` was taken for this command, nothing is queued for it, or `queued_until − now ≤ lead_s()`.
3. Newest `Capture`; its task replaced by the command's `instruction`; subtask = `operation.subtask` ("" = S1 plans).
4. Continuation (see below) for a policy that `accepts_continuation`.
5. `policy.act(…, deadline_s())`, or `act_samples(…, K)` when the commit policy has `samples` K > 1.
6. `commit.plan(candidates, spec, floor, runtime.last_action())` → `Commitment(actions, chosen, reason, detail)`.
7. `TimedChunk(t_start = capture.t)` → `runtime.submit`: elapsed actions skipped, the queue replaced from the first
   due time; then L recorded.
8. Subtask-less command and the chunk reports a subtask (π0.5² planner): `PlannedSubtask` posted; a change under the
   same command → unchecked `StepEvent(source="s1-planner")`. XR-1 reports none.

## Who decides what
| Who | Decides | Where |
|---|---|---|
| State owner | which command exists, its instruction, lease, review | `StateOwner` → `accept / renew / review / revoke` |
| Runtime | which command needs chunks; accepts or refuses each chunk; one action per tick; counts starvation | `target`, `submit`, `tick` |
| Runner | when to ask, on which observation, how many samples, the floor, the deadline | `step`, `lead_s`, `floor`, `deadline_s` |
| Commit policy | which candidate, how many actions (never below the floor) | `CommitPolicy.plan` |
| Guard | hold "starved" after 0.5 s without a valid action; resumes when chunks return | `Guard` |
| Execution monitor | "ask now" on a deviation (`refill_now(cmd_id)`) | [`execution-monitor.md`](../features/execution-monitor.md) |

## Latency → lead time and floor
- L estimate = max of the last `latency_window` (20) samples: a commitment must cover the slow tail.
- Gap-free with one request in flight iff lead ≥ L and committed span n·dt ≥ 2L (plus one tick of quantization).
- `lead_s() = max(refill_s, L + dt)`; `floor(H) = ⌈2L/dt⌉ + 1`; before the first sample lead = `refill_s`, floor = H.
- Floor > H (even the whole chunk cannot cover 2L): counted per request in `floor_over_chunk`; starvation follows.
- Lockstep: simulated time stands still while S1 thinks, so L = 0 and the floor is 1 (XR-1 parity runs:
  `s1_floor_last` 1).

## Deadline (decisions 2026-10-22)
- `deadline_s()` = the last chunk's duration: an older answer lies entirely in the past.
- On `TimeoutError` the request is abandoned (`timeouts`) and the next one made at once on the newest observation.
- `S1Runner(deadline=False)` in lockstep: a wall-clock deadline there only throws away good answers.

## Continuation (decisions 2026-10-22)
- For a policy with `accepts_continuation`, after the first L sample: `Continuation(queued_from(cmd, capture.t),
  delay_steps = ⌈L/dt⌉)` = the actions that run while it thinks (`continuations` counted).
- Placement and skipping are unchanged; using it is the policy's job (`serve.py --rtc H` for π0.5²).
- `RemotePolicy` accepts it; `HistoryPolicy` (XR-1) does not: XR-1 is synchronous.

## Commitment policies (`commit.py`)
Every policy commits at least `min(floor, H)` and at most the chunk, always a prefix of one candidate.
- **Motion** (robot-agnostic): continuous dims; numeric hold value → distance from it; `KEEP` → step-to-step change.
- **`FixedCommit`** (default): the whole chunk, or `max_commit` raised to the floor.
- **`PhaseCommit`** (PACE): through the first local speed minimum after the floor below `slow_fraction` (0.3) × peak;
  none → whole chunk. Reasons `phase boundary` / `no slow point`.
- **`ConsensusCommit`** (CA³C without its world model; opt-in `speculative_predict`; K = `samples`, default 3, ≥ 2):
  1. Switch limit = first step where candidates disagree on an irreversible switch, raised to the floor.
  2. Per step and switch the strict majority's value; a tie keeps the value in effect (previous step, or at step 0 the
     switch in `last`); only with `last` None is a step-0 tie unknown.
  3. Chosen = among candidates following that sequence, the one closest (continuous dims) to their mean.
  4. Agreement: a candidate agrees while its motion differs from the chosen one's by ≤ `tolerance` (0.5) × median
     speed, smoothed over `smooth` (3) steps; the prefix ends at the first step ≥ floor without a strict majority.
  5. Commit `min(agreed, switch limit, then.plan(chosen))`; `then` = Fixed or Phase.
  - Reasons: `candidates agree`, `then`'s reason, `candidates diverge`, `stop before a disputed switch`, `switch
    decided by majority`, `switch tie keeps the value in effect`, `switch without a majority`.
  - Majority, not plurality: 2 of 5 closing first do not decide. With K = 2 any disagreement after the floor ends it.

## Configuration
- `S1Runner(runtime, policy, latest, now, refill_s, poll_s=0.005, error_backoff_s=0.5, commit=FixedCommit(),
  latency_window=20, on_step, on_plan, deadline=True)`; `refill_now(cmd_id) → bool`.
- `make_commit_policy(commit_policy="fixed"|"phase", speculative_predict=False, **params)`: a parameter that does not
  apply raises.
- `realtime.py`: `--s1 pi052|xr1`, `--refill` (1.0 s), `--lockstep`, `--commit-policy`, `--speculative-predict`,
  `--samples`, `--tolerance`; no flags for `smooth`, `slow_fraction`, `max_commit`.
- Per run: `S1Stats` → `summary.json` (`s1_latency_est_s`, `s1_floor_last`, `s1_timeouts`, `starved_ticks`, …);
  per call `s1_requests.jsonl`, `commits.jsonl`.

## Evidence
- Tests: both files pass in the full suite re-run 2026-10-24 (329 passed); `-m timing`: 2 passed.
- Deterministic closed loop (real runtime + runner, 50 Hz): with one of three samples misperceiving, fixed grasps air
  and consensus grasps; a policy ignoring the floor starves; randomized invariants over K 2–5 and both motion kinds.

| Live run (RoboCasa, `realtime.py`) | Result | Status |
|---|---|---|
| XR-1 parity, lockstep, `--refill 0.05`, fixed (`results/xr1-parity/20261022-fixed`, 20 runs) | 0 starved ticks, 0 timeouts; one request per ~14.5 ticks (31 per 450-tick episode) | valid (E23) |
| XR-1 real time, F13 (`results/campaign/20261024-134629/F13`) | L estimate 0.9 s, floor 38 > 16-action chunk; 64 requests, 36 accepted, 13 timeouts; 651 of 985 ticks starved | valid (E24) |
| π0.5² over the WAN, keep-alive (2026-10-21) | — | withdrawn: corrupted frames |
| RTC / delay A/B with π0.5² (2026-10-22) | — | withdrawn: corrupted frames; not re-run |