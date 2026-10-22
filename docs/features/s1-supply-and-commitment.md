# S1 supply and commitment: runner, latency-derived floor, commitment policies (steps 2 + 4.2, 2026-10-20 / 21)

Code: `rojev/system1/runner.py` (`S1Runner`, `S1Stats`), `rojev/system1/commit.py` (`FixedCommit`, `PhaseCommit`, `ConsensusCommit`, `make_commit_policy`), `rojev/system1/policy.py` (`Policy.act`, `act_samples`), `rojev/control/execution.py` (`target`, `queued_until`, `submit`, `starved_ticks`). Tests: `tests/test_system1_runner.py` (14, 2 `timing`), `tests/test_system1_commit.py` (42). Spec: `docs/plan/v3-v1-spec.md` §14. Remote policy: [`remote-models.md`](remote-models.md); deviations: [`execution-monitor.md`](execution-monitor.md).

In short: S1 returns a chunk of actions; the commitment is how many of them the robot is committed to. Three policies: `FixedCommit` (default), `PhaseCommit`, `ConsensusCommit` (only when the user turns on `speculative_predict`). How short a commitment may be is set by the policy's latency, not by the policy.

## Cycle (one `S1Runner.step`, own thread)
1. `runtime.target()`: the pending command, else the active one while it moves (or is held "starved"); none → no request.
2. Ask if `refill_now(cmd)` was taken for this command, nothing is queued for it, or `queued_until(cmd) − now ≤ lead_s()`.
3. `latest()` = newest `Capture`; `obs.task` replaced by the command's `AcceptedGoal.instruction` (never the stale environment text); subtask = `operation.subtask`.
4. `policy.act(obs, subtask)`, or `act_samples(obs, subtask, K)` when the commit policy has `samples` K > 1.
5. `commit.plan(candidates, spec, floor, last)` → `Commitment(actions, chosen, reason, detail)`; `last` = `runtime.last_action()` (the action last applied).
6. `TimedChunk(epoch, cmd, obs_id, t_start = capture.t, dt, committed actions)` → `runtime.submit`: action k due at `capture.t + k·dt`; actions already due are skipped, never replayed; the chunk replaces the command's queue from its first due time.
7. Latency sample `now − capture.t` recorded (also when the chunk is refused).

## Who decides what
| Who | Decides | Where |
|---|---|---|
| State owner | which command exists, its instruction, lease, review | `StateOwner` → `ExecutionRuntime.accept / renew / review / revoke` |
| Runtime | which command needs chunks; accepts or refuses each chunk; one action per tick; counts starvation | `target`, `submit`, `tick`, `starved_ticks` |
| Runner | when to ask, on which observation, how many samples, the floor | `S1Runner.step`, `lead_s`, `floor` |
| Commit policy | which candidate, how many actions (never below the floor) | `CommitPolicy.plan` |
| Guard | hold "starved" after `watchdog_s` 0.5 s without a valid action; resumes when chunks return | `Guard` |
| Execution monitor | "ask now" on a deviation (`refill_now(deviation.cmd_id)`) | [`execution-monitor.md`](execution-monitor.md) |

## Latency → lead time and floor (`runner.py`)
- L = observation age when its chunk is submitted (robot clock): includes K-sample cost, network, server queue. Estimate = max of the last `latency_window` = 20 samples (`latency_s`); max, so a commitment covers the slow tail.
- Gap-free condition (one request in flight): lead ≥ L and committed span n·dt ≥ 2L, plus one tick of quantization.
- `lead_s() = max(refill_s, L + dt)`: `refill_s` is a lower bound only; a too-low setting is overridden after the first sample.
- `floor(horizon) = ⌈2L/dt⌉ + 1`, passed to `plan`; per request in `S1Stats.floors`.
- Floor > chunk length (even the whole chunk cannot cover 2L): counted per request in `S1Stats.floor_over_chunk`, not silent; floor = chunk is still gap-free.
- First request (no sample): lead = `refill_s`, floor = whole chunk (shortest candidate). Only this refill can starve, by at most one latency.
- Policy error: no latency sample.

## Commitment policies (`commit.py`)
Every policy commits at least `min(floor, horizon)` and at most the chunk, always a prefix of one candidate.
- **Motion** (robot-agnostic): continuous dims only; numeric hold value → distance from it (velocity-like); `KEEP` → step-to-step change (position-like).
- **`FixedCommit`** (default): whole chunk, or `max_commit` raised to the floor (a cap below it would starve).
- **`PhaseCommit`** (PACE): through the first local speed minimum after the floor below `slow_fraction` (0.3) × the chunk's peak speed; none → whole chunk (or `max_commit`). Reasons `phase boundary` / `no slow point`.
- **`ConsensusCommit`** (CA³C without its world model; opt-in), K = `samples` (default 3, ≥ 2):
  1. First step where candidates disagree on an irreversible switch (`first_switch_disagreement`); switch limit = that step, raised to the floor (horizon if none).
  2. Switch values up to the limit, per step and switch (`majority_switches`): the strict majority's value; on a tie (no strict majority) the value in effect stays: the previous step's, and at step 0 the robot's switch in `last`. Only with `last` None (nothing applied yet) is a step-0 tie unknown until a majority forms.
  3. Candidates following that sequence (any value where unknown); chosen = the one closest (squared error, continuous dims) to their mean.
  4. Agreement walk: per step, count candidates whose motion differs from the chosen one's by ≤ `tolerance` (0.5) × median candidate speed, smoothed over the last `smooth` (3) steps; chosen counts itself. Prefix ends at the first step ≥ floor without a strict majority (2·agreeing ≤ K); one outlier among three cannot shorten it. Median speed 0 → any motion is disagreement.
  5. Commit `min(agreed, switch limit, then.plan(chosen))`; `then` = Fixed or Phase.
  - Reasons: `candidates agree`, `then`'s reason, `candidates diverge`, `stop before a disputed switch`; disputed switch inside the floor: `switch decided by majority`, `switch tie keeps the value in effect`, `switch without a majority` (step-0 tie before anything was applied, or no candidate follows the majority). `detail`: horizon, agreed, capped, switch, votes (fewest candidates backing the committed switch values at a step, `2/3`), min_agreeing.
  - Majority, not plurality: 2 of 5 closing first (largest identical group) do not decide; the step where 3 of 5 have closed does.
  - Odd K never ties on one binary switch; with K = 2 any disagreement after the floor ends the commitment.

## Failure handling
| Case | What happens | Pinned by |
|---|---|---|
| One wrong sample (misperceived object) | outside the largest switch group or far from its mean → never committed | `test_an_outlier_is_never_the_committed_candidate`, closed-loop minority failure |
| Early disagreement | agreement checked only from the floor; floor always committed | invariants test |
| Disputed switch beyond the floor | commitment stops before it; next request decides on a fresher observation | `test_disputed_switch_beyond_the_floor_…` |
| Disputed switch inside the floor | majority decides; commitment = floor (no livelock) | `test_disputed_switch_inside_the_floor_…` |
| Tie (even K) inside the floor | value in effect kept until a strict majority changes it; at step 0 the robot's switch (`last`); `switch without a majority` only with `last` None | `test_even_k_tie_…`, `test_a_tie_at_the_first_step_…` (both),  `test_the_largest_group_is_not_a_majority` |
| Policy error / unreachable service | `errors`, `last_error`; sleep `error_backoff_s` 0.5 s, retry; guard holds; loop never stops | `test_policy_errors_are_counted_…` |
| Request slower than any of the last 20 | gap possible; counted in `starved_ticks`; guard holds "starved" after 0.5 s; next chunk resumes | runtime starvation tests |
| Policy slower than its whole chunk | chunk entirely in the past → `chunk_rejected`; robot never moves | `test_policy_slower_than_its_chunk_…` |
| Change of mind / review | S1 conditioned on the new instruction; pending command supplied first; old chunk cannot delay the takeover; switches frozen in review (runtime) | runner + runtime tests |
| Stale chunk (revoked, superseded, lease over, wrong `dt` or shape, holding) | `submit` refuses with `chunk_rejected` + reason; never applied | `test_a_chunk_in_flight_when_revoked_is_rejected` |
| Deviation | `refill_now(cmd_id)` → next step asks on the newest observation; new chunk replaces the queue | `test_refill_now_…` |
| Deviation of another command (older, or none) | ignored, `refill_ignored` +1 (also when the command is replaced before the next step) | `test_a_refill_for_another_command_…` |
| Floor > chunk | counted in `floor_over_chunk`; starvation follows (runtime `starved_ticks`) | `test_a_floor_longer_than_the_chunk_is_counted` |
| Unbatched K samples (a policy without its own `act_samples`) | default = K sequential calls → L ≈ K × single latency → floor ≈ K× larger; starves when the chunk is shorter than the floor | `test_unbatched_samples_…` |

## Configuration
- `make_commit_policy(commit_policy="fixed"|"phase", speculative_predict=False, **params)`: strict; a parameter that does not apply raises (`min_commit` removed 2026-10-21, now rejected).
- `S1Runner(runtime, policy, latest, now, refill_s, poll_s=0.005, error_backoff_s=0.5, commit=FixedCommit(), latency_window=20)`; `refill_now(cmd_id) → bool` (taken only for `runtime.target()`).
- `realtime.py`: `--refill` (1.0 s), `--commit-policy fixed|phase`, `--speculative-predict`, `--samples`, `--tolerance`; no flags for `smooth`, `slow_fraction`, `max_commit`.
- Per run: `S1Stats` (requests, accepted, errors, infer_ms, obs_age_s, committed, commit_reasons, forced, refill_ignored, floors, floor_over_chunk); `realtime.py` writes `s1_requests.jsonl`, `commits.jsonl`, summary `s1_latency_est_s`, `s1_floor_last`, `starved_ticks`.

## Evidence
Tests re-run 2026-10-22 after the majority / refill / floor changes (`uv run pytest tests/test_system1_commit.py tests/test_system1_runner.py`): 54 pass; timing tests (`-m timing`): 2 pass.
- Closed-loop reach-and-grasp (deterministic, real runtime + runner, 50 Hz): one of three samples misperceives → fixed grasps air, consensus grasps (`switch decided by majority`, 0 starved); noisy samples → grasp on 10/10 seeds; phase / consensus / consensus+phase at 0.05, 0.1, 0.2 s → 0 starved; a policy ignoring the floor starves (> 20 ticks); unbatched K = 3 at 0.1 s → floor ≥ 31, starves only with a 25-step chunk.
- Randomized invariants: 300 candidate sets (K 2–5) × 5 policies × velocity and position robots.

| Live run (RoboCasa, `realtime.py`) | Result |
|---|---|
| Stand-in 0.3 s, derived floor, 7 runs (2026-10-21) | floor 14, 0 starved ticks |
| CloseFridge 12 s, fixed vs consensus K = 3, hand-set floor 13 (before derivation, 2026-10-20) | both 234 chunk / 8 start-up hold ticks, 0 rejected chunks, 0 violations; consensus cut 5 of 12 commitments (`candidates diverge`) |
| π0.5² over the WAN (keep-alive), MicrowaveThawingFridge 30 s, revisable planner (2026-10-21) | real-time factor 0.97, 0 starved ticks, S1 latency p50 0.7 / p90 0.8 s, floor 34 |

## Not validated
- Consensus / phase with a real stochastic policy (π0.5²): roadmap T8a decides whether either is enabled in benchmark arms.
- Consensus with remote π0.5²: `RemotePolicy.act_samples` = one request, K draws in one forward batch (2026-10-22); K = 1/3/5 latency and draw diversity not measured on GPU ([`remote-models.md`](remote-models.md) "To measure").
- `last` is the last applied action, not the last queued one; actions queued before the chunk takes over are not seen (switches stay frozen outside a running chunk).
