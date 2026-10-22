# Native runtime: what (if anything) should leave Python (research brief, 2026-10-22)

Status: findings, not decisions. Measured on Apple M5 (10 cores), macOS, CPython 3.12.15 (project env) and CPython
3.14.7 free-threaded (GIL off). "Unmeasured" marks everything not run here.

## 1. Answer in one screen
- **v1 (RoboCasa, remote GPU, 20 Hz): move nothing.** Our Python core costs ~22 µs per tick, which is 0.04% of the 50 ms
  budget. The env step takes ~41 ms and the S1 round trip is ~559 ms. Those are MuJoCo/robosuite, the network and the GPU, none of them Python overhead.
- **The real jitter sources are GC and pure-Python CPU hogs sharing the interpreter**, not the speed of our code.
  The largest one is cheap to fix in Python: a full collection in the RoboCasa process takes **45–96 ms**, and **0.0 ms after `gc.freeze()`**.
- **Real SO-101 at 30 Hz: Python is enough** if the control loop gets its own process (or free-threaded CPython) and
  the heap is frozen. A rewrite buys nothing measurable at 33 ms periods.
- **v2 Helix-type fast S1 (~200 Hz, 5 ms): the control path has to stop sharing an interpreter with anything.**
  Measured GIL tails (8–10 ms) and GC tails (9–35 ms) both exceed 5 ms. Free-threaded Python removes the GIL tail but not
  the GC tail. That is when a native real-time core is justified: runtime + guard + loop + driver I/O, behind the current interface.
- **Never native:** the models (already C++/CUDA), the owner and S2 (~1 Hz decisions), the harness, eval.
- **Transport:** a native QUIC stack is the right tool *if* HTTP/3 comes back, but no stack beats the 216 ms ping.
  A closer GPU region is worth more than any rewrite.

## 2. Measurements

### 2.1 Per-call Python cost (no load)
Script: throwaway `prof_core.py`, outside the repo. It uses the repo's stand-ins (`tests/test_control_execution.py`, `tests/test_system1_runner.py`,
`tests/test_system1_monitor.py` patterns), the real `PANDA_OMRON` spec (12 dims, `executors/robocasa-pi052/embodiment.py`),
a 50-step chunk with a fresh splice every 25 ticks, blend 0.2 s, monitor window 10, and the exact `realtime.py:285` log line.
n = 20 000 per row unless noted.

| Call (file:line) | p50 µs | p99 µs | max µs | 3.14t p50 / p99 µs |
|---|---|---|---|---|
| `ExecutionRuntime.tick` (`control/execution.py:277`) | 6.4 | 9.1 | 29 | 6.5 / 9.3 |
| `Guard.step` incl. tick (`control/guard.py:27`) | 6.5 | 9.2 | 26 | 6.6 / 9.5 |
| `ExecutionMonitor.observe` (`system1/monitor.py:170`) | 7.4 | 8.4 | 40 | 8.0 / 8.9 |
| on_tick JSON line to file (`realtime.py:285`) | 4.3 | 8.5 | 93 | 4.3 / 5.2 |
| **Whole tick without the env**: `ControlLoop._tick` + guard + monitor + JSON (`control/loop.py:107`) | **22.1** | **29.7** | 84 | 22.8 / 29.0 |
| `S1Runner.step` issuing a request, `FixedCommit`, 0 ms policy (`system1/runner.py:106`), n = 5 000 | 20.8 | 23.0 | 44 | 20.0 / 26.0 |
| `S1Runner.step` with `ConsensusCommit` K = 3, n = 2 000 | 410 | 440 | 568 | 393 / 436 |
| `S1Runner.step` idle poll (every 5 ms, `runner.py:64`) | 1.5 | 1.5 | 21 | 1.4 / 1.5 |
| `wire.encode_request` webp90, 3 × 256² synthetic frames, 33 KB, n = 200 | 11 053 | 11 454 | 26 134 | 11 223 / 19 586 |

Budget share of the whole tick at its p99 (29.7 µs):

| Rate | Period | Share |
|---|---|---|
| 20 Hz (RoboCasa) | 50 ms | 0.06% |
| 30 Hz (SO-101) | 33 ms | 0.09% |
| 50 Hz (tests) | 20 ms | 0.15% |
| 200 Hz (Helix S1) | 5 ms | 0.6% |
| 1000 Hz (servo-class) | 1 ms | 3% |

- Raw interpreter speed is not the bottleneck at any of these rates. A native port would make the 22 µs perhaps 10–50× smaller (unmeasured), which saves nothing a 20–200 Hz loop can see.

### 2.2 Wake-up jitter under load (where Python actually hurts)
Setup: the real `ControlLoop` in wall-paced mode (`loop.py:95`), a null backend, guard + monitor + JSON log each tick, 3 s per
row. The load runs in one extra thread. Lag is how late a tick starts (`TickRecord.lag_s`), in ms. The loop never re-anchors, so lateness
carries over to the next ticks.

| Load thread | 200 Hz p99 / max (GIL) | 200 Hz p99 / max (3.14t) | 1 kHz p99 / max (GIL) | 1 kHz p99 / max (3.14t) |
|---|---|---|---|---|
| none | 1.21 / 1.24 | 1.17 / 1.22 | 0.23 / 3.37 | 0.21 / 2.97 |
| pure-Python busy loop | **8.3 / 8.3** (50% of ticks > 1 period) | 1.06 / 1.20 | **10.3 / 11.8** (87% > period) | 0.20 / 0.23 |
| numpy 400² matmul | 1.01 / 1.25 | 1.10 / 1.23 | 0.21 / 0.23 | 0.20 / 0.57 |
| WebP encode (`wire.py`, Pillow) | 1.02 / 1.04 | 2.06 / 5.18 | 0.22 / 0.24 | 0.20 / 0.22 |
| S1-runner-like polling thread | 1.19 / 1.31 | 1.17 / 1.24 | 0.22 / 2.93 | 0.21 / 2.30 |
| cyclic-garbage allocator (GC stress) | **22.0 / 23.4** (gen-2 pause 10.3) | **9.0 / 9.3** (GC pause 17.3) | **28.5 / 42.1** | **8.1 / 8.9** |
| all of the above | **55.4 / 65.3** | 10.3 / 13.7 | **101.5 / 124.8** | 9.7 / 10.5 |

- The idle floor of about 1 ms at 200 Hz is macOS `time.sleep` oversleep, which is an OS timer effect, not Python. A native loop on macOS
  sees the same floor unless it uses Mach time-constraint threads. PREEMPT_RT Linux behaviour was not measured.
- C extensions that release the GIL (numpy, Pillow's encoder, socket waits) do not disturb the loop.
  Pure-Python CPU work does: up to one 5 ms switch interval (`sys.getswitchinterval()`) per contention, and more when several threads contend.
- GC is stop-the-world in both builds. Free-threading removes the GIL tail but not the GC tail.

### 2.3 The real processes
| Measurement | Value | Source |
|---|---|---|
| RoboCasa env.step, live π0.5² runs (`results/runs/pi052-live-v2`, `-1404`, 612 / 615 ticks) | p50 41 ms, p99 44–46, max 47–50 | `ticks.jsonl` `step_ms` |
| Loop lag in those runs | p50 1.8 ms, p99 2.5–2.7, max 3.2–3.6; 0–1 ticks over 50 ms | `ticks.jsonl` `lag_ms` |
| env.step measured standalone, moving / holding (100 / 40 steps) | p50 35.1 / 11.8 ms | this brief |
| Tracked objects in the RoboCasa executor process after env build | 519 358 | `gc.get_objects()` |
| Full `gc.collect(2)` in that process (three runs) | **96, 49, 45 ms** | this brief |
| Same, after `gc.collect(); gc.freeze()` | **0.0 ms** (×3) | this brief |
| Earlier recorded stall: video encode on a thread | 236 ms (GIL) | `docs/decisions.md` 2026-10-21 "Live viewer…" |
| Scenario batch 2026-10-21 13:35 | env.step 200–260 ms on ticks 2–9 (startup), then lag recovers ~9 ms/tick; absent in the `-rerun` batch; cause not determined | `results/scenarios/skills-20261021-1335*/` |
| Monitor calibration runs 1001 / 1003 | lag up to 379 / 2306 ms (CPU contention, as logged) | `results/monitor/skills-20261021/`, `docs/decisions.md` 2026-10-21 "Monitor fixes…" |
| Remote S1 round trip (WAN, India → us-east) | p50 559 ms = network 239 + GPU/server rest; ping 216 | `docs/decisions.md` 2026-10-21 "Remote S1 default…" |

- Whether automatic gen-2 collections fire during live runs is **unmeasured**. CPython 3.12 runs one only when new survivors
  reach a quarter of the long-lived heap, which is rare with a heap this size. Growing logs (`S1Stats` lists, owner events) do raise the chance.
  The spikes in `results/monitor/skills-20261021/run_1000..1008` (single ticks of 55–81 ms) are not attributed.
- Our Python share of the remote round trip is the runner (21 µs) plus encoding (about 11 ms in C, outside the GIL). That is about 2% of 559 ms.

## 3. Component table
| Component | Does | Measured cost / jitter | Bottleneck type | Would C++/Rust fix it? | Alternative | Effort / risk | Recommendation (trigger) |
|---|---|---|---|---|---|---|---|
| `ControlLoop` + `Rate` (`control/loop.py`, `robot/base.py:33`) | Paces ticks; latest-value slot | 22 µs / tick; jitter from other threads (2.2) | GIL + GC (shared interpreter), OS timer | Yes for isolation and determinism, not for speed | Own process; free-threaded 3.14t; `gc.freeze()` | Medium / medium | Keep Python for v1 and SO-101. Go native when the control rate reaches ≥ 100 Hz on hardware, or when measured p99 lag exceeds 10% of the period in deployment |
| `ExecutionRuntime` (`control/execution.py`) | Authority rules, splice, blend, bounds, ACKs | 6.4 µs p50 | none at ≤ 1 kHz | Speed no; determinism yes when in a native loop | Same, plus a parity-tested port | High: ~14 rules (docstring), three threads use it | Port only together with the loop (one native core), never alone |
| `Guard` (`control/guard.py`) | Watchdog, owner timeout, stop | +0.1 µs over tick | Stop path depends on the interpreter being alive | Yes: an independent stop path (plan p. 17, "independent local mechanism") | Servo firmware / driver-level timeout; a separate process watchdog | Low–medium / low | First native piece on a real arm, or a hardware timeout instead (trigger: first real-arm safety review) |
| `ExecutionMonitor` (`system1/monitor.py`) | Per-tick residual score | 7.4 µs | numpy small-array overhead | Not needed | Stays on the tick or moves to an observer thread | Low | Keep Python. Advisory, never a stop |
| on_tick JSON log (`realtime.py:281`) | Per-tick record | 4.3 µs p50, max 93 µs (file I/O) | I/O on the control thread | Not needed | Ring buffer flushed by another thread or after the run | Low | Keep. Move off-thread above ~200 Hz |
| `S1Runner` + `CommitPolicy` (`system1/runner.py`, `commit.py`) | Supply chunks, latency floor | 21 µs (Fixed), 410 µs (Consensus K = 3, holds GIL in Python/numpy loops) | Network + GPU (559 ms) | No: network- and model-bound | Batched K samples in the service (spec step 7) | — | Keep Python. Watch Consensus GIL hold only when the loop is ≥ 200 Hz in-process |
| `wire` encode (`system1/wire.py:36`) | WebP-90 frames | ~11 ms / 3 cameras (synthetic; repo reports ~3 ms per camera on real frames) | C (Pillow), releases GIL (2.2) | No | Encode on the GPU host side of a local link; smaller frames | — | Keep |
| Transport (`transport/http.py`, `h3.py`) | Keep-alive HTTP/1.1; HTTP/3 via aioquic | Network p50 239 (HTTP/1.1) vs 302 ms (HTTP/3), aioquic tails inconsistent (decisions 2026-10-21) | Network RTT; aioquic is pure-Python QUIC | Partly: a native QUIC stack removes Python from packet processing; it cannot remove the RTT | quinn (Rust) via PyO3, quiche / msquic bindings, or a sidecar proxy; nearer GPU region; pushed chunks | Medium / medium | Only when HTTP/3 is revisited (lossy robot Wi-Fi, streamed frames). Re-measure first |
| `StateOwner` / `DecisionWorker` (`owner/state_owner.py`, `system2/worker.py`) | Intent, ~1 Hz decisions, 50 ms owner period | Not profiled (event-driven, ≤ 20 Hz) | Model latency (OneJev ~64 ms per question, MJev ~1.3 s) | No | — | — | Never |
| Models / `serve.py` | π0.5², OneJev, MJev on GPU | ~308 ms GPU share of the S1 round trip | GPU, PyTorch (C++/CUDA already) | No | Batching, compile, quantisation, a closer GPU | — | Never |
| Robot drivers: SO-101 Feetech bus (not yet in repo: `robot/so101.py` has only `simulated`) | Sync read/write over USB serial | Unmeasured | Serial bus timing | Not at 30 Hz | LeRobot's Python Feetech driver | Low in Python | Python first. Native only together with the native core (≥ 100 Hz) |
| MuJoCo backend (`robot/sim/`) | Physics, cameras | env.step 35–41 ms (RoboCasa) | C already; robosuite Python around it | No (our code) | Fewer / smaller renders | — | Never: the core never bends to the simulator |

## 4. Robotics fit
- **Real-time needs.** SO-101 at 30 Hz has a 33 ms period. Measured worst-case Python tails under a GIL hog (8–12 ms) fit,
  while GC tails in a large heap (45–96 ms) do not, until the heap is frozen or the loop runs in a small process. A Helix-style
  S1 at 200 Hz has a 5 ms period: no shared-interpreter configuration measured here keeps p99 under 5 ms under load
  (best: 3.14t, 9–10 ms with GC stress). The 200 Hz figure is Figure's number, "not Fenon targets" (plan p. 3). The plan's targets
  are behaviour change p50 ≤ 70 / p95 ≤ 100 ms and a safety stop as an "independent local mechanism with measured finite
  stopping time" (plan p. 17).
- **Where a high rate really lives.** Servo position loops run in servo firmware (Feetech, Dynamixel) or in ros2_control at
  0.5–1 kHz. A 200 Hz S1 needs the policy *and* the chunk executor at 200 Hz, not a 1 kHz Python loop. The native core only
  has to match the S1 rate, plus interpolation if the driver wants more. Unmeasured here.
- **ROS 2 / ros2_control.** `Robot` already mirrors ros2_control's read/write split (`robot/base.py` docstring). There are two paths:
  - (a) a `TimedChunk` becomes a `JointTrajectory` (`time_from_start` = k·dt) sent to `JointTrajectoryController`, which interpolates in C++ at its own rate.
    This gives native real-time without our own native code, but the authority rules (revoke, epoch, lease, irreversible switches) would sit outside the RT loop.
  - (b) our runtime + guard as a C++ ros2_control controller plugin keeps those rules inside the RT loop.
  - (a) suits a quick real-arm bring-up; (b) is the v2 shape if ROS 2 becomes the deployment substrate.
- **Safety.** Today `Guard.stop` needs the Python control thread alive (`guard.py` docstring). A real arm should have a stop
  path that survives a hung or GC-paused interpreter: a servo or driver timeout (hold on silence), or the guard in a native core with
  its own watchdog. This is the strongest non-latency reason to go native. It is small (watchdog + hold + bounds) and can come first.
- **C++ vs Rust (ecosystem, not measured):**

  | Need | C++ | Rust |
  |---|---|---|
  | ros2_control plugin | native (pluginlib) | no first-class path (ros2-rust is community, no controller API) |
  | MuJoCo | C API native | community bindings |
  | Python binding | pybind11 / nanobind + scikit-build-core | PyO3 + maturin (uv builds maturin projects as a PEP 517 backend) |
  | QUIC | msquic, quiche (C API) | quinn, quiche (native) |
  | Servo SDKs | Dynamixel SDK, Feetech SCServo (C++) | community crates |
  | Safety of the stop path | manual memory discipline | memory- and thread-safety by construction |

  If ROS 2 is the deployment target, C++. If the deployment is a standalone core under Python, Rust. Do not mix them.
- **Keeping the architecture rules.**
  - A native core is a second implementation of the `control` layer's public interface (`accept / renew / review / revoke / stop / hold / submit / target / queued_until / tick / drain_acks`, `Guard.heartbeat / stop / step`).
    It is not a new layer. `tests/test_control_execution.py` and `tests/test_system1_runner.py` become parity tests run against both implementations.
  - The protocol types crossing the boundary (`AcceptedGoal`, `Revoke`, `TimedChunk`, `ExecutionAck`, `SCHEMA` in `protocol/commands.py`) need one source of truth: generated, or owned by one side. Two hand-written copies would drift.
  - Dependencies keep pointing down: protocol → native core → `control` facade → `system1` / `owner` (`tests/test_architecture.py`).
  - The core never bends to the simulator: RoboCasa stays a Python backend behind the same `Backend` protocol.
  - Builds stay uv-managed (`uv add` of a maturin or scikit-build-core workspace member; `uv.lock` is never hand-edited).
- **A free-threaded blocker found here.** `aioquic` ships only abi3 wheels, and its dependency `pylsqpack` failed to build for
  3.14t (`uv run --python 3.14t --with aioquic`). `aioquic` is a core dependency (`pyproject.toml:8`), so the whole SDK cannot
  run free-threaded today. numpy 2.5.4, mujoco 3.16.0 and Pillow installed and ran with the GIL off.

## 5. Ordering and decision rules
1. **v1 now (no language change).**
   - Freeze the long-lived heap after setup in executor entrypoints (`gc.collect(); gc.freeze()`): 45–96 ms → 0 ms measured.
     It is process-level, so it belongs in each app's `main`, not in `rojev/`.
   - Log per-tick work time and GC pauses (`gc.callbacks`) in `ticks.jsonl`, so spikes can be attributed instead of guessed.
   - Keep pure-Python CPU work out of the control process (already done for video).
2. **Real SO-101 (30 Hz).**
   - Python core in its own process: control loop + runtime + guard + driver. S1, owner and S2 run elsewhere, over a local queue or shared memory.
   - Alternatively, free-threaded CPython once the `aioquic` blocker is gone (make the transport an optional extra, or pick a native QUIC).
   - A hardware or driver timeout so the arm holds when Python stalls.
   - Rule: rewrite nothing while p99 lag stays under 10% of the period on the real arm.
3. **v2 Helix fast S1 (≥ 100–200 Hz, GPU next to the arm).**
   - Native real-time core (runtime + guard + loop + driver I/O) with Python feeding chunks and commands through a narrow, lock-free interface. Parity tests come first.
   - Or ros2_control path (a) for bring-up and (b) for the full rules.
   - Trigger: a measured S1 cycle ≤ 10 ms on the deployment GPU, plus measured p99 tick lag above 10% of the period with process isolation already in place.
4. **Transport.**
   - Native QUIC only if HTTP/3 returns (lossy Wi-Fi, streamed frames, pushed chunks), re-measured against keep-alive first.
   - A closer GPU region or a local GPU is the larger lever for a 216 ms ping.
5. **Never:** models, owner, S2, eval, harness.

## 6. Limits of this brief
- macOS desktop, not a robot computer: no PREEMPT_RT, no CPU isolation, no Feetech bus. Linux and serial timings are unmeasured.
- The load threads are synthetic. The GC stressor is a worst case. Real live runs showed tails of 3.6 ms at 20 Hz.
- Native-port speedups, servo bus times, and C++/Rust ecosystem claims are not measured here.
- Re-run: the throwaway scripts (`prof_core.py`, `prof_ft.py` for 3.14t with a transport stub, `gc_env*.py`) are not in the repo.
  The method above is enough to rebuild them.
