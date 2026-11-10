# How rojev runs a robot (2026-11-10)

The whole system on one page. Numbers: [`../paper/experiments.md`](../paper/experiments.md). Reasons:
[`../decisions.md`](../decisions.md). Where the code is: [`../CODEBASE.md`](../CODEBASE.md).

## 1. The idea
- **S1 (System 1) decides how to move.** A vision-language-action model (images + robot state + a short text in,
  actions out). It returns a *chunk*: the next ~0.8 s of actions. It runs on a GPU and is asked over and over.
- **S2 (System 2) decides what to do, and whether it worked.** Slower models that pick the next step, answer the
  user and check whether a step is done. They never send actions.
- **The owner** sits between them. Models only propose; the owner decides what becomes a command, and keeps the task
  memory. A command counts as running only once the robot has applied its first action.
- **The control layer** moves the robot and keeps it safe, whatever the models do.

## 2. The parts
| Part | Job | Model today |
|---|---|---|
| Control loop | steps the robot at its fixed rate (20 Hz) and keeps the newest camera frames | — |
| Execution runtime | applies one action per tick; drops stale or cancelled actions; protects the gripper | — |
| Guard | holds the robot still if the owner goes silent (2 s) or actions stop arriving (0.5 s) | — |
| S1 runner | asks S1 for the next chunk before the current one runs out | the S1 model |
| Owner | decides when to ask S2, accepts or refuses its answers, issues commands, keeps memory | — |
| Planner (S2) | picks the next step from the steps S1 was trained on | Embodied-R1.5 (built, not yet run live) |
| Jev (S2) | answers the user ("wait", "go on", a new task) and motion problems | OneJev 1.6 4B |
| Done checker (S2) | checks whether the running step is finished | Embodied-R1.5 or OneJev; none reliable yet |
| Shared assembly | wires all of the above for any robot from a config file | — |
| Robot plug-in | the robot's actions, cameras and simulator or driver | LIBERO, RoboCasa |

## 3. How data flows
```
 user ──speech──▶ OWNER ──"what next?" (newest frames + memory)──▶ S2: planner · Jev
                   ▲  ◀──────────────── proposal ──────────────────┘
                   │  command (step text, or "hold")
                   │        ▼
                   │   RUNTIME ◀── chunk ── S1 runner ◀── S1 model (GPU)
                   │      │ one action per tick           ▲ newest frames
                   │   GUARD → robot / simulator ─────────┘
                   │      │ gripper opened or closed
                   └── "step done" ◀── done checker (S2)
```
A command tells S1 either one step ("open the drawer"), the whole task, or to hold still.

## 4. When S2 is asked
There is no fixed beat: a planner that thinks for seconds is asked only when its answer could change.

| Event | Answered by |
|---|---|
| The run starts | planner |
| A step was confirmed done | planner |
| S1 picked its own next step (π0.5² only) | planner |
| A command ended on its own (refused, expired, or stopped by the guard) | planner |
| One step has run longer than its budget (off unless set) | planner |
| Nothing asked for 10 s (a safety net) | planner |
| The user spoke | Jev |
| The robot's motion went wrong (optional motion monitor) | Jev |
| The robot has been holding for a set time (optional) | Jev |
- Jev's answer goes first: while it is pending, planning waits.
- After a stop or wait from the user, the planner cannot restart the task; only a new answer to the user can.
- Only the newest question is kept, so a slow model never works through a backlog.

## 5. Memory
- **Task memory**: for each step, *done* or *undone*, with when, why and on which frame. Only the owner writes it;
  every question to S2 carries a copy.
- **Done** is recorded when the done checker confirms a step (at once, so the planner already sees it), or when the
  planner moves on past a step that was running.
- **Undone** is recorded when the planner picks a step that memory says is done: the scene contradicts the memory
  (a drawer closed again), so the step is owed again.
- Waits, stops, questions and retries are not progress.
- **Episode log**: when a task ends, one line with the goal, every step record and the conversation. Failures are
  kept. Nothing reads it back yet.
- **Not known yet:** that a step was undone before the planner picks it again (writing each skill's effects by hand
  per environment was rejected); failed attempts; where objects are.

## 6. Checking a step is done
A check is a model call, so it runs when a step has probably just ended, not every tick.
1. **Cue:** the gripper (or another switch) changed on an applied action → check now.
2. **Fallback:** no cue → check every 1 s.
3. **Confirm:** a "done" answer is rechecked 0.5 s later; two in a row → "step done" goes to the owner.
- Only steps with a known step text are checked. A failed check is counted, never treated as done.
- The 0.7 confidence bar and the two confirmations are set by hand, not measured.

## 7. When something is slow or dead
Every command has a *lease*: an expiry time. The owner renews it while the system is healthy; if it is not renewed,
the command expires and the robot holds still. Keeping the robot alive is the owner's job, not a model's.

| Situation | What happens |
|---|---|
| Planner slow | the robot keeps doing the current step; the answer is used if it is still about that step |
| Planner silent for 30 s | the owner stops renewing; the robot holds; the next answer moves it again |
| User speaks | the robot keeps moving but cannot open or close the gripper until Jev answers |
| Jev late (over 1 s) | the robot holds and says so; the late answer is applied to the moment it was made for |
| S1 slow | the runner asks earlier, by the delay it measured; in real time an answer older than one chunk is dropped |
| S1 or network dead | after 0.5 s without actions the guard holds; the next chunk resumes |
| Owner dead | after 2 s the guard holds and cancels all commands |
| Done checker fails | the failure is counted; nothing is marked done |

## 8. S1 today
| Model | Robot | Status |
|---|---|---|
| GR00T N1.7, NVIDIA's LIBERO checkpoints | LIBERO (simulated Franka arm) | 49/50 on LIBERO-goal through the whole runtime, as NVIDIA's published 98% (E33) |
| XR-1 (Xiaomi-Robotics-1) | RoboCasa (simulated Franka on a mobile base) | 18/20 on its published test cases (E23) |
| π0.5² | RoboCasa | kept for comparison |
| GR00T N1.7, base model | SO-101 arm (planned) | reproduces NVIDIA's numbers zero-shot (E32); no SO-101 training yet |
- One S1 service serves any model in LeRobot's format; a new one needs only a small config file.
- **S1 only gets text it was trained on.** The planner may only pick from the trained steps; Jev may only switch to a
  trained task sentence. The user's own words never reach S1.
- **S2 sees the present; the past reaches it as text.** The planner and the done checker get the newest frame of each
  camera, and the planner also gets the memory's list of finished steps. The done checker asks about each camera
  separately; a yes from any camera means done. The offline test asks the same way.

## 9. S2 today
| Role | Model | Status |
|---|---|---|
| Planner for an S1 that follows step text | Embodied-R1.5 | built; failed the offline test: right next step 63% of the time (needed 90%) and never redoes a step the list wrongly calls done (E34) |
| Planner for an S1 that takes the whole task | none (the task goes to S1 as is) | in use (LIBERO, XR-1) |
| Jev (talking to the user) | OneJev 1.6 4B | built; 55 of 82 user lines right in the first campaign (E27) |
| Done checker | Embodied-R1.5 or OneJev | built; no model has passed the done-check test yet: R1.5 is at chance on SO-101 steps and 0.69 on real failures (E17, E31, E34) |
- An answer outside the allowed steps keeps the current step, so a garbled answer never moves the robot.
- One client talks to any vision-language model served by vLLM; its prompts live in a config file (new, may change).

## 10. Robots and the shared assembly
| Robot | Runs through | Note |
|---|---|---|
| LIBERO Franka (7-number action: hand motion + gripper) | the shared assembly | LeRobot's own LIBERO environment; runs on the GPU machine (Linux only) |
| RoboCasa Franka on a mobile base | its own older wiring | kept as is; uses the new owner, memory and done checks |
| SO-101 (simulated) | tests only | no trained S1 |
- A new robot supplies its action layout, camera names, start pose and simulator or driver; the assembly does the rest.
- Settings and plug-ins come from one config file; a misspelt setting is refused.
- The shared assembly has no motion monitor yet; the older RoboCasa wiring has no step planner. The two can drift.

## 11. Two time modes
| Mode | What it does | Used for |
|---|---|---|
| Lockstep | simulated time stops while any model thinks | benchmarks (the shared assembly's default) |
| Real time | everything runs at once on the robot's clock; nothing waits | responsiveness, hardware |

Lockstep hides model delay; real hardware must pass in real time.

## 12. What a run leaves behind
| File | Content |
|---|---|
| `config.json` | the robot, task, models and settings |
| `owner.jsonl` | every owner event: question asked, answer accepted or refused (and why), command, hold, step done / undone |
| `decisions.jsonl` | every S2 answer, why, and how long it took |
| `s1_requests.jsonl` | every S1 call and its timings |
| `ticks.jsonl` | every action applied |
| `utterances.jsonl` | what the robot said |
| `episodes.jsonl` | the episode log |
| `summary.json` | how it ended, timing, memory, counts |
| `view.mp4` (optional) | the cameras side by side with a status panel |

LIBERO also writes `results.json`: success per episode, task and suite.

## 13. Tested, and not validated
| Tests (re-run 2026-11-10, laptop, no GPU) | Result |
|---|---|
| Core | 357 passed |
| Shared assembly with a stand-in robot | planner + done checker finish a two-step task |
| LIBERO plug-in (fake simulator under LeRobot's real code) | 21 passed |
| S1 service, R1.5 bench (stand-in models) | 32 and 2 passed |
| RoboCasa app; scenario matrix with a scripted S1 (2026-11-09) | 37 passed; 17 of 17 cases |
| GR00T on LIBERO-goal through the whole runtime (A100, 2026-11-10, E33) | 49/50; no starved ticks, rule violations or S1 errors |

**Not validated yet:**
- GR00T and R1.5 together on a GPU;
- any done checker good enough to trust;
- anything on a real robot.
