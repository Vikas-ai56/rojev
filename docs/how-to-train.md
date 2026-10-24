## My conclusion

**Keep the S1/S2 architecture, but change the training priority: teach the models to succeed from the states their own decisions create—not merely imitate clean demonstrations.**

I reread the plan. It already includes handoff training, recovery examples, instruction revisions, delayed advice, and joint latent alignment. Those are the right ingredients. However, I would promote **recovery, task memory, and downstream consequences** ahead of adaptive depth or a new latent interface. Jev_Helix_Implementation_Plan Jev_Helix_Implementation_Plan

Also, I would no longer assume the plan’s unified typed Jev is the main planner. Given your reported E13 result, treat **generative planning + trained Jev checking/grounding** as one candidate, and compare it against the VLA’s native planning capabilities.

The research points to **seven useful training directions**.

---

## 1. Zoom out: what actually needs to improve?

For a composite task, these are different capabilities:

```text
Understand the goal
        ↓
Choose a useful subgoal
        ↓
Execute it successfully
        ↓
Finish in a state that supports the NEXT subgoal
        ↓
Recognize what happened and update task progress
        ↓
Recover when something goes wrong
```

A model can improve at one box without improving the complete episode.

For example, “pick up the object” can succeed while leaving the object in a grasp that makes the next operation impossible. **Training for local success is not necessarily training for composite success.**

That distinction is explicitly studied in *Sequential Dexterity*: it trains successors on predecessor end states, then also improves predecessors using the successor’s feasibility signal. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v229/chen23e/chen23e.pdf)

### One evaluation distinction before discussing training

| Goal | What is allowed |
|---|---|
| **Official RoboCasa365 leaderboard improvement** | Follow the published restriction: unaltered `pretrain_human300`; no added subtask annotations or other RoboCasa training data. |
| **Improve your complete research system on RoboCasa tasks** | You can investigate recovery rollouts, simulator-derived labels, and hierarchical training—but disclose that additional supervision and report it separately. |

The official documentation explicitly prohibits its subtask annotations for leaderboard training. Also, training on demonstrations of the held-out composite tasks means those tasks are **no longer zero-shot unseen**, even if they were unseen during pretraining. [RoboCasa](https://robocasa.ai/leaderboard?utm_source=chatgpt.com)

**Most recommendations below belong initially to the second track.**

---

# 2. Highest priority: train both sides of a skill transition

### What to change

Your plan already says:

> Train the next skill from the actual states produced by the previous skill.

I would extend that to:

> **Train the previous skill to leave a useful state for the next skill.**

These solve different problems.

| Direction | Training question |
|---|---|
| **Forward** | Can “insert” succeed from the grasp states our policy actually produces? |
| **Backward** | Can “grasp” learn to produce states from which “insert” is more likely to succeed? |

**Evidence:** Stanford’s *Sequential Dexterity* uses precisely this bidirectional idea. A learned transition-feasibility function supplies an auxiliary reward to preceding policies and helps determine switching and recovery. Its experiments concern dexterous manipulation—not RoboCasa—so the transferable contribution is the mechanism, not its benchmark score. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v229/chen23e/chen23e.pdf)

### How I would apply it

- **S1:** Fine-tune on actual handoff states, including imperfect—but recoverable—ones.
- **S2:** Train subgoal selection to consider the next required operation, not only immediate completion.
- **A small feasibility/value model:** Learn whether a specific executor can complete the next operation from the current state.

**Example**

```text
Weak target:
“Pick up the block.”

Better task-conditioned target:
“Pick up the block in a configuration that permits insertion.”
```

Initially, this can be learned through successful transition demonstrations. Later, successor-success estimates can guide reinforcement learning.

**Constraint:** This requires actual downstream outcomes or competent corrective demonstrations. A language model saying “this grasp looks suitable” is not sufficient supervision.

**Why it matters for unseen composites:** It targets reusable compatibility between skills, rather than memorizing entire task sequences.

---

# 3. Train recovery from your policy’s mistakes—not only more perfect demonstrations

### The training-distribution problem

A clean demonstration teaches:

```text
good state → good action → good state
```

Your deployed policy also encounters:

```text
slight mistake → unusual state → uncertain observation → recovery decision
```

The second sequence needs training data of its own.

**Evidence:** CMU’s *RaC* improves long-horizon behavior by collecting recovery-and-correction interventions during policy rollouts. Importantly, it keeps the imitation objective unchanged: the improvement comes from **what trajectories are collected and imitated**. Figure’s logistics work similarly retains useful corrective behavior during data curation. [arXiv](https://arxiv.org/html/2509.07953)

### What I would train

| Model | Supervision |
|---|---|
| **S1** | Correct movements from the actual failed or disturbed state |
| **S2/Jev** | Continue, retry, change strategy, reobserve, or abandon—conditioned on what actually happened |
| **Outcome critic** | Whether the continuation ultimately succeeds, including failed attempts |

**Example**

The object slips during transport. Do not only reset and record another perfect pick-and-place. Record:

```text
slip → recognize loss → find object → recover grasp → resume remaining task
```

### Important constraints

- Do not behavior-clone uncontrolled failed actions as desirable behavior.
- Preserve failures for outcome/critic learning; imitate the valid corrective portions.
- Recovery may temporarily undo progress. A reward that insists on monotonic progress can penalize the right behavior.
- Mix original competent demonstrations with corrections, rather than fine-tuning exclusively on failures.

RoboCasa’s own sequential fine-tuning study reports substantial forgetting of previously learned tasks. That is a reason to test replay of earlier skills throughout adaptation. [arXiv](https://arxiv.org/html/2603.04356v1)

**My priority:** This comes before combining OneJev and MJev weights.

---

# 4. Teach S1 reusable effects, and teach S2 how to compose them

This is the strongest route I see toward **unseen task combinations**.

### A. Make S1’s conditioning describe the required change

Compare:

```text
“Do stage 4 of task 72.”
```

with:

```text
“Move this object from the counter into that open drawer.”
```

The second exposes reusable relationships: object, source, destination, and desired effect.

**Evidence:** Xiaomi-Robotics-1 pretrains on language descriptions of state transitions, then post-trains for robot embodiments and imperative instructions. Its architecture also uses action supervision to make the VLM’s representations more useful to its action expert. This is evidence for the training formulation—not proof that a small relabeling exercise will reproduce its large-scale results. [arXiv](https://arxiv.org/html/2607.15330v1)

### B. Train the high-level policy from successful coaching

Physical Intelligence reports a particularly relevant π0.7 experiment: step-by-step language coaching improves an unfamiliar appliance task, and the successful coaching is then used to fine-tune a high-level policy that generates subtasks autonomously. [Pi](https://www.pi.website/blog/pi07)

**For your system, I would test:**

1. Keep a competent S1 fixed.
2. Obtain successful subgoal sequences on **training tasks**, using a competent teacher or operator.
3. Train the generative S2 on the observation/history → next-subgoal pairs.
4. Train OneJev on narrow acceptance, completion, and revision judgments from those same episodes.
5. Evaluate new compositions with no coaching.

### What this does—and does not—solve

- It can address an S2 that sequences known capabilities poorly.
- It cannot create a missing motor skill through better wording.
- S1 must actually support the proposed instruction granularity.
- Changing the desired goal requires a valid new action continuation. Do not relabel movements toward A as demonstrations for B.
- Synthetic compositions must respect physical and task prerequisites; concatenating unrelated demonstrations is not enough.

**Useful diagnostic:** With the same executor, compare full-task instructions against compatible oracle subgoals. A large improvement indicates a promising supervision problem; little improvement points toward execution or interface limitations.

---

# 5. Train task memory as a capability—not just attach a history string

Your plan specifies bounded history and progress, but I would make **learning to maintain and use them** a major training objective.

**Evidence:** PI’s *Multi-scale Embodied Memory* combines short-term visual history with longer-term textual memory. The model learns both what to do and what to retain. The authors also warn that memory can introduce harmful shortcuts, such as copying previous behavior instead of responding to current evidence. [Pi](https://www.pi.website/research/memory)

### What the models should learn to distinguish

```text
Attempted ≠ completed
Previously held ≠ currently held
Not visible ≠ absent
Old instruction ≠ current instruction
Current skill completed ≠ entire task completed
```

### Proposed training examples

- Same current scene, different completed steps → different next operation.
- Same instruction, different failure history → different recovery.
- Object temporarily occluded → maintain uncertainty rather than invent its location.
- Memory says “placed,” but fresh evidence shows it remains held → correct the memory.

### Division of training

- **S2:** Learn a compact progress state and its updates.
- **S1:** Learn from a short observation history where motion/contact information is necessary.
- **Jev checker:** Learn when evidence supports completion, contradicts it, or is insufficient.

**Constraint:** Adding history changes the model’s input contract. It requires training; it is not necessarily a safe inference-only modification.

For your shared-state preference, this does **not** require a writable knowledge database. The runtime can own a small record while the model learns to propose evidence-grounded updates.

---

# 6. Train Jev on the situations your system actually reaches

This is where the broader AI literature provides a particularly useful connection.

**Thinking Machines’ on-policy distillation** trains a student using supervision on the student’s own generated trajectories, rather than only polished teacher trajectories. The motivation is exposure mismatch: learners encounter mistakes and contexts absent from expert examples. Its demonstrated setting is language models. [Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)

### My proposed robotics adaptation

```text
Run the current system
        ↓
Collect its actual decision states and proposed operations
        ↓
Teacher / simulator / reviewed outcomes label those states
        ↓
Fine-tune Jev on the mistakes and ambiguities it actually encounters
```

This is closely aligned with the robotics data-aggregation principle, rather than a claim that token-level distillation transfers unchanged.

### Three separate targets—not one “confidence” score

| Target | Source of supervision |
|---|---|
| **Instruction compatibility** | Accepted instruction, task context, reviewed semantic labels |
| **Observed completion or failure** | Temporal observations and independent outcome labels |
| **Executor success from this state** | Actual rollouts of that executor |

I would not combine these into one probability.

### Make visual dependence unavoidable

Your reported inability to demonstrate camera use deserves targeted training:

- Same text, different relevant images → different correct answers.
- Same image, different instructions → different valid operations.
- Same image/instruction, different history → different progress judgments.
- Missing evidence → explicit uncertainty.

Then repeat correct-image, mismatched-image, and no-image evaluations.

**Training change:** Start with task-specific readouts/adapters and calibration. Expand trainable parameters only if those changes fail to capture the needed distinctions.

This is a more defensible use of fine-tuning than hoping a generic typed model becomes a good robot planner after a few demonstrations.

---

# 7. Do not assume full end-to-end action gradients are the best way to train S2

This is the most important correction to the **latent-interface part** of our plan.

The document proposes letting S1’s action loss update permitted S2 parameters while preserving typed/spatial losses. That remains a valid experiment—not the only recipe. Jev_Helix_Implementation_Plan

### Two credible approaches exist

| Approach | Training mechanism | Potential advantage |
|---|---|---|
| **Helix-style joint alignment** | Action gradients pass through the conditioning representation into S2 | Learn representations specifically useful for execution |
| **Protected/insulated S2** | Train S2 with appropriate semantic/action-representation objectives while blocking selected continuous-action gradients | Preserve language/visual competence and avoid disruptive action-head gradients |

Figure describes joint training. PI’s **Knowledge Insulation** instead trains its VLM using discrete-action and general vision-language objectives while stopping continuous action-expert gradients from entering the backbone. PI reports that naive continuous-action adaptation can damage training and transfer. [FigureAI](https://www.figure.ai/news/helix)

### What I would compare

- **A:** Existing text/subgoal interface.
- **B:** Learned latent receiver with S2 protected.
- **C:** Learned latent receiver with controlled joint fine-tuning.

Evaluate both composite execution **and** instruction/grounding competence.

Two constraints:

- A frozen arbitrary Jev latent is not automatically useful; the receiver still needs meaningful action supervision.
- Backpropagating through a latent adapter does not automatically teach a discrete `argmax` selector which candidate to choose. Retain explicit selection supervision or a suitable separate learning method.

**A Helix-like runtime does not obligate you to use Helix’s exact gradient routing.**

---

# 8. Train the temporal mismatch explicitly

This addresses your non-negotiable interaction requirements, although it may not be the largest contributor to an ordinary static-instruction benchmark.

**Evidence:** Helix trains with observation-time offsets between S2 and S1. Training-time RTC teaches action generation conditioned on already committed prefixes while simulating inference delay. These address different parts of asynchronous execution. [FigureAI](https://www.figure.ai/news/helix)

### A useful training sample

```text
Current images and robot state
+ older, still-valid S2 conditioning
+ actions already committed
→ correct remaining action sequence
```

Then add a separate family:

```text
Current state
+ revised instruction
+ conflicting or irrelevant advice
→ correct revised decision and physical continuation
```

**Do not confuse these cases:**

- Old conditioning under the same goal may remain usable.
- Conditioning for a superseded goal may be invalid.
- A confident new suggestion may still contradict current evidence.

I would train the fast supervisor on useful, irrelevant, and plausible-but-wrong advice. S1 should learn valid transitions after an accepted change—not obey arbitrary advisor messages.

**Constraint:** A model cannot learn the correct changed-goal trajectory from a demonstration that never performs it. Collect or generate competent continuations from those intermediate states.

---

## 9. Two additional directions worth testing

### A. Outcome-conditioned improvement, not imitation alone

PI’s **RECAP** combines demonstrations, corrections, and autonomous experience. It learns a value function and uses estimated advantages to condition improved behavior. That directly addresses learning from the quality of outcomes rather than treating every recorded action equally. [Pi](https://www.pi.website/download/pistar06.pdf)

**Your adaptation:** Once the basic executor works, train an outcome critic from successes and failures, then investigate advantage-conditioned or otherwise appropriate policy improvement.

Keep terminal task success central. A robot that repeatedly “makes progress” without completing the task should not receive a high overall return.

### B. Improve a difficult phase without retraining the whole VLA

PI’s **RLT** investigates compact representations and small RL policies for precise manipulation, including refining actions proposed by a base VLA. [Pi](https://www.pi.website/research/rlt)

**Your adaptation:** When logs show that one insertion, alignment, or release phase dominates failures, improve that phase specifically.

This may be a better use of limited training resources than updating a multi-billion-parameter S2 to compensate for a motor-control weakness. However, the representation/interface itself needs training; it is not an arbitrary Jev latent.

---

## 10. What I would actually fine-tune first

This is my recommended starting comparison—not a full delivery plan:

| Training branch | Initially train | Initially retain | Question it answers |
|---|---|---|---|
| **S1 transition/recovery branch** | Action expert or appropriate policy adapters on handoffs and corrections | Existing action representation and competent base behavior | Does execution from realistic intermediate states improve? |
| **S2 task-management branch** | Generative subgoal model/adapters, memory updates, Jev’s narrow decision readouts | A fixed competent S1 | Does better sequencing/progress/revision improve full-task success? |
| **Joint interface branch** | Conditioning projection and receiving layers; then selected encoder parameters | Text-conditioned baseline | Does learned conditioning add value beyond the first two? |

Use the pretrained model’s native action objective unless the experiment deliberately changes it. Do not change model, action format, data mixture, and scheduler simultaneously.

### The most informative outcome pattern

- **S1 training helps strongly:** Execution and handoffs were limiting the system.
- **S2 training helps strongly:** Selection, memory, or completion interpretation was limiting it.
- **Both help, but neither alone is enough:** The boundary between the two deserves joint optimization.
- **Latents do not outperform text:** Keep text; architecture elegance is not the objective.

For unseen compositions, preserve a genuine held-out task-composition split across training examples, teacher coaching, generated data, and hyperparameter selection. Train reusable capabilities on training tasks—not disguised variants of the final test tasks.

---

## What I would deprioritize

For **composite success**, I would not begin with:

- Adaptive-depth routing before the full model is accurate.
- Merging OneJev and MJev merely to remove a model boundary.
- Speculative action generation before recovery and handoffs work.
- A larger planner before establishing whether oracle subgoals help.
- More synthetic data without quality controls.

RoboCasa’s data study found that broader human-task coverage helped downstream learning, while adding its tested synthetic-data mixture did not automatically improve results. Its target-fine-tuning study is separate from zero-shot leaderboard evaluation. [arXiv](https://arxiv.org/html/2603.04356v1)

## Bottom line

**The strongest improvement hypothesis is not “a better S2 will fix S1.” It is:**

> **Train S1 to execute and recover from real handoff states; train S2 to compose reusable effects and maintain accurate task memory; train their interface without destroying either model’s existing capabilities.**

For your project, I would investigate **handoff-compatible training, rollout-based recovery, and memory-aware subgoal supervision first**. The latent Helix interface then becomes a controlled experiment on top of stronger models—not a substitute for making them stronger.

# Papers
- [Xiaomi-Robotics-0](https://arxiv.org/html/2602.12684v1)
---

# Addendum (2026-10-23): what our measurements add
- **Delay.** Train S1 with committed prefixes and delays from our measured distribution (plan S1-B). The released
  XR-1 checkpoint has no prefix input (synchronous); Xiaomi's async recipe (XR-0) is the reference. S2 needs no delay
  training in route A; Helix-style S1/S2 offset training only for the latent route. Evaluate in lockstep now
  (`hardware-port.md`), real time on hardware.
- **Completion.** Train a completion / progress output (Helix's "percentage task completion" action; plan
  `L_progress`), e.g. a Jev auxiliary head on sim-labelled frames. Budgets / watchdogs are a safety stop, not this.
- **Memory.** Train it in (Helix vision memory + state history; PI MEM); never feed untrained history.
- **Data integrity.** Training data must pass `model-bringup/checklist.md`; live data recorded before the
  2026-10-22 render fix is corrupted.
- **Corrections to the text above.** E13 was corrected on 2026-10-19 and does not show Jev is a weak planner (the
  v1 spec already makes Jev an interaction layer). §7's Knowledge Insulation link points to Figure, not PI.
