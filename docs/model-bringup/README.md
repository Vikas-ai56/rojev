# Model bring-up: putting a new or retrained model behind the runtime

Read before plugging any S1 / S2 model in (third-party checkpoint or our own training). Written after 2026-10-22, when a
day of S1 conclusions turned out to rest on corrupted camera images (`decisions.md`, render-context entry).

| Doc | Use it for |
|---|---|
| [`checklist.md`](checklist.md) | The gates, in order; each item says what broke when it was skipped and how to check it |
| [`search-directions.md`](search-directions.md) | Where to look when a number looks wrong or a model underperforms: sources, instruments, bisection |
| [`groot.md`](groot.md) | GR00T N1.7 on the SO-101: contract from source, our adapter, what is not validated |
| [`s2-bakeoff.md`](s2-bakeoff.md) | Embodied-R1.5 as planner and done checker on real SO-101 data: case sets, run, scoring, limits |

Rules of thumb
- Reproduce the model's published numbers on its own test cases, in our harness, before any of our own metrics.
- Look at what the model sees (frames) and does (video) in the first run; one look beats hours of statistics.
- A result is valid only if the gates it depends on passed for that run; a failed gate withdraws the result.
- Model-specific needs (history, crop, prompt, normalization) live in the model's executor / service, never in `rojev/`.
