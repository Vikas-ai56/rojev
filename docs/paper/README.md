# Paper workspace: OneJev as a typed probabilistic supervisor over a frozen VLA

Single place for everything paper-relevant. Engineering-only notes stay in `docs/features/`; decisions in `docs/decisions.md`.

## Files
| File | Contents |
|---|---|
| `related-work.md` | **§0 key transfer-learning findings first**; then literature positions, verified numbers, concurrent work |
| `method.md` | System design: 3 levels, supervisor loop, candidate builder, control options, timing variants, arms, metrics |
| `experiments.md` | Every experiment run so far, with exact numbers and run directories; invalid runs flagged |
| `appendix-engineering.md` | Reproducibility: executor I/O contracts and eval pitfalls found (build notes → `docs/features/v2-engineering-notes.md`) |
| `references.bib` | BibTeX for every source (arXiv entries generated from the arXiv API) |
| `citations.md` | Index: key → source → used for |
| `sources/` | PDFs of key papers |

## Working outline
- **Title (draft):** Typed probabilistic supervision of frozen vision-language-action policies.
- **Research question:** Does an image-conditioned typed decision model (OneJev), selecting subtasks for a frozen steerable VLA, improve task completion, recovery and uncertainty handling vs the VLA's own planner and a generative VLM planner, with identical low-level weights?
- **Claimed contributions (to be supported by experiments):**
  1. A closed-loop typed supervisor: MJev spatial evidence → OneJev subtask decisions, with calibrated control options (continue/retry/reobserve/done/target-missing/abort) and a risk–coverage threshold analysis.
  2. A deterministic, data-derived candidate builder over a 21k-subtask vocabulary (coverage@K measured).
  3. A controlled comparison on identical executor weights: own planner vs generative VLM vs OneJev vs oracle.
  4. A timing study (paused / latency-injected / async), filling a gap no paper addresses directly (related-work §4).
- **Explicitly not claimed:** new motor skills, cross-domain transfer (related-work §1).

## Status (2026-10-17)
| Item | State |
|---|---|
| Literature review + verified bibliography | done (29 curated entries) |
| Executor selection evidence (SmolVLA weak, eval bugs found) | done (E7) |
| π0.5² subtask vocabulary + ordered timelines | done (E8, E9 data) |
| Candidate-builder coverage@K | E9 exact-string builder in use; E9b templates measured (partial), not adopted; parked until the pipeline works |
| π0.5² language-following | no separate test; measured inside the supervisor workflow (decision 2026-10-17) |
| Supervisor arms A–E | not started |
