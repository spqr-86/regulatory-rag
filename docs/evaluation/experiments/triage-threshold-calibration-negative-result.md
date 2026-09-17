# Triage threshold calibration — negative result (issue #9)

## Question

Can a different set of hard-gate thresholds reduce **unsafe generations** (the fast path
answering when it should have escalated or abstained) without losing coverage?

## Setup

- **What changed:** threshold profiles across the hard gates (`top_score`, `passage_count`,
  `keyword_overlap`), evaluated as a grid.
- **What stayed fixed:** the retrieval snapshot (immutable), the reviewed questions, the
  routing logic.
- **Dataset:** 43 reviewed questions (`eval/data/triage_calibration_dev_annotations.json`),
  annotated for evidence coverage and critical misses.
- **Runs:** one pass over **81 profiles**; no LLM judge, no API cost.
- **Runner:** `eval/triage_calibration.py`.
- **Artifacts:** plan and result —
  [`2026-09-11-triage-threshold-calibration.md`](../../superpowers/plans/2026-09-11-triage-threshold-calibration.md);
  [roadmap](../../roadmap.md).

## Results

No profile in the grid reduced unsafe generations relative to the current defaults. The
defaults were retained. After the terminal route contract,
`TRIAGE_SOFT_THRESHOLD` no longer controls routing at all — it only distinguishes the
diagnostic reason `triage_borderline` from `triage_clearly_bad`, and both lead to the complex
path.

## Interpretation

The hard-gate thresholds are not where unsafe generations come from. Moving them trades one
error class for another rather than shrinking the total; the lever is on the generation side
(contract, prompt, model), not in the numeric gate.

## Decision

Keep the default thresholds. Closed as a **measured negative result**, not as an unfinished
tuning step.

## Threats to validity

Development set of 43 questions; annotations encode one reviewer's judgment of what evidence
is sufficient. No generation quality is measured here, only routing outcomes.

## Next action

Attack unsafe generations where they are produced: the terminal route contract and the
generation prompt. See [triage-gap-terminal-contract](./triage-gap-terminal-contract.md).