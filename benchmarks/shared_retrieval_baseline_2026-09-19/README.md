# Shared retrieval baseline — 2026-09-19

Baseline commit: `2cb34ee7644fc7ea31741f3f46da35dfcb4fe284`.

This directory fixes the starting point for issues #55–56. No index was rebuilt,
no environment file was changed, and no paid embedding, generation, or evaluation
call was made while capturing it.

## Reproduced gates

- Baseline before feature code: 798 unit tests passed, 469 deselected, 5 warnings.
- Candidate after PR 1 implementation: 821 unit tests passed, 470 deselected,
  5 warnings.
- `scripts/check_docs.py --ci`: OK before and after the change.
- Dedicated shared-retrieval checks: 24 passed, including an integration check
  with two disposable local Chroma collections.

The exact non-secret configuration, package versions, prompt/expectation/profile
hashes, local index file hashes, and collection identities are in `snapshot.json`.
Index hashes are a read-only file inventory rather than an atomic database backup.

## Generic characterization

`tests/v7/fixtures/shared_retrieval_baseline.json` records six deterministic
backend cases from the baseline implementation:

1. simple accepted;
2. simple escalates and complex is accepted;
3. complex fails and the packed simple snapshot is accepted;
4. successful empty retrieval;
5. retrieval exception;
6. router clarification.

Each expected result fixes the terminal route/reason, ordered final passages,
scores, attempt stages, rejected-candidate order, and technical-failure flag.
The candidate retrieval-only graph must match these signatures while invoking no
generator. These are algorithm characterization fixtures; they do not claim live
retrieval or answer quality.

## Department V2 baseline

The last compatible saved run is
`eval/runs/object_profile_traps_2026-09-17_deepseek-v4.1-flash/v2/config.json`:
V2 collection `department_demo_v2`, 1,115 chunks, prompt v4, four trap questions,
and the profile/expectation hashes copied into `snapshot.json`. That run had the
experimental verifier enabled, so it is provenance evidence rather than the final
control for the new `--no-verify` comparison.

Before Department cutover, compare baseline and candidate on the same read-only V2
collection, model, prompt, profiles, expectations, and `--no-verify` setting. Fix
the paid-run question list and budget before starting it.

## Scoped and latency comparison still required

PR 2 adds immutable scoped cases for external, company internal, unit A/unit B,
single-law, single-LNA, empty branch, one-branch complex, colliding chunk IDs,
profile on/off, and missing object facts.

PRs 3–4 compare the old Department retriever with the shared runtime in two modes:
workers=1 and workers=2. Use identical code/config/data; record cold and warm runs,
single and dual corpus, queue/reranker wait, retrieval wall time, and end-to-end
p50/p95 with sample sizes. Do not attribute policy changes to parallel execution.
