# RRF_K sweep — negative result

## Question

Does the RRF fusion constant `RRF_K` change retrieval quality enough to tune it? If the
hybrid ranking can be improved by a constant, it is the cheapest possible win.

## Setup

- **What changed:** `RRF_K` scanned over `{5, 10, 20, 40, 60, 90, 120, 200}` via
  `eval/tune_retrieval.py` on the `simple` path.
- **What stayed fixed:** everything else — top-k, reranker, glossary, query plan.
- **Datasets:** 43 held-out questions (`golden_retrieval_labeled.jsonl`) and 262 synthetic
  questions (`retrieval_gt_reviewed.jsonl`).
- **Runs:** one run per value; the script flags spreads within one-question noise
  (`NOISE_SPREAD`).
- **Artifacts:** `benchmarks/tune_rrf_k_simple_2026-09-04.json` (gitignored); table in
  [roadmap](../../roadmap.md) step 3.

## Results

| Dataset | n | HR@5 | HR@10 | HR@12 | MRR |
|---|---:|---:|---:|---:|---|
| held-out | 43 | 0.814 | 0.860 | 0.884 | 0.646 at k=5; 0.626 at k=20–200 |
| synthetic | 262 | 0.870 | 0.924 | 0.939 | 0.716 at k=5; 0.713 at k=10–200 |

Hit Rate is **identical at every value on both datasets**. Only the order inside the same
result list changes, and only at small `k`. The MRR spread (0.020 held-out, 0.003 synthetic)
is comparable to the weight of a single question.

## Interpretation

The knob is dead. With only two lists (vector and BM25) that already agree, the fusion
constant cannot reorder the top of the ranking. The real bottleneck is the ~12% of questions
whose chunk is not found at any `k` — a coverage problem, not a fusion problem.

## Decision

Keep `RRF_K = 60`. Rejected: any change to the constant. This is a measured negative result,
not an untested default.

## Threats to validity

The held-out set is small (43) and the synthetic set rewards lexical matching (see
[retrieval-backbones](./retrieval-backbones.md)); a spread of one question is noise on both.

## Next action

Stop tuning fusion and analyze the misses by name. That analysis later found the `complex`
candidate-pool bug and several chunking defects — see
[triage-gap-terminal-contract](./triage-gap-terminal-contract.md) and
[roadmap](../../roadmap.md) step 2a.