# Benchmarks

Current baseline metrics and local eval artifacts.

**In git:** this README only.
**Local only:** `eval_v7_*.jsonl`, `retrieval_*_*.json`, `triage_gap_*.json`, `cps_*.json` —
eval run artifacts, listed in `.gitignore`.

## Current baseline — 2026-09-11

Generation eval (`eval/run_v7_eval.py`, judge `gpt-4o`, `tests/dataset.csv` 56 questions,
53 valid) — final terminal triage contract:

| Metric | Value | Pre-contract baseline (2026-09-08) |
|--------|-------|------------------------------------|
| In-scope correctness (0–10) | **7.47** | 7.40 |
| Correctness, all questions | **7.26** | 7.09 |
| Faithfulness (0–1) | **0.891** | 0.808 |
| Answer relevance (0–1) | **0.879** | 0.881 |
| OOS abstain rate | **1.00** | 1.00 |
| False-sufficiency rate | **11.4%** | 13.0% |
| Complex-path rate | 17.0% | 13.2% |
| Latency p50 / p95 / mean | 4.51 / 15.70 / 6.83 s | 4.8 / 14.7 / 5.9 s |
| Cost per query | $0.00387 ($0.205 run total) | $0.0033 |

**Config:** V7 LangGraph, OpenAI `text-embedding-3-small`, `gpt-4o-mini` (simple) /
`gpt-4o` (complex), single hard-gate triage with a terminal route contract, CrossEncoder
reranker (cap 100), HybridChunker (`max_tokens=400`), 12 documents, dataset 56 questions.

Retrieval eval (`eval/run_retrieval_eval.py`, held-out 133, 2026-09-08):

| path | HR@5 | HR@12 | MRR | p50 latency |
|------|------|-------|-----|-------------|
| hybrid (production) | 0.692 | 0.827 | 0.542 | 502 ms |
| vector-only | — | 0.842 | 0.539 | 141 ms |
| bm25-only | — | 0.677 | 0.387 | 19 ms |

> The terminal contract improved faithfulness and false sufficiency without moving
> correctness or relevance beyond the judge's ~±0.25 run-to-run variance. Compare runs
> only under the same judge. Canonical values:
> [docs/reference/FACTS.md](../docs/reference/FACTS.md).

## Running eval

```bash
python eval/run_v7_eval.py --skip-judge --output benchmarks/eval_v7_$(date +%F).jsonl  # pipeline only, ~$0
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl               # full, gpt-4o judge, ~$0.25
python eval/run_retrieval_eval.py --path hybrid                                        # retrieval IR metrics
```

## Updating the baseline

After a meaningful change, re-run and update the tables above manually, then commit.

---

[Русская версия](README_RU.md)
