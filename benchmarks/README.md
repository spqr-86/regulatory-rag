# Benchmarks

Current baseline metrics and local eval artifacts.

**In git:** this README only.
**Local only:** `eval_v7_*.jsonl`, `retrieval_*_*.json`, `triage_gap_*.json`, `cps_*.json` —
eval run artifacts, listed in `.gitignore`.

## Current baseline — 2026-09-08

Generation eval (`eval/run_v7_eval.py`, judge `gpt-4o`, `tests/dataset.csv` 56 questions,
53 valid) — first full run after the 2026-09-02 `llm_factory` judge fix:

| Metric | Value | Earlier (mini judge, 2026-05-30) |
|--------|-------|----------------------------------|
| In-scope correctness (0–10) | **7.56** | 7.44 |
| Correctness, all questions | **7.30** | 7.39 |
| Faithfulness (0–1) | **0.840** | 0.859 |
| Answer relevance (0–1) | **0.847** | 0.872 |
| OOS abstain rate | **1.00** | 1.00 |
| False-sufficiency rate | **4.8%** | 9.8% |
| Complex-path rate | 20.8% | 24.1% |
| Latency p50 / p95 / mean | 5.2 / 19.1 / 7.2 s | — / — / 9.71 |
| Cost per query | $0.0045 ($0.24 run total) | — |

**Config:** V7 LangGraph, OpenAI `text-embedding-3-small`, `gpt-4o-mini` (simple) /
`gpt-4o` (complex), `V7_V8_ENABLE_EVIDENCE_ASSESS=true`, CrossEncoder reranker (cap 100),
HybridChunker (`max_tokens=400`), 12 documents, dataset 56 questions.

Retrieval eval (`eval/run_retrieval_eval.py`, held-out 133, 2026-09-08):

| path | HR@5 | HR@12 | MRR | p50 latency |
|------|------|-------|-----|-------------|
| hybrid (production) | 0.692 | 0.827 | 0.542 | 502 ms |
| vector-only | — | 0.842 | 0.539 | 141 ms |
| bm25-only | — | 0.677 | 0.387 | 19 ms |

> The `gpt-4o` judge is stricter than the earlier `gpt-4o-mini` — absolute faithfulness /
> relevance sit slightly below the historical numbers while being better calibrated.
> Run-to-run judge variance is ~±0.25 on the correctness scale. Compare runs only under
> the same judge. Canonical values: [docs/reference/FACTS.md](../docs/reference/FACTS.md).

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
