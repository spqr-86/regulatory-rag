# Benchmarks

Current baseline metrics and local eval artifacts.

**In git:** this README only.
**Local only:** `eval_v7_*.jsonl`, `retrieval_*_*.json`, `triage_gap_*.json`, `cps_*.json` —
eval run artifacts, listed in `.gitignore`.

## Current baseline — 2026-09-23

Generation eval (`eval/run_v7_eval.py`, judge `gpt-4o`, `tests/dataset.csv` 56 questions,
53 valid) — current default (`deepseek/deepseek-v4.1-flash` on both paths, reasoning effort `low`):

| Metric | Value | 17.09 showcase run | Terminal contract (2026-09-11) | Pre-contract baseline (2026-09-08) |
|--------|-------|--------------------|---------------------------------|------------------------------------|
| In-scope correctness (0–10) | **8.09** | 7.91 | 7.47 | 7.40 |
| Correctness, all questions | **8.32** | 7.98 | 7.26 | 7.09 |
| Faithfulness (0–1) | **0.974** | 0.926 | 0.891 | 0.808 |
| Answer relevance (0–1) | **0.853** | 0.887 | 0.879 | 0.881 |
| OOS abstain rate | **1.00** | 1.00 | 1.00 | 1.00 |
| False-sufficiency rate | **7.1%** | 10.0% | 11.4% | 13.0% |
| Complex-path rate | 20.8% | 24.5% | 17.0% | 13.2% |
| Latency p50 / p95 / mean | 9.5 / 25.9 / 12.0 s | 24.0 / 71.3 / 30.95 s | 4.51 / 15.70 / 6.83 s | 4.8 / 14.7 / 5.9 s |
| Cost per query | $0.0021 ($0.111 run total) | $0.00657 ($0.348) | $0.00387 ($0.205 run total) | $0.0033 |

**Config:** V7 LangGraph, OpenAI `text-embedding-3-small`, `deepseek/deepseek-v4.1-flash` on
both paths via OpenRouter with `OPENROUTER_REASONING_EFFORT=low`, single hard-gate triage with a
terminal route contract, CrossEncoder reranker (cap 100), HybridChunker (`max_tokens=400`),
12 documents, dataset 56 questions. Cost is the run's token usage priced against
`src/pricing.py::PRICE_PER_1M`. The 17.09 run used `gpt-4o` on the complex path and the
provider's default (high) reasoning effort. Details:
[golden-set-effort-low](../docs/evaluation/experiments/golden-set-effort-low.md),
[showcase-default-golden-set](../docs/evaluation/experiments/showcase-default-golden-set.md).

Retrieval eval (`eval/run_retrieval_eval.py`, held-out 133, 2026-09-08):

| path | HR@5 | HR@12 | MRR | p50 latency |
|------|------|-------|-----|-------------|
| hybrid (production) | 0.692 | 0.827 | 0.542 | 502 ms |
| vector-only | — | 0.842 | 0.539 | 141 ms |
| bm25-only | — | 0.677 | 0.387 | 19 ms |

> The showcase default clears the >7.5 in-scope correctness target for the first time, at a
> ~5× latency regression (unexplained) and a real cost above the terminal-contract baseline
> once DeepSeek is actually priced. Compare runs only under the same judge. Canonical values:
> [docs/reference/FACTS.md](../docs/reference/FACTS.md).

## Running eval

```bash
python eval/run_v7_eval.py --skip-judge --output benchmarks/eval_v7_$(date +%F).jsonl  # pipeline only, ~$0
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl               # full, gpt-4o judge, ~$0.25
```

Retrieval IR metrics. `--path simple` is the production hybrid (vector + BM25 → RRF);
`vector` and `bm25` are the single backbones; `complex` is vector → rerank → MMR.

```bash
# headline test set: 90 practitioner questions (docs/reference/FACTS.md)
python eval/run_retrieval_eval.py --path simple --gt eval/data/golden_retrieval_labeled_ext.jsonl
python eval/run_retrieval_eval.py --path vector --gt eval/data/golden_retrieval_labeled_ext.jsonl
python eval/run_retrieval_eval.py --path bm25   --gt eval/data/golden_retrieval_labeled_ext.jsonl

# development set (default --gt eval/data/retrieval_gt_reviewed.jsonl)
python eval/run_retrieval_eval.py --path simple

# smoke check before a full run
python eval/run_retrieval_eval.py --path simple --gt eval/data/golden_retrieval_labeled_ext.jsonl --limit 3
```

Reproduction requires the indexed corpus: the source documents (`source_docs/`) and the
ChromaDB index are not in git — see [getting started](../docs/getting-started.md) and the
document list in [FACTS](../docs/reference/FACTS.md#corpus). Query embeddings call the
configured provider (OpenAI by default, a fraction of a cent per run). Raw run outputs
(`benchmarks/*.json`, `benchmarks/*.jsonl`) are gitignored; the tables above are copied
from them manually.

## Updating the baseline

After a meaningful change, re-run and update the tables above manually, then commit.

---

[Русская версия](README_RU.md)
