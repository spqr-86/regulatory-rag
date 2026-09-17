# Retrieval backbone: vector vs BM25 vs hybrid

## Question

Which retrieval backbone should production use? The decision was made on measured data
rather than "hybrid, because best practice", after arXiv 2607.26497 (30.07.2026) reported
lexical search beating dense and agentic retrieval on corpora >10M tokens (issue #12).

## Setup

- **What changed:** nothing in production. Only the measurement was added:
  `eval/run_retrieval_eval.py --path vector|bm25|simple`, where `simple` is the production
  hybrid (vector + BM25 → RRF). Backbones bypass RRF and reranking and call the retriever
  directly.
- **What stayed fixed:** the frozen retrieval config and the router's plan (top-k, glossary)
  exactly as in production; no LLM generation, no triage.
- **Datasets:** 90 practitioner questions (headline), 133-question development set; earlier
  sweeps also used 43 held-out and 262 synthetic questions.
- **Runs:** one run per backbone per dataset.
- **Metrics:** Hit Rate@5/10/12, MRR, p50/p95 latency.
- **Artifacts:** `benchmarks/retrieval_ext_*`, `retrieval_backbone_*`, `retrieval_latency_*`
  (gitignored); tables copied into [roadmap](../../roadmap.md) and
  [FACTS](../../reference/FACTS.md).

## Results

90 practitioner questions (headline):

| Backbone | HR@5 | HR@12 | MRR | p50 |
|---|---:|---:|---:|---:|
| hybrid (production) | 0.633 | 0.811 | 0.503 | 550 ms |
| vector-only | 0.589 | 0.822 | 0.486 | 148 ms |
| bm25-only | 0.500 | 0.667 | 0.352 | 24 ms |

133-question development set, with latency:

| Backbone | HR@12 | MRR | p50 | p95 |
|---|---:|---:|---:|---:|
| bm25-only | 0.677 | 0.387 | 19 ms | 52 ms |
| vector-only | 0.842 | 0.539 | 141 ms | 167 ms |
| hybrid (simple) | 0.827 | 0.545 | 502 ms | 872 ms |
| complex | 0.887 | 0.595 | 6935 ms | 11178 ms |

The 262-question synthetic set gives the **opposite** ranking (vector 0.813, BM25 0.935,
hybrid 0.939): its questions repeat the source chunk's wording, so lexical search wins by
construction.

## Interpretation

- **BM25-only is disqualified** — 15+ points behind on HR@12 on both real sets. The arXiv
  result does not transfer to this smaller corpus with real question wording.
- **Vector vs hybrid is a near-tie.** Hybrid leads HR@5 (+5.3 pp) and MRR — the top of the
  list that feeds reranking; vector leads HR@12 (+3.8 / +1.5 pp) and is ~3.5× faster at p50.
- **The synthetic set is unfit as a tuning target for the retriever.** It measures lexical
  overlap, not retrieval quality. This is why the headline moved to practitioner questions.

## Decision

Keep the hybrid backbone; `V7Config` unchanged. Vector-only is a legitimate Pareto
alternative (better HR@12 and p50) at ~1.5 pp HR@12 cost, and is the documented lighter
fallback if BM25-index maintenance becomes a factor.

## Threats to validity

Pooling bias (labels chosen from the pools of both systems; complex reached HR@12 1.000 on
the earlier 43-question held-out set — evidence of bias, not quality). The 133 set is a
development set, not an untouched final test. One run per backbone; complex p95 is driven by
a few deep cross-reference queries.

## Next action

Break results down by question type — parked until the practitioner set grows past ~150
questions.