# LangSmith v8, stage 1: v7 baseline without judge

Date: 2026-10-15. First experiment on the `regrag-golden` dataset, code evaluators only.

| | |
|---|---|
| Experiment | `v7-low-451e31af` |
| Command | `LANGSMITH_TRACING_V2=true python -m eval.ls.run --split verified --prefix v7-low --skip-judge` |
| git sha | `4d19ed6` (branch `feat/v8-langsmith-eval`) |
| effort | `low` (`OPENROUTER_REASONING_EFFORT`) |
| Examples | 24 verified: 14 in scope, 10 OOS (3 `false_premise`, 7 `out_of_scope`) |
| Errors | 0 of 24 |

## Results

| Key | Score | n |
|---|---|---|
| `retrieval_hit` | 1.00 | 14 |
| `doc_cited` | 1.00 | 14 |
| `oos_handled` | 1.00 | 10 |
| `forbidden` | not measured | 0 (`must_not_contain` is empty in all 24) |
| MRR | 0.626 | 14 |

Hit rate and `doc_cited` are at the ceiling, so they cannot separate variants on this set.
MRR is the informative number.

First relevant chunk rank, in-scope examples:

| Rank | Cases |
|---|---|
| 1 | G006, G002, G015, G028, G017, G013 |
| 2 | G020, G009, G008, G010 |
| 4 | G001, G052 |
| 6 | G019 |
| 11 | G022 |

No misses. The weak spots are G022 (rank 11), G019 (rank 6), G001 and G052 (rank 4).

OOS handling: 7 refusals by text (`refusal`), 3 empty answers from the domain gate (`empty`:
G035, G036, G039), none through `route_decision == "abstain"`.

## Cost and latency

- Latency per example (root run): p50 9.9 s, p90 22.2 s, max 36.3 s. Whole run 5 min 02 s,
  `max_concurrency=1`.
- Tokens: 161 112 in total, from the LangSmith traces.
- Dollar cost: **not available.** LangSmith reports `total_cost = 0` because the model
  (`deepseek/deepseek-v4.1-flash` via OpenRouter) has no price in its table. Take the dollar
  figure from the OpenRouter dashboard or from `eval/pricing.py`, not from the trace.

## Caveats

- All 24 examples carry `corpus_support=unverified`, so there is no "supported by corpus"
  subset to separate corpus gaps from retrieval failures.
- `visual_enrichment` timed out (>3 s) on passages 3, 5, 6 of some questions. It does not
  change the scores here, but it adds latency noise.
- 14 in-scope examples is a small sample: one example moves a rate by 7 points.
- No judge yet: this run says nothing about answer correctness or citation quality
  (tasks 4 and 4a).
