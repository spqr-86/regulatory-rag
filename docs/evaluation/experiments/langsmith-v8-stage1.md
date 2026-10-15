# LangSmith v8, stage 1: v7 baseline without judge

Date: 2026-10-15. First experiments on the `regrag-golden` dataset, code evaluators only.

The same configuration was run twice. Run 2 is the baseline (it also reports cost); run 1 is
kept to show run-to-run noise.

| | Run 1 | Run 2 (baseline) |
|---|---|---|
| Experiment | `v7-low-451e31af` | `v7-low-d081d82b` |
| git sha | `4d19ed6` | `3e3133f` |
| Cost reported | no | yes |

Command: `LANGSMITH_TRACING_V2=true python -m eval.ls.run --split verified --prefix v7-low --skip-judge`.
Branch `feat/v8-langsmith-eval`, effort `low` (`OPENROUTER_REASONING_EFFORT`).
Examples: 24 verified, 14 in scope and 10 OOS (3 `false_premise`, 7 `out_of_scope`). No errors in either run.

## Results

| Key | Run 1 | Run 2 | n |
|---|---|---|---|
| `retrieval_hit` | 1.00 | 1.00 | 14 |
| `doc_cited` | 1.00 | 1.00 | 14 |
| `oos_handled` | 1.00 | 1.00 | 10 |
| `forbidden` | not measured | not measured | 0 (`must_not_contain` is empty in all 24) |
| MRR | 0.626 | 0.647 | 14 |

Hit rate and `doc_cited` are at the ceiling, so they cannot separate variants on this set.
MRR is the informative number.

## Run-to-run noise

Same code and settings, 7 of 14 in-scope examples changed rank between the runs:

| Case | Run 1 | Run 2 |
|---|---|---|
| G001 | 4 | 12 |
| G022 | 11 | 9 |
| G019 | 6 | 6 |
| G052 | 4 | 5 |
| G009 | 2 | 1 |
| G010 | 2 | 1 |
| G028 | 1 | 2 |
| other 7 (G002, G006, G008, G013, G015, G017, G020) | same | same |

A single example can move by up to 8 places, while aggregate MRR moved by 0.02. Likely cause
is the query expansion by the LLM, not checked. Consequences: do not read a per-example rank
change as a regression, and a difference between two system variants smaller than roughly
0.05 MRR on this set is not distinguishable from noise (two runs only, so this is a rough
bound).

Stable weak spots: G019 (rank 6 in both), G022 (11 and 9), G001 (4 and 12), G052 (4 and 5).
Misses: none in either run.

## Cost and latency

| | Run 1 | Run 2 |
|---|---|---|
| Latency p50 | 9.9 s | 8.4 s |
| Latency p90 | 22.2 s | 14.0 s |
| Latency max | 36.3 s | 33.5 s |
| Tokens (LangSmith traces) | 161 112 | 160 598 |
| Cost (project rate card) | not computed | **$0.0368** total, $0.0015 per example |

Run time was about 4 to 5 minutes at `max_concurrency=1`.

Cost comes from the token usage in the graph state, priced with `src/pricing.py` (the same rate
card as the v7 runner), because LangSmith shows `total_cost = 0` for `deepseek/deepseek-v4.1-flash`
via OpenRouter. No unpriced models in run 2.

## Caveats

- All 24 examples carry `corpus_support=unverified`, so there is no "supported by corpus"
  subset to separate corpus gaps from retrieval failures.
- `visual_enrichment` timed out (>3 s) on some passages. It does not change the scores here,
  but it adds latency noise.
- 14 in-scope examples is a small sample: one example moves a rate by 7 points.
- No judge yet: these runs say nothing about answer correctness or citation quality
  (tasks 4 and 4a).
