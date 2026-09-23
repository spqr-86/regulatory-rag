# Evaluation report

How the system is measured, what the numbers are, what was rejected, and where the
evidence is weak. Current production values live in
[FACTS](../reference/FACTS.md); this report is the interpretation layer on top of them.

- **Datasets and their provenance:** [datasets.md](./datasets.md)
- **Per-experiment decision memos:** [experiments/](./experiments/)
- **How to reproduce a run:** [how-to/run-evaluation.md](../how-to/run-evaluation.md)
- **Metric definitions and report format:** [reference/evaluation.md](../reference/evaluation.md)

## What is measured

Retrieval and generation are measured **separately**, because they fail differently and are
fixed in different places. A single end-to-end score hides whether a wrong answer came from
a missing chunk or from a bad decision over a good chunk.

| Layer | Question | Dataset | Metrics | Runner |
|---|---|---|---|---|
| Generation | Is the answer correct, grounded, and does it refuse when it should? | 56-question golden set | correctness, faithfulness, relevance, OOS abstain, false-sufficiency, cost, latency | `eval/run_v7_eval.py` |
| Retrieval | Is the right chunk in the top-k? | 90 practitioner questions (headline), 133 dev questions | Hit Rate@k, MRR, p50/p95 latency | `eval/run_retrieval_eval.py` |
| Routing | Are escalate/answer decisions safe? | 43 reviewed questions, no judge | escalation rate, coverage, critical misses | `eval/triage_calibration.py` |
| Unit-scoped Q&A | Are object facts applied correctly, or is a norm's threshold used on the wrong quantity? | 9-question pair set, 4-question trap set | expected status match, required sub-answers, forbidden conclusions | `eval/run_object_profile_pair.py` + `eval/score_object_profile_run.py` |

Pipeline cost is measured per query from provider token usage (`src/v7/usage.py`,
`eval/pricing.py`), not estimated from a constant, and is reported next to quality.

## Headline results

### Generation — 56-question golden set, judge `gpt-4o`

Current default (23.09.2026: `deepseek/deepseek-v4.1-flash` on both paths, reasoning effort
`low`) against the 17.09 showcase run (effort high, `gpt-4o` complex path), the terminal
triage contract (2026-09-11) and the pre-contract baseline (2026-09-08), same judge, 53/56
valid:

| Metric | Current default | 17.09 showcase | Terminal contract | Pre-contract baseline |
|---|---:|---:|---:|---:|
| In-scope correctness (0–10) | **8.09** | 7.91 | 7.47 | 7.40 |
| Correctness, all questions | **8.32** | 7.98 | 7.26 | 7.09 |
| Faithfulness (0–1) | **0.974** | 0.926 | 0.891 | 0.808 |
| Answer relevance (0–1) | 0.853 | 0.887 | 0.879 | 0.881 |
| OOS abstain rate | **1.00** | 1.00 | 1.00 | 1.00 |
| False-sufficiency rate | **7.1%** | 10.0% | 11.4% | 13.0% |
| Complex-path rate | 20.8% | 24.5% | 17.0% | 13.2% |
| Latency p50 / p95 (s) | 9.5 / 25.9 | 24.0 / 71.3 | 4.51 / 15.70 | 4.8 / 14.7 |
| Cost per query | $0.0021 | $0.00657 | $0.00387 | $0.0033 |

Every target in [reference/evaluation.md](../reference/evaluation.md) is met, relevance only
just (0.853 vs >0.85). The reasoning-effort fix
([#63](https://github.com/spqr-86/regulatory-rag/pull/63)) removed most of the 17.09 latency
regression and cut cost to a third without losing correctness or faithfulness; relevance fell
(in-scope 0.953 → 0.912), and latency is still ~2× the GPT-4o-mini baseline. Details:
[experiments/golden-set-effort-low.md](./experiments/golden-set-effort-low.md); the 17.09 run
and the pricing fix behind its cost:
[experiments/showcase-default-golden-set.md](./experiments/showcase-default-golden-set.md).

### Retrieval — 90 practitioner questions (headline)

The 90-question set is the honest headline: real question wording, independent of the
retrieval config, frozen before any tuning ([datasets.md](./datasets.md)).

| Backbone | HR@5 | HR@12 | MRR | p50 |
|---|---:|---:|---:|---:|
| hybrid (production) | 0.633 | 0.811 | 0.503 | 550 ms |
| vector-only | 0.589 | 0.822 | 0.486 | 148 ms |
| bm25-only | 0.500 | 0.667 | 0.352 | 24 ms |

~19% of questions are not retrieved in top-12 by any path. The combined 133-question
development set scores higher (hybrid HR@12 0.827) because 43 of its questions are
chunk-derived and easier — see the threat note below.

### Unit-scoped Q&A — object-profile mode

Mode `v2` (object sheet passed whole as a profile, not retrieved) is the default. On the
pre-registered 9-question pair run (`gpt-4o-mini`, temperature 0):

| Metric | v1 (sheet in index) | v2 (profile) |
|---|---:|---:|
| Expected status + reasons | 7/9 | 8/9 |
| Required sub-answers credited | 7/17 | 11/17 |
| Forbidden conclusions | 2 | 1 |

The remaining `v2` error is `q1`: a norm threshold applied to a fact of a different
quantity. It survived prompt v3, a typed object sheet, and a post-generation verifier, and
was only fixed by the model — see
[experiments/department-qa-object-profile.md](./experiments/department-qa-object-profile.md).

### Cheap-model selection on the adversarial trap set

Four pre-registered trap questions on norms with thresholds (area ≤ 100 m², distance ≤ 30 m,
5-year interval). A model passes only if it does not apply a threshold to the wrong
magnitude or invert the rule:

| Model | Traps passed | Cost, 4 questions |
|---|---:|---:|
| `deepseek/deepseek-v4.1-flash` | **4/4** | $0.022 |
| `openai/gpt-5-mini` | 4/4 | $0.049 |
| `google/gemini-3-flash-preview` | 4/4 | $0.030 |
| `openai/gpt-4o-mini` | 2/4 | $0.004 |
| `anthropic/claude-haiku-4.5` | 2/4 | $0.057 |
| `deepseek/deepseek-v4-flash` | 1–1.5/4 | $0.003 |

DeepSeek V4.1 Flash is the cheapest model that passed all traps and is the showcase default
for the simple path. The contract status did not separate correct from wrong answers here —
model selection was done on semantics, not on the validator. Details:
[experiments/cheap-model-selection.md](./experiments/cheap-model-selection.md).

## Decisions backed by measurement

| Question | Result | Decision |
|---|---|---|
| Which retrieval backbone? | bm25 alone is 15+ points behind; vector vs hybrid is near-tie, hybrid wins top-5/MRR, vector wins HR@12 and 3.5× latency | keep hybrid ([memo](./experiments/retrieval-backbones.md)) |
| Tune `RRF_K`? | Hit Rate flat across {5…200}; MRR moves within one-question noise | keep 60; dead knob, negative result ([memo](./experiments/rrf-k-negative-result.md)) |
| Can a better threshold profile cut unsafe answers? | 81 profiles, none reduced unsafe generations | keep defaults ([memo](./experiments/triage-threshold-calibration-negative-result.md)) |
| Does a structured triage gap help? | escalations 0.248 → 0.128, no hit lost, on 133 | shipped ([memo](./experiments/triage-gap-terminal-contract.md)) |
| Sheet as chunks or as a profile? | v2 beats v1 on all three department metrics | `v2` default ([memo](./experiments/department-qa-object-profile.md)) |
| Can a cheaper model hold the traps? | DeepSeek V4.1 Flash 4/4 at ~1/8 the cost of GPT-5 mini | showcase default ([memo](./experiments/cheap-model-selection.md)) |
| Does the showcase default hold on the full golden set, and what does it really cost? | In-scope correctness 7.91 clears the >7.5 target for the first time; real cost (DeepSeek priced) is ~70% above the prior baseline, not below, and latency regresses ~5× | headline numbers updated to this run; DeepSeek priced in `src/pricing.py`; superseded by the 23.09 rerun below ([memo](./experiments/showcase-default-golden-set.md)) |
| Does the reasoning-effort fix hold on the full golden set? | p50 24.0 → 9.5 s, cost $0.0066 → $0.0021, correctness and faithfulness not worse, false-sufficiency 7.1% (target met); relevance 0.887 → 0.853 | headline numbers moved to this run; relevance dip listed as a limitation ([memo](./experiments/golden-set-effort-low.md)) |
| Does scoped-service wiring (PR4/#59) change retrieval behaviour vs the old Department path? | Old vs new: 8/9 evidence byte-identical, both regressions are a prompt version change; workers=1 vs 4: identical prompt/evidence on every question | cutover accepted, no new paid run for wiring-only PR4 ([memo](./experiments/department-scoped-service-pr4-wiring.md)) |

## Threats to validity

Read every number above with these caveats next to it, not after it.

- **Pooling bias.** Retrieval labels were chosen from candidate pools returned by the
  systems being measured. A system that contributed more candidates (complex) is scored on
  its own output — complex reached HR@12 1.000 on the earlier 43-question held-out set, which
  is evidence of bias, not of quality. Backbone comparisons are valid as *relative* rankings;
  they are not absolute quality estimates, and a new retriever that never contributed to the
  pool would be under-scored.
- **The judge is not validated.** Generation scores come from an LLM judge (`gpt-4o`). Its
  agreement with human labels is not measured (issue #8); judge run-to-run variance is
  ~±0.25 on the correctness scale, so aggregate moves below that are noise. Per-question
  traces, not aggregates, drive targeted fixes.
- **Absolute numbers are judge-specific.** Runs before 2026-09-02 were silently judged by
  `gpt-4o-mini`; they are not comparable to current numbers. Compare only under one judge.
- **Small N, single run.** The 56-question generation set, the 9-question pair set and the
  4-question trap set are small. The trap comparison is one run per model; a model that
  fails a trap failed it once, and one that passes has not been shown stable. Treat those
  results as directional.
- **Weak labels and defects in the corpus.** Part of the retrieval ground truth is
  model-labeled with ~80% hand-check agreement. Chunking defects (a chunk carrying another
  section's heading, tables losing their header row, truncated clauses) put a ceiling on
  retrieval that no tuning fixes; the honest lower bound would need corpus-wide labeling.
- **Synthetic sets measure the wrong thing for lexical search.** Questions generated from
  chunk text repeat the source's wording and favour BM25 by construction. The 262-question
  synthetic set is a development set only; the 90 practitioner questions are the headline.
- **Raw run artifacts are not all in git.** `benchmarks/*.json(l)` are gitignored; the
  tables in [FACTS](../reference/FACTS.md), [roadmap](../roadmap.md) and this report are
  copied from them. Unit-scoped Q&A runs are committed in full under `eval/runs/`, including
  prompts and raw model outputs. Reproduction requires the index and source documents,
  which are not in git.
- **Runtime guarantees are not content guarantees.** For unit-scoped Q&A, `answered` means
  "citations checked", not "the cited text supports the claim". Support and completeness are
  measured in eval, not enforced at runtime.

## Reproduce

```bash
source .venv/bin/activate

# generation: full golden set with the gpt-4o judge (~$0.25)
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl
python eval/run_v7_eval.py --skip-judge --limit 5      # free smoke

# retrieval: headline 90-question set, per backbone
python eval/run_retrieval_eval.py --path simple --gt eval/data/golden_retrieval_labeled_ext.jsonl
python eval/run_retrieval_eval.py --path vector --gt eval/data/golden_retrieval_labeled_ext.jsonl
python eval/run_retrieval_eval.py --path bm25   --gt eval/data/golden_retrieval_labeled_ext.jsonl

# routing: deterministic threshold audit and grid (no judge)
python eval/triage_calibration.py

# unit-scoped Q&A: trap set and offline scoring (paid; see how-to)
.venv/bin/python eval/score_object_profile_run.py --run <run>/v2 \
  --expectations eval/data/object_profile_traps_expectations.yaml
```

CI runs `pytest -m "not integration and not slow"` and `python scripts/check_docs.py --ci`;
neither spends API budget.