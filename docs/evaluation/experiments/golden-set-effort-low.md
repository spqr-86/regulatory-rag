# Golden set on the current wiring, reasoning effort low

## Question

The 17.09 showcase run ([showcase-default-golden-set](./showcase-default-golden-set.md)) came
with a ~5× latency regression (p50 24 s). The cause was found on 23.09: OpenRouter dropped the
thinking budget, so DeepSeek ran with the provider's default reasoning (effort=high, no output
cap). [#63](https://github.com/spqr-86/regulatory-rag/pull/63) set
`OPENROUTER_REASONING_EFFORT=low`; a 20-question A/B showed p50 15.5 → 9.3 s at correctness
9.1 → 8.9. Does the fix hold on the full 56-question set, and what does quality look like on
the wiring that actually ships?

## Setup

- **What changed vs 17.09:** reasoning effort high (implicit) → `low`; complex path
  `gpt-4o` → `deepseek/deepseek-v4.1-flash` (shipped 18.09). Two changes at once — this run
  measures the current default, not the effect of either change alone.
- **What stayed fixed:** `eval/run_v7_eval.py`, full terminal triage contract, `gpt-4o` judge,
  CrossEncoder reranker, cap 100, the same 56-question golden set, cost priced against
  `src/pricing.py` ($0.15/$0.60 per 1M for DeepSeek).
- **Run:** 2026-09-23, 53/56 valid (the three empty answers are domain-gate refusals of
  out-of-scope questions). Artifact: `benchmarks/eval_v7_deepseek_low_2026-09-23.jsonl`
  (gitignored, kept locally). Providers behind OpenRouter varied per call (Together, Alibaba,
  DeepInfra, Novita and others).
- **Counter fix:** the run reported OOS rejection 0.75. The counter dropped empty answers from
  the denominator and looked for «нет» only in the first 50 characters, so a refusal opening
  with «Я не могу игнорировать свои инструкции» scored as a miss. Fixed in `c9b907f`; the
  artifact carries the value recomputed from the saved answers (7/7) next to the as-run one.

## Results

| Metric | 23.09 (DeepSeek both paths, effort low) | 17.09 (DeepSeek simple / GPT-4o complex, effort high) |
|---|---:|---:|
| In-scope correctness | **8.09 / 10** | 7.91 / 10 |
| Correctness, all questions | 8.32 / 10 | 7.98 / 10 |
| Faithfulness | **0.974** | 0.926 |
| Answer relevance | 0.853 | 0.887 |
| — in-scope questions only | 0.912 | 0.953 |
| OOS abstain rate | 1.00 (7/7) | 1.00 |
| False-sufficiency rate | **7.1%** (target <10% ✅) | 10.0% |
| Complex-path rate | 20.8% | 24.5% |
| Latency p50 / p95 / mean | **9.5 / 25.9 / 12.0 s** | 24.0 / 71.3 / 31.0 s |
| Cost / query | **$0.0021** ($0.11 / run) | $0.0066 ($0.35 / run) |

## Interpretation

The latency fix holds on the full set: p50 drops 2.5× and p95 2.8×, and the cost per query
falls to a third, because fewer reasoning tokens are generated and the complex path no longer
pays `gpt-4o` prices. Correctness and faithfulness do not get worse; the gains (+0.18
in-scope correctness) sit inside the judge's ±0.25 run-to-run variance and should be read as
"no loss", not as an improvement. False-sufficiency now clears its <10% target (3 of 42
simple-path answers scored below 5/10 vs 10.0% before).

One number moved the wrong way: answer relevance, 0.887 → 0.853, and it is not an artefact
of the out-of-scope refusals — relevance on in-scope questions alone fell 0.953 → 0.912. The
overall value still clears the >0.85 target, but only just. Lower reasoning effort is the
likely cause; this run cannot separate it from the complex-path model change.

## Decision

Headline numbers in `FACTS.md`, `README.md`, `README_RU.md` and `docs/evaluation/README.md`
move to this run. The relevance dip is listed under Limitations rather than hidden.

## Threats to validity

Single run; judge variance ~±0.25 on the correctness scale. Two variables changed at once
(effort and complex-path model). OpenRouter routed calls across several providers, so latency
depends on which provider served a given call. Cost is priced at the $0.15/$0.60 reference
rate, not at what OpenRouter actually billed.

## Next action

None required for the showcase. If relevance matters, an A/B of effort `low` vs `medium` on
the in-scope subset would show whether the dip is the price of the latency fix.
