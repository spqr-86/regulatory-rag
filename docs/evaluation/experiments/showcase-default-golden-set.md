# Showcase default on the full golden set, and a real price for it

> Superseded for headline numbers by [golden-set-effort-low](./golden-set-effort-low.md)
> (23.09.2026): the latency regression below was the provider default reasoning effort,
> fixed in #63.

## Question

The showcase default (`deepseek/deepseek-v4.1-flash`, simple path) was chosen on a 4-question
adversarial trap set ([cheap-model-selection](./cheap-model-selection.md)). Does it hold on the
full 56-question golden set that headline numbers are quoted from, and what does it actually
cost — DeepSeek was not in the rate card at the time, so every simple-path query priced at $0?

## Setup

- **What changed:** simple-path model only, `openai/gpt-4o-mini` → `deepseek/deepseek-v4.1-flash`
  (OpenRouter). Complex path still ran `gpt-4o` at run time — the complex path moved to DeepSeek
  too on 18.09.2026, one day after this run; this memo's numbers predate that change and are not
  a measurement of the current complex-path wiring.
- **What stayed fixed:** `eval/run_v7_eval.py`, full terminal triage contract, `gpt-4o` judge,
  CrossEncoder reranker, cap 100, same 56-question golden set as the 2026-09-11 baseline.
- **Run:** `eval/run_v7_eval.py`, 2026-09-17, 53/56 valid. Artifact:
  `benchmarks/eval_v7_deepseek_2026-09-17.jsonl` (gitignored, kept locally).
- **Pricing gap found in the artifact:** `unpriced_models: ["deepseek/deepseek-v4.1-flash"]` on
  every simple-path row — the run's own `total_cost_usd` ($0.2282) counts only the complex
  path's `gpt-4o` calls and is not the real cost of the run.

## Rate card fix

`deepseek/deepseek-v4.1-flash` had no entry in `src/pricing.py::PRICE_PER_1M`. Checked live on
OpenRouter (21.09.2026, not from memory): the headline discounted price is $0.12 input /
$0.48 output per 1M tokens, but OpenRouter routes across ~21 provider slots for this model
that on the same day ranged $0.12–0.375 input / $0.48–1.50 output, with $0.30/$1.20 (the
undiscounted base rate) the most common. DeepSeek's own official off-peak rate is $0.15/$0.60
(doubles 01:00–04:00 and 06:00–10:00 UTC weekdays). Added **$0.15/$0.60** as the stable
reference rate rather than the fluctuating OpenRouter headline discount, so the cost guard
doesn't understate a run priced on a pricier route.

Re-pricing the run's actual token usage (262,868 prompt + 134,788 completion tokens on
`deepseek/deepseek-v4.1-flash`, summed from per-call `usage` records) against the new rate and
adding it to the run's existing `gpt-4o` complex-path cost gives the real total.

## Results

| Metric | 17.09-7 (DeepSeek simple / GPT-4o complex) | 11.09 baseline (GPT-4o-mini simple / GPT-4o complex) |
|---|---:|---:|
| In-scope correctness | **7.91 / 10** (target >7.5 ✅, first pass) | 7.47 / 10 |
| Correctness, all questions | 7.98 / 10 | 7.26 / 10 |
| Faithfulness | 0.926 | 0.891 |
| Answer relevance | 0.887 | 0.879 |
| OOS abstain rate | 1.00 | 1.00 |
| False-sufficiency rate | **10.0%** (target <10%, 0.1 pp outside, formally) | 11.4% |
| Complex-path rate | 24.5% | 17.0% |
| Latency p50 / p95 | 24.0 s / 71.3 s | 4.51 s / 15.70 s |
| Cost / query, as run's own total reported it | $0.00430 | $0.00387 |
| **Cost / query, real (DeepSeek priced at $0.15/$0.60)** | **$0.00657** ($0.348 / run) | $0.00387 |

## Interpretation

Correctness, faithfulness and false-sufficiency all improve over the 11.09 baseline — in-scope
correctness clears its >7.5 target for the first time. Two costs came with it that the run's
own summary hid: latency p50 moved from 4.5 s to 24 s (~5×, not explained by this run — DeepSeek
via OpenRouter is simply slower than `gpt-4o-mini` here), and the real cost per query is $0.00657
once DeepSeek is actually priced, not the $0.00430 the unpriced run reported — about 70% more
than the baseline, not less. A cheap-looking model was showing as cheap because it wasn't priced
at all; the trap-set selection ([cheap-model-selection](./cheap-model-selection.md)) measured
correctness-per-dollar on 4 questions using OpenRouter list prices at the time, not this run's
actual usage, so the two numbers were never expected to match, but $0 was never a legitimate
value either way.

## Decision

Headline numbers in `FACTS.md`/`README.md`/`README_RU.md`/`docs/evaluation/README.md` updated
to this run (7.91 / 10.0% replacing 7.47 / 11.4%), with cost corrected to the real $0.00657
figure. `deepseek/deepseek-v4.1-flash` added to `src/pricing.py::PRICE_PER_1M` at $0.15/$0.60.

## Threats to validity

Single run, same judge-variance caveats as every golden-set number (~±0.25 on the correctness
scale). The complex path here still ran `gpt-4o`, not the DeepSeek complex path shipped since
18.09.2026 — a full-contract rerun on the current wiring has not been done and would very likely
move cost and latency further (complex-path rate here was already 24.5% vs 17.0% baseline, and
DeepSeek's own latency dominates that path too). The $0.15/$0.60 rate is a snapshot from
21.09.2026, not a contract — OpenRouter's actual routed price varies per call and this project
does not track which of the ~21 provider slots served a given request, so real spend can sit
anywhere in the $0.12–0.375/$0.48–1.50 range depending on routing.

## Next action

Rerun the full golden set on the current wiring (DeepSeek on both paths, post-18.09.2026) to get
a cost/latency number that matches production, rather than continuing to headline a run whose
complex path no longer reflects what ships. Investigate the ~5× latency regression before
treating 24 s p50 as an accepted cost of the showcase default.
