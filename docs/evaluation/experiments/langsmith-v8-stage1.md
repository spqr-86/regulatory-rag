# LangSmith v8, stage 1: v7 baseline and first judge run

Date: 2026-10-15. First experiments on the `regrag-golden` dataset, code evaluators (runs 1–2), then LLM judges (run 3).

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

## Run 3: with judges

Experiment `v8-judges-84daf785`, git sha `1eb6807`, same settings as run 2 plus the two LLM
judges from `eval/ls/judge.py` (correctness: `openai/gpt-4o`, temperature 0, seed 12345;
citations: `get_citation_judge_llm()`). Command:
`LANGSMITH_TRACING_V2=true python -m eval.ls.run --prefix v8-judges`. 24 examples, no errors.

| Key | Score | n |
|---|---|---|
| `correctness` | 0.79 (11/14) | 14 |
| `forbidden_claim` | 1.00 | 8 (examples with `forbidden_claims`) |
| `citations` | 0.69 (9/13) | 13 (G022 `needs_review`, excluded) |
| `retrieval_hit` / `doc_cited` / `oos_handled` | 1.00 / 1.00 / 1.00 | 14 / 14 / 10 |
| MRR | 0.645 | 14 |

Correctness misses:

- G001: the answer leaves out the members of the knowledge-check and labour-safety committees
  and the trade-union representatives. Fair verdict. *Corrected 15.10: wrong, the answer lists
  both (its items 5 and 8); see "Judge alignment".*
- G028: no subclasses 3.1 to 3.4 of harmful conditions. Fair verdict. *Corrected 15.10: the
  answer names them in its reasoning; see "Judge alignment".*
- G052: disputed. The answer quotes part 4 of art. 27 of 426-FZ from the fragment ("may not be
  carried out for five years from the end of the attestation"); the judge reads this as a
  contradiction of the reference. Only "after that SOUT is mandatory" is really missing. A case
  for the judge alignment (task 5).
- G020 failed in the smoke run (`v8-judges-smoke-b218e2af`) and passed here: generator noise.

Citation misses: G002, G006, G008, G019. Same cause in all four: the answer cites
"[Фрагмент N]" without the provision number that the fragment carries. This is the generator
prompt, not retrieval or the judge; it is the first candidate after the baseline.

Cost: generation $0.0354 ($0.0015 per example). Judge calls are not counted by the runner and
LangSmith shows `total_cost = 0`, so the judge cost has to be taken from OpenRouter. Run time
22 minutes at `max_concurrency=1`.

## Judge alignment (task 5, 15.10)

Human labels: 17 runs of `v8-judges-84daf785` (Petr confirmed each case; the assistant
proposed a label first, so this is a check against both of them, not a blind label).

Iteration 0, judges as run:

| judge | agreement | fp (missed an error) | fn (rejected a correct answer) |
|---|---|---|---|
| correctness | 11/14 = 79 % | — | G001, G028, G052 |
| citations | 8/13 = 62 % | G001 | G002, G006, G008, G019 |

- Correctness: in G001 and G028 the judge says a fact is missing while the answer has it;
  in G001 it also penalises correct categories beyond the reference (civil servants). G052:
  "contradicts the reference" is wrong, only "after that SOUT is mandatory" is missing.
- Citations: the four fn are one rule. The judge failed an answer that does not name the
  provision; the labels judged accuracy only. G001 fp: the answer attributes programme "б"
  to the employer, the fragment gives only "а"; the judge checks references, not the claim.

Iteration 1, citations (code change, no new judge calls): status = accuracy, an unnamed
document/provision goes to a separate `citations_complete`. Rescored from the stored verdicts:
**11/12 = 92 %** (fp G001; G008 and G022 needs_review). Completeness fails on G002, G006, G008,
G019 — the generator candidate above.

Limits: tn = 0 — the only labelled citation error (G001) is the one the judge missed, so these
numbers say nothing yet about catching errors. The rule was changed after seeing these cases,
so the threshold has to be confirmed on a fresh labelled batch.

## Caveats

- All 24 examples carry `corpus_support=unverified`, so there is no "supported by corpus"
  subset to separate corpus gaps from retrieval failures.
- `visual_enrichment` timed out (>3 s) on some passages. It does not change the scores here,
  but it adds latency noise.
- 14 in-scope examples is a small sample: one example moves a rate by 7 points.
- Runs 1 and 2 have no judge. Run 3 judges are not yet aligned with manual labels (task 5);
  G052 shows at least one wrong correctness verdict.
