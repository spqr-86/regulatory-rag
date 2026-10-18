# Evaluation — reference

Quality evaluation framework for the V7 pipeline. Current scores: [FACTS](FACTS.md#metrics).
To run a evaluation, see [how-to/run-evaluation](../how-to/run-evaluation.md). Full
methodology, headline tables, dataset cards and threats to validity:
[evaluation report](../evaluation/README.md).

## Components

1. **Dataset** — `tests/dataset.csv`, verified cases only. Case identity is `case_id` / UUID;
   references and retrieval labels are frozen for each run. OOS and false premises are
   distinct types. Missing labels are unscored coverage, not retrieval misses.
2. **Runner** — `eval/ls/run.py`, using LangSmith's executor. `eval/run_v7_eval.py` delegates
   to it. Narrow retrieval-only, Department and ObjectProfile checks remain separate.
3. **Judges** — fact-based correctness and normative citations (`eval/ls/judge.py`), plus
   independent evidence-checked groundedness (`eval/ls/groundedness.py`). All use the
   existing model factory. Groundedness alignment and the first unified baseline are pending.

## Unified metrics

| Metric | Meaning | Eligible cases |
|---|---|---|
| correctness | All required reference facts are correct; supplemental facts are optional | Regular and false-premise |
| premise_corrected | Same substantive verdict on the correction; refusal keywords give no credit | False-premise only |
| groundedness | Material claims supported by exact generator context; quotes validated in code | Regular and false-premise; missing context unresolved |
| citations / citations_complete | Accurate normative addresses / sufficient references, assessed separately | Regular and false-premise |
| retrieval_hit / MRR | First matching labelled chunk / reciprocal rank, miss = 0 | Matching snapshot; missing labels are unresolved |
| oos_handled | Explicit refusal; failed or empty output earns no credit | Out-of-scope only |

For each metric: `eligible`, `scored`, `passed`, `failed`, `unresolved`, `not_applicable`,
`pass_rate=passed/scored`, `coverage=scored/eligible`. Zero denominators yield null. Assigned
pending cases stay in eligible denominators. `target_failed` is separate from judged
answer quality; provider/format errors are unresolved, never zeros. Missing context is
unscored, not a fabricated pass. There is no composite quality score or conversion from
the historical scale. The verified set is a development benchmark, not held-out evidence.

Summary also records assigned/completed/pending, target reliability, p50/p95 seconds,
accounted RAG cost, separate judge usage/cost and incompleteness. Paid embeddings are
untracked in `llm_usage`, so complete billing is not claimed. Rejudge excludes reused
pipeline expenses from its new total and preserves the source assessments/version IDs.

## Historical metrics (retained report series)

| Metric | What it checks | Target |
| :--- | :--- | :--- |
| **faithfulness** | Answer is grounded in retrieved context (LLM judge, 0–1) | > 0.85 |
| **answer_relevance** | Answer addresses the question (LLM judge, 0–1) | > 0.85 |
| **correctness_mean** | Match against ground truth (LLM judge, 0–10) | > 7.5 |
| **correctness_inscope** | Same, in-scope questions only | > 7.5 |
| **false_sufficiency_rate** | Share of simple-path answers with correctness < 5.0 | < 10% |
| **oos_rejection_rate** | Share of out-of-scope questions correctly abstained | > 0.90 |
| **complex_path_rate** | Share of queries routed to the complex path | — |
| **mean_elapsed_sec** | Average response latency | — |

Latest measured values: [FACTS](FACTS.md#metrics). `false_sufficiency` catches the main
anti-pattern — the system took the fast path and gave a bad answer.

## Triage calibration

`eval/triage_calibration.py` audits route decisions without an LLM judge. Its reviewed
43-question development set records evidence coverage and critical misses, then compares
threshold profiles against the current defaults. The 2026-09-11 grid evaluated 81
profiles; none reduced unsafe generations, so issue #9 retained the defaults. This is a
measured negative result, not a missing tuning step.

> **Note on the judge.** Scores depend on the judge model. The current judge is stricter
> than the earlier one (see [FACTS](FACTS.md#models)), so absolute numbers are lower than
> historical runs but better calibrated. Compare runs only under the same judge.

## Unified report format

`manifest.json` is created before execution: frozen selected references, case IDs and
versions, CSV/label/prompt hashes, snapshot identity/report hash, git SHA/dirty flag,
models and allowlisted effective settings (no credentials). `outputs.jsonl` records
targets, including query/run IDs and failures before any judge. `judges.jsonl` stores raw
structured verdicts and usage; `assessments.jsonl` records isolated evaluator outcomes.
`results.jsonl` appends one completed case per line and `summary.json` updates atomically.
Interruption preserves readable partial results. An output directory is never overwritten.

## Historical report format

Despite its `.jsonl` extension, the old runner wrote a single JSON object:
`{aggregate, results, dataset_size, valid_results, timestamp}`.
`aggregate` holds the metrics above; `results` is a list of per-question records
(`question`, `ground_truth`, `answer`, `path`, `*_score`, `*_reasoning`, `elapsed_sec`).
The judge's `reasoning` is saved per call — this makes score drops diagnosable
per-question (see [Design decisions §8](../explanation/design-decisions.md) on why we
trust per-question traces over noisy aggregates).

## See also

- [Benchmarks and baseline](../../benchmarks/README.md)
- [How-to: run an evaluation](../how-to/run-evaluation.md) · [add eval questions](../how-to/add-eval-questions.md)
