# Evaluation — reference

Quality evaluation framework for the V7 pipeline. Current scores: [FACTS](FACTS.md#metrics).
To run a evaluation, see [how-to/run-evaluation](../how-to/run-evaluation.md).

## Components

1. **Dataset** — `tests/dataset.csv`. Golden set, columns `question` and `ground_truth` (56 questions: in-scope + OOS + false-premise).
2. **Runner** — `eval/run_v7_eval.py`. Runs the dataset through the compiled V7 graph, computes metrics, writes a JSONL report.
3. **Judge metrics** — `eval/advanced_generation_metrics.py` (`evaluate_faithfulness`, `evaluate_answer_relevance`) + `evaluate_correctness` in the runner. All LLM-as-judge via `get_judge_llm()`; judge model in [FACTS](FACTS.md#models).

## Metrics

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

## Report format

JSONL: each line is `{aggregate, results, dataset_size, valid_results, timestamp}`.
`aggregate` holds the metrics above; `results` is a list of per-question records
(`question`, `ground_truth`, `answer`, `path`, `*_score`, `*_reasoning`, `elapsed_sec`).
The judge's `reasoning` is saved per call — this makes score drops diagnosable
per-question (see [Design decisions §8](../explanation/design-decisions.md) on why we
trust per-question traces over noisy aggregates).

## See also

- [Benchmarks and baseline](../../benchmarks/README.md)
- [How-to: run an evaluation](../how-to/run-evaluation.md) · [add eval questions](../how-to/add-eval-questions.md)
