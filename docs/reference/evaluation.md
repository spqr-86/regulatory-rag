# Evaluation Guide

Quality evaluation framework for the V7 pipeline.

## Components

1. **Dataset** — `tests/dataset.csv`. Golden set, columns `question` and `ground_truth`
   (~57 questions: in-scope + OOS + false-premise).
2. **Runner** — `eval/run_v7_eval.py`. Runs the dataset through the compiled V7 graph,
   computes metrics, writes a JSONL report.
3. **Judge metrics** — `eval/advanced_generation_metrics.py` (`evaluate_faithfulness`,
   `evaluate_answer_relevance`) + `evaluate_correctness` inside the runner. All three are
   LLM-as-judge using `gpt-4o-mini` (`JUDGE_LLM_PROVIDER=openai`, configured via `get_judge_llm()`).

## Running

```bash
source venv/bin/activate
python eval/run_v7_eval.py                                  # full dataset
python eval/run_v7_eval.py --skip-judge                     # pipeline only, no LLM judge (~$0)
python eval/run_v7_eval.py --limit 5                        # quick smoke test
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

CLI flags: `--limit N` (cap number of questions), `--skip-judge` (no LLM scoring),
`--output PATH` (default: `benchmarks/eval_v7_{date}.jsonl`).

## Metrics

| Metric | What it checks | Target |
| :--- | :--- | :--- |
| **faithfulness** | Answer is grounded in retrieved context (LLM judge, 0–1) | > 0.85 |
| **answer_relevance** | Answer addresses the question (LLM judge, 0–1) | > 0.85 |
| **correctness_mean** | Match against ground truth (LLM judge, 0–10) | > 7.5 (achieved: 7.9) |
| **false_sufficiency_rate** | Share of simple-path answers with correctness < 5.0 | < 10% |
| **complex_path_rate** | Share of queries routed to complex path | — |
| **mean_elapsed_sec** | Average response latency | — |

`false_sufficiency` catches the main anti-pattern: the system took the fast path and gave a bad answer.

## Report Format

JSONL: each line is `{aggregate, results, dataset_size, valid_results, timestamp}`. `aggregate` contains
the metrics above; `results` is a list of per-question records (`question`, `ground_truth`, `answer`,
`path`, `*_score`, `*_reasoning`, `elapsed_sec`). Reasoning from each judge call is saved —
this makes it straightforward to diagnose score drops.

## LLM-as-Judge

Each metric is a separate request to the judge model with its own prompt. The judge must return
both a score and `reasoning`. When comparing runs, account for run-to-run variability:
small per-question deltas (especially on OOS questions) are noise; trust aggregates and large changes.

## See Also

- [Benchmarks and baseline](./../../benchmarks/README.md)
- [Adding questions to the dataset](../guides/adding-questions.md)
