# eval/

Offline evaluation suite for the V7 RAG pipeline. Runs the golden question dataset through the pipeline, measures quality metrics, and saves results to `benchmarks/`.

## Files

| File | Purpose | Used by |
|------|---------|---------|
| `run_v7_eval.py` | Main eval runner — runs dataset through V7 graph, writes JSONL to `benchmarks/` | CLI |
| `advanced_generation_metrics.py` | LLM-based metrics: faithfulness, answer relevance, context relevance, completeness | `run_v7_eval.py` |
| `metrics.py` | Deterministic (no LLM) metrics: completeness via lemma overlap, abstain rates, inversion detection, citation parsing | `tests/test_eval_metrics.py` |
| `retrieval_metrics.py` | Standard IR metrics: hit rate, MRR, precision/recall, NDCG | `tests/test_retrieval_metrics.py` |
| `custom_evaluators.py` | LangSmith-style correctness evaluator (not used in main pipeline, kept for reference) | — |
| `compare.py` | A/B diff of two eval JSON runs — **needs adaptation** for V7 metric keys (backlog #7) | CLI |

## Running Eval

```bash
# Full eval with LLM judge (gpt-4o-mini)
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl

# Pipeline only, no judge (~$0)
python eval/run_v7_eval.py --skip-judge

# Quick smoke test
python eval/run_v7_eval.py --limit 5 --skip-judge
```

Results are saved to `benchmarks/eval_v7_YYYY-MM-DD.jsonl`. See `benchmarks/README.md` for metrics history and field reference.
