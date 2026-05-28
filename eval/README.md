# eval/

Offline evaluation suite for the V7 RAG pipeline.

---

## What this folder is for

Run the golden question dataset through the pipeline, measure quality metrics, and save results to `benchmarks/` for tracking over time.

---

## Files

### `run_v7_eval.py` — main eval runner

Runs the golden dataset through the V7 graph and writes one JSONL line per question to `benchmarks/`.

Flags:
- `--limit N` — process only the first N questions (quick smoke test)
- `--skip-judge` — skip the LLM judge; saves pipeline answers with no correctness score (~$0)
- `--output path` — override the default output path

Imports from: `advanced_generation_metrics`, `src.v7.graph`, `src.infra.llm_factory`

### `advanced_generation_metrics.py` — LLM-based generation metrics

Evaluates generation quality using an LLM judge. All functions return a dict with a numeric score and a reasoning string.

Functions:
- `evaluate_faithfulness(question, context, answer, llm)` — are all claims grounded in the context?
- `evaluate_answer_relevance(question, answer, llm)` — does the answer address the question?
- `evaluate_context_relevance(question, context, llm)` — is the retrieved context relevant to the question?
- `evaluate_completeness(question, answer, reference_answer, llm)` — completeness vs. reference answer
- `evaluate_citation_quality(answer, context, source_docs)` — citation count and diversity (no LLM)
- `evaluate_generation_comprehensive(...)` — runs all of the above in one call

Used by: `run_v7_eval.py`

### `metrics.py` — deterministic (no-LLM) metrics

Lemma-based and rule-based metrics. No external calls.

Functions:
- `compute_completeness(ground_truth, answer)` — fraction of ground truth key lemmas found in the answer
- `compute_abstain_rate(results)` — fraction of queries the system declined to answer
- `compute_false_abstain_rate(results)` — false abstains on in-domain queries (target: 0)
- `compute_correct_abstain_rate(results)` — correct abstains on OOS queries (target: 1.0)
- `compute_inversion_detected(must_not_contain, answer)` — detects forbidden patterns (norm inversions)
- `compute_inversion_rate(results)` — aggregate inversion rate over a result set
- `parse_citations(answer)` — extracts `[Фрагмент N: Doc, п. X.X]` citations
- `compute_citation_rate(answer)` — fraction of sentences with at least one citation
- `compute_citation_in_retrieval(answer, passages)` — fraction of citations with a valid fragment number
- `compute_citation_doc_match(answer, passages)` — fraction of citations whose doc name matches the source
- `compute_retrieval_stats(results)` — average top score and passage count

Used by: `tests/test_eval_metrics.py`

### `retrieval_metrics.py` — retrieval quality metrics

Standard IR metrics. No LLM calls.

Functions:
- `hit_rate_at_k(retrieved, relevant, k)` — 1.0 if any relevant doc is in top-K
- `mean_reciprocal_rank(retrieved_list, relevant_list)` — MRR over a batch
- `precision_at_k(retrieved, relevant, k)` — P@K
- `recall_at_k(retrieved, relevant, k)` — R@K
- `ndcg_at_k(retrieved, relevant_scored, k)` — NDCG@K (requires relevance score dict)
- `evaluate_retrieval(retrieved, relevant, k)` — all metrics for one query
- `evaluate_retrieval_batch(retrieved_list, relevant_list, k)` — averaged metrics + MRR over a batch

Used by: `tests/test_retrieval_metrics.py`

### `custom_evaluators.py` — LangSmith-style correctness evaluator

`check_correctness(run, example)` — LLM judge scoring 0-10, designed for the LangSmith `evaluate()` API. Currently **not used** in the main pipeline; kept for reference.

### `compare.py` — A/B comparison of two eval runs

Loads two JSONL result files and prints a diff table. **Needs adaptation** for V7 metric keys (backlog item).

---

## How to run an eval

```bash
cd /home/petr/projects/ai/regulatory-rag
source venv/bin/activate

# Full eval with LLM judge
python eval/run_v7_eval.py

# Pipeline only, no judge (~$0)
python eval/run_v7_eval.py --skip-judge

# Quick smoke test (5 questions)
python eval/run_v7_eval.py --limit 5 --skip-judge

# Custom output path
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

---

## Where results are saved

`benchmarks/` — one JSONL file per run, named `eval_v7_YYYY-MM-DD.jsonl`.

Each line is a JSON object with fields: `question`, `answer`, `ground_truth`, `path`, `abstained`, `top_score`, `passage_count`, `latency_s`, plus judge scores when `--skip-judge` is not set.

See `benchmarks/README.md` for the full field reference and history of runs.
