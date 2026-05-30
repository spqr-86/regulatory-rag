# How to run an evaluation

Runs the golden dataset through the V7 graph and scores it. Metric definitions and the
report format are in [reference/evaluation](../reference/evaluation.md).

```bash
source venv/bin/activate

python eval/run_v7_eval.py                              # full dataset, with LLM judge
python eval/run_v7_eval.py --skip-judge                 # pipeline only, no judge (~$0)
python eval/run_v7_eval.py --limit 5                    # quick smoke test
python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
```

**Flags:** `--limit N` (cap questions), `--skip-judge` (no LLM scoring),
`--output PATH` (default `benchmarks/eval_v7_{date}.jsonl`).

**Cost:** the judge issues separate LLM calls per metric per question — a full run costs
a few cents to ~$0.30. Use `--skip-judge` for a free pipeline-only smoke run.

**Output:** a JSONL report under `benchmarks/`. The judge model is set by
`JUDGE_MODEL_NAME` (see [FACTS](../reference/FACTS.md#models)).

**Re-judging without re-running the pipeline:** `scripts/rejudge.py` re-scores saved
answers with a judge — cheap A/B of judge prompts or models without paying for retrieval
and generation again.

For a single-question trace instead of a full run:

```bash
python scripts/trace_v7.py "your question"
```
