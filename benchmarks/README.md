# Benchmarks

This directory contains the current baseline metrics and local eval artifacts.

**In git:** this README only.  
**Local only:** `eval_v7_*.jsonl`, `cps_*.json` — eval run artifacts, listed in `.gitignore`.

## Current Baseline (2026-05-15)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Correctness | **7.9 / 10** | > 7.5 | ✅ |
| Faithfulness | **0.988** | > 0.85 | ✅ |
| Answer Relevance | **0.753** | > 0.85 | 🔄 |
| False Sufficiency Rate | **15%** | < 10% | 🔄 |
| Complex path rate | 59% | — | — |
| Mean latency | 17.4s | < 10s | 🔄 |

**Config:** V7 LangGraph, OpenAI embeddings (text-embedding-3-small), Gemini 2.5 Flash (simple) + thinking (complex), 11 PDFs, HybridChunker v3.0-hybrid, dataset 57 questions.

## Running Eval

```bash
# Pipeline-only (no LLM judge, ~$0)
python eval/run_v7_eval.py --skip-judge --output benchmarks/eval_v7_$(date +%F).jsonl

# Full eval with LLM judge (OpenAI gpt-4o-mini)
python eval/run_v7_eval.py --output benchmarks/eval_v7_$(date +%F).jsonl
```

## Updating the Baseline

After a meaningful improvement, update the table above manually and commit.

---

[Русская версия](README_RU.md)
