# Project Passport: Regulatory RAG

**Purpose:** Automated search and Q&A over Russian normative documents (ГОСТ, СНиП, СП, ТК РФ).  
**Stack:** LangGraph, Streamlit, FastAPI, ChromaDB, Docling, Gemini 2.5 Flash / Gemini 3 Flash, OpenAI embeddings.

## Architecture

- **V7 LangGraph Pipeline:** deterministic state graph without LLM routing —
  `intent_gate → router → rag_simple → evaluate_triage → rag_complex → generate_answer`.
  Hard gates on numeric thresholds; explicit `abstain` when data is insufficient.
- **Two-tier LLM:** Gemini 2.5 Flash on simple path (~5s), Gemini 3 Flash with `thinking_budget=4096` on complex path.
- **Hybrid retrieval:** vector search (ChromaDB) + BM25, RRF merge, FlashRank reranking, MMR.
- **Pluggable backends:** LLM and vector store abstracted via factories (`src/infra/llm_factory.py`, `src/backends/`).
- **REST API:** FastAPI on port 8503 — `POST /query`, `GET /health`.

## Results

- Eval via `eval/run_v7_eval.py` (57 questions, LLM-as-judge gpt-4o-mini):
  - Correctness: **7.9/10** (target 7.5 achieved)
  - Faithfulness: **0.988**
  - False-sufficiency: 0.15
  - Cost per query: **$0.0102** (baseline May 2026)
- Corpus: 11 PDFs, index not yet rebuilt after v3.0-hybrid (previous: 1973 chunks)

## Challenges

- False-sufficiency 15% — some queries are incorrectly classified as sufficient on the simple path.
- Run-to-run variability of LLM judge in eval (~±0.3 correctness); trust aggregates, not per-question deltas.

## Developer Role

- Design and implementation of V7 graph on LangGraph (thin nodes + hard gates).
- Hybrid retrieval, RRF merge, MMR, FlashRank reranking.
- Domain term glossary (morphological matching, zero latency).
- Eval framework with LLM-as-judge metrics.
- Security hardening: slowapi rate limiting, pickle → JSON cache, threading.RLock.
- src/ restructure: infra/, indexing/, backends/ (Boards 1–5).
