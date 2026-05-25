# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-05-25

### Changed
- `src/` restructured into `src/infra/` (LLM factory, prompt manager, semantic cache) and `src/indexing/` (file handler, vector store, chroma helpers)
- Two-tier LLM: Gemini 2.5 Flash (simple path, ~5s) + Gemini 3 Flash (complex path, thinking_budget=4096)
- Streamlit UI wording neutralized — removed domain-specific OT references

### Added
- CI workflow (`pytest -m unit`, 231 tests, no LLM eval cost)
- `src/backends/` — pluggable VectorStoreBackend protocol + ChromaBackend

### Removed
- Legacy Multi-Agent RAG (`agents/multiagent_rag.py`, `agents/`) — replaced by V7
- Source PDFs from git history (tracked via `.gitignore`)

### Fixed
- Circular import in `src/v7/__init__.py` (broke in fresh CI environment)
- `loguru` dependency replaced with `structlog` (loguru not in requirements.txt)
- FlashRank score inflation in evaluate_complex — vector scores used instead of reranker scores

---

## [1.0.0] — 2026-05-16

### Added
- **V7 LangGraph pipeline** — fully deterministic graph with hard-gate thresholds (no LLM routing)
- Two-stage retrieval: fast path (BM25 + vectors, top-12) / slow path (top-60 + MMR)
- `llm_verifier` → `rewriter` loop (one retry on borderline retrievals)
- `abstain` node — explicit refusal when retrieval confidence is low
- `domain_gate` — pre-retrieval OOS filter via cosine similarity to corpus centroid
- Security hardening: slowapi rate limiting, request ID, pickle → JSON cache, threading.RLock
- FastAPI REST API (`POST /query`, `POST /query/gosts`, `GET /health`)
- GOST RAG — 108 docs, 9,344 chunks, DeepSeek V3 generation
- Cost-per-sample baseline: **$0.0102/query** (N=10, May 2026)

### Changed
- Correctness: **7.9/10** (up from 6.04 in January 2026)
- Faithfulness: **0.988**
- Chunking fix (v2.3-noise-clean): 830 → **1,973 chunks** — P1 bbox bug resolved
- Full Labour Code (ТК РФ) added to corpus

---

## [0.1.0] — 2026-01-01

### Added
- Initial RAG pipeline: PDF ingestion via Docling, OpenAI embeddings, ChromaDB
- Multi-Agent RAG with ReAct agents (LangGraph), Gemini Flash, verifier
- Streamlit UI, hybrid retrieval (BM25 + semantic), FlashRank reranking
- 50-question golden evaluation dataset
- Prompt management system (Jinja2 templates, versioned registry)
- Term glossary (`config/term_glossary.yaml`) for deterministic query expansion
