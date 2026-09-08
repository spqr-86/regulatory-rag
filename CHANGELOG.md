# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Removed
- V8 evidence-assess triage variant and the `V7_V8_ENABLE_EVIDENCE_ASSESS` flag
  (with `V8_EVIDENCE_*` thresholds and `V8_SIMPLE_RERANK_TOP_K`). `evaluate_triage` is
  now a single deterministic hard-gate path — the reranker score is uncalibrated for
  this domain and had already caused a ranking bug (design decisions §3). The
  correctness numbers in the `[2.0.0]` entry were measured on the removed path; FACTS
  and README now carry the hard-gate re-run (56-q golden, `gpt-4o` judge): in-scope
  correctness 7.4, faithfulness 0.808, relevance 0.881, false-sufficiency 13%
  (vs 4.8%), complex-path 13% (vs 20.8%), $0.0033/query.

---

## [2.0.0] — 2026-09-08

173 commits since v1.1.0. The pipeline is now a single deterministic LangGraph ("V7")
with an evidence-aware triage stage; the legacy multi-agent chain and the
verifier/rewriter subgraph are gone. Offline retrieval + answer evaluation (module 04)
and online monitoring (module 05) are new. Domain-specific wording was removed — the
system is a general regulatory-RAG showcase.

Full run under the `gpt-4o` judge (56-question dataset, 53 valid, 2026-09-08):
in-scope correctness 7.56/10, faithfulness 0.840, answer relevance 0.847, OOS rejection
1.00, false-sufficiency 4.8%, complex-path rate 20.8%, latency p50/p95 5.2 s / 19.1 s,
cost $0.0045/query. Canonical values: `docs/reference/FACTS.md`.

### Added

**Retrieval**
- Hybrid dense + BM25 retrieval with RRF ranking (`src/v7/nodes/rag_simple.py`).
- CrossEncoder reranker; `RERANK_CANDIDATE_CAP` (=100) as the dominant latency/quality
  knob. FlashRank stays selectable via `RERANKER_BACKEND`.
- Cross-reference expansion — pulls referenced п./ст. into the passage window
  (`src/v7/cross_ref.py`); cache for cross-ref source fetches.
- Contextual embedding + table `element_type` in indexing.
- BM25 guarantee keyed on `passage_identity` (not `chunk_id`).
- Address layer: normative-docx parser + `AddressIndex` + build script
  (`src/address_layer.py`); MCP `get_norm` tool reads the address layer and verifies
  citations via `get_chunk`.

**Triage**
- V8 evidence-assess triage (`V7_V8_ENABLE_EVIDENCE_ASSESS`, on in production):
  three-way answer / improve / abstain verdict from reranker top-1 + coverage, instead
  of the legacy boolean hard gate.
- Structured sufficiency gap before escalation (issue #13, stage B2): triage emits a
  typed `TriageGap` instead of a boolean, closes gaps in-place by appending to the tail
  without reordering, escalates only when the gap stays open. Escalation rate
  0.188 → 0.128 on held-out 133, no Hit Rate@12 loss.
- Zero-overlap triage escalation.

**Prompts**
- Prompts moved to Jinja2 templates via `PromptManager`.
- `generate_answer` v2 → v8: anti-hallucination, false-premise check, anti-sycophancy,
  value-condition binding. Dead templates/registry entries removed.

**Evaluation (module 04)**
- Retrieval evaluation runner — Hit Rate@k, MRR, per-query latency p50/p95/p99, and
  `--path vector|bm25` backbones for baseline comparison (`eval/run_retrieval_eval.py`).
- Synthetic retrieval ground-truth generator (A→Q\*) with crash-safe resume and
  `--finalize` (issue #5); question-anchor enforcement, junk-chunk filtering.
- Held-out reference set 43 → 133 questions: dual-pass judge consensus, paid arbiter for
  rejected questions, agentic labelling, hand review applied.
- Per-query cost + latency: token usage reaches graph state (`src/v7/usage.py`), every
  run priced from tokens actually spent, split by retrieval path. Shared rate card
  `src/pricing.py` with `price_for()` — unknown model raises; unpriced models named in
  `unpriced_models`. `eval/pricing.py` is a re-export.
- RRF_K sweep (knob turned out to be dead); `RERANK_CANDIDATE_CAP` sweep.

**Monitoring (module 05)**
- `docker-compose.yml` — Postgres + Grafana monitoring stack (issue #15).
- Telemetry: query event + injectable event writer; single graph entry point
  (`src/v7/runner.py::run_query`) with `source` (`ui`/`api`/`eval`/`mcp`) threaded
  through; Postgres event writer + journal upload (issue #23); `FallbackWriter`;
  eval runs split by `run_id`.
- Grafana dashboard for request monitoring (issue #19).
- 👍/👎 feedback buttons, upsert into `feedback`, monitoring panel (issue #20).

**LLM factory / config**
- Per-path LLM provider + model config (`SIMPLE_*` / `COMPLEX_*`); DeepSeek provider.
- `get_judge_llm()` with dedicated `JUDGE_LLM_PROVIDER` / `JUDGE_MODEL_NAME`.

**Indexing**
- Docling `HybridChunker` replaces manual chunking (`max_tokens=400`, `merge_peers`).
- lxml fallback for DOCX with broken `.rels`.
- Noise cleaning: dash normalization, glue hyphenated line-breaks.

**MCP**
- `regulatory-mcp` — retrieval-only MCP server (`mcp_server.py`): raw chunks with
  composite `chunk_id` (`source::N`), lifespan + `AppContext`, CrossEncoder. Migrated to
  MCP 2.x (`FastMCP` → `MCPServer`, `mcp>=2,<3`).
- Retrieval-only `/retrieve` and `/corpus` REST endpoints; `source_filter` uses native
  metadata filter.

**Packaging / ops**
- `Dockerfile` + `.dockerignore` for the application image (Streamlit 8502 / FastAPI 8503).
- `[project]` table in `pyproject.toml`; `__version__` in `src/__init__.py`, served by
  `GET /` version and the FastAPI `version`.

**Docs**
- Diátaxis-lite tree (`docs/explanation|how-to|reference`), `docs/reference/FACTS.md` as
  single source of truth, `scripts/check_docs.py` freshness check in CI, design-decisions
  ADRs, `docs/roadmap.md` as the single plan location.

### Changed
- Latency: mean 16.3 s → 7.2 s (CrossEncoder candidate cap, visual-proof timeout 3 s,
  `MAX_VISUAL_PROOFS` 3 → 1).
- Default LLM provider: Gemini → OpenAI (`gpt-4o-mini` simple / `gpt-4o` complex),
  matching the deployed configuration; default reranker backend → `crossencoder`.
- Project renamed sia / Safety Incident Analyzer → regulatory-rag; prompts and UI
  neutralized (no ОТ / labour-safety references).
- structlog logs to stderr — stdout reserved for JSONRPC (stdio-MCP).

### Removed
- Legacy verifier/rewriter subgraph, `SemanticCache`, `USE_V7_GRAPH` toggle,
  `final_chain.py`, multiagent prompts, `ers_rag`, `gosts_pipeline` and the
  `POST /query/gosts` endpoint.
- Dead config: `constants.py`, `GOSTS_DB_PATH`, `CHUNK_OVERLAP`, `LLM_PROVIDER`,
  `MODEL_NAME`, `GEMINI_FAST_MODEL`; stale `config/__init.py__` (broken import, never
  loaded). `GEMINI_API_KEY` now read from env directly.
- WTA-specific eval datasets, GOST index, source PDFs from git tracking.

### Fixed
- `llm_factory`: the settings model name never reached `ChatOpenAI` —
  `kwargs.setdefault("model_name", model)` sat in the gemini branch only, so every OpenAI
  getter fell back to the constructor default (`gpt-4o-mini`) and `JUDGE_MODEL_NAME` /
  `COMPLEX_MODEL_NAME` were ignored. All eval runs before 2026-09-02 were judged by mini.
- `eval/generate_retrieval_gt.py`: a hard-coded gpt-4o-mini rate was applied to any model
  — the pre-flight estimate and `COST_ABORT_USD` guard could be off by a price ratio.
- subpara resolution tolerates mangled list numbering from HybridChunker (#13).
- domain gate enabled at 0.25 to block OOS; fallback unconditional-acceptance in
  `evaluate_complex`; `chunk_id` populated end-to-end; pagination in `get_by_filter`.
- NameError on library re-index in the app.

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
