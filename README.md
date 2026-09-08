# Regulatory Compliance RAG

**RAG pipeline for Russian regulatory documents (ГОСТ, СНиП, Trudovoy Kodeks, fire-safety and labour-safety rules) — answers questions with citations or explicitly abstains when uncertain.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![CI](https://github.com/spqr-86/regulatory-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/spqr-86/regulatory-rag/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Answer quality** (56-question golden set, `gpt-4o` judge): in-scope correctness **7.56 / 10** · faithfulness **0.840** · answer relevance **0.847** · OOS rejection **1.00** · false-sufficiency **4.8%** · complex-path **20.8%** · **~$0.0045/query**, p50 **5.2 s**.

> Metrics are judge-dependent — canonical values live in [docs/reference/FACTS.md](./docs/reference/FACTS.md). The reasoning behind the architecture is in [docs/explanation/design-decisions.md](./docs/explanation/design-decisions.md).

[Russian README →](./README_RU.md)

---

## The problem

Regulatory documents in industrial domains (workplace safety, fire safety, construction) span hundreds of PDFs with dense cross-references. Manual lookup is slow and error-prone. A hallucinated answer to a compliance question isn't a UX issue — it's a liability.

This project explores how far RAG + deterministic guardrails can go toward reliable Q&A over regulatory corpora.

---

## How it works

```
User query
    ↓
intent_gate          — regex noise filter + (optional) cosine-to-centroid OOS gate, before retrieval
    ↓
router               — query plan + glossary expansion + multi-query (RRF merge)
    ↓
rag_simple           — hybrid retrieval (BM25 + vectors, top-12) + CrossEncoder rerank
    ↓
evaluate_triage      — deterministic sufficiency gate (no LLM scoring)
    ├── sufficient    → generate_answer
    └── insufficient  → rag_complex (top-60 + MMR) → evaluate_complex
                            ├── pass  → generate_answer
                            └── fail  → abstain (explicit refusal)
```

Key design decisions:
- **No LLM routing** — all branching decisions use deterministic score thresholds
- **Abstain > hallucinate** — the system refuses to answer when retrieval confidence is low
- **Two-stage retrieval** — a fast path handles most queries; the slow path activates only when needed

The deployed triage runs the V8 evidence-assess variant (reranker score + coverage);
the legacy hard-gate path stays available behind a flag. See
[docs/explanation/triage.md](./docs/explanation/triage.md).

📖 **Docs:** [architecture](./docs/explanation/architecture.md) · [design decisions](./docs/explanation/design-decisions.md) · [FACTS](./docs/reference/FACTS.md) · [full documentation](./docs/README.md)

---

## Metrics

| Metric | Value |
|---|---|
| In-scope correctness | 7.56 / 10 |
| Correctness (all questions) | 7.30 / 10 |
| Faithfulness | 0.840 |
| Answer relevance | 0.847 |
| OOS rejection rate | 1.00 |
| False-sufficiency rate | 4.8% |
| Complex-path rate | 20.8% |
| Latency p50 / p95 / mean | 5.2 / 19.1 / 7.2 s |
| Cost / query | $0.0045 ($0.24 / run) |
| Retrieval HR@5 / HR@12 / MRR (hybrid, 90 practitioner questions) | 0.63 / 0.81 / 0.50 |

Eval: 56-question golden dataset (`tests/dataset.csv`), `eval/run_v7_eval.py`, LLM judge
`gpt-4o`. Numbers are judge-dependent — compare runs only under the same judge. Retrieval
Hit Rate / MRR are measured separately on a 90-question test set of real questions taken
verbatim from OT/PB practitioner forums, kept only where a corpus answer exists and not
used for any tuning (`eval/run_retrieval_eval.py`); see [docs/roadmap.md](./docs/roadmap.md).
Canonical values: [docs/reference/FACTS.md](./docs/reference/FACTS.md).

---

## Quick start

```bash
git clone https://github.com/spqr-86/regulatory-rag.git
cd regulatory-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY (LLM + embeddings by default)
```

Drop your PDF/DOCX regulatory documents into `source_docs/`, then:

```bash
python index.py                 # index documents → ChromaDB (destructive: wipes the collection first)
streamlit run app.py --server.port 8502   # UI at http://localhost:8502
uvicorn api:app --port 8503                # REST API at http://localhost:8503/docs
```

Defaults to OpenAI (LLM + embeddings) + ChromaDB. See [Backend abstraction](#backend-abstraction) to swap any layer via `.env`.

Optional monitoring stack (Postgres for query events, Grafana on top):

```bash
cp .env.example .env              # set POSTGRES_PASSWORD and GF_SECURITY_ADMIN_PASSWORD
docker compose up -d              # both services healthy; schema applied by db/migrations on an empty volume
V7_TELEMETRY_WRITER=postgres      # in .env: write query events to the stack instead of JSONL
```

**Where to look:** dashboard *Regulatory RAG — запросы* at
`http://localhost:3000/d/regrag-queries` (login from `.env`) — cost for the period, 👎 rate,
queries per day split by `source`, routes, cost and latency at p50/p95, and the latest 👎
with the question behind each one.

**Where to click:** under every Streamlit answer there are 👍/👎 buttons (a 👎 opens an
optional comment box). One vote per answer — pressing again overwrites the row, so the
panels count answers, not clicks. Without the stack the buttons are hidden: a vote would
have nowhere to land.

**If Postgres is not up:** nothing breaks. Answers keep working, events go to the
`logs/events.jsonl` journal — also when the database dies mid-flight — and
`python scripts/ingest_events.py logs/events.jsonl` replays them into the table once it is
back (idempotent by `query_id`). Port conflicts, a volume with an old password, empty
panels: [docs/how-to/run-monitoring-stack.md](./docs/how-to/run-monitoring-stack.md),
section «Если стек не поднялся».

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        Docs[PDF / DOCX] --> Docling[Docling Parser]
        Docling --> Split[HybridChunker max_tokens=400, merge_peers]
        Split --> Embed[OpenAI Embeddings]
        Embed --> DB[(ChromaDB)]
    end

    subgraph V7 [V7 LangGraph Pipeline]
        Q[Query] --> Gate{intent_gate + domain gate}
        Gate -->|noise / out-of-scope| End[END / abstain]
        Gate -->|in-domain| Router[router + glossary + multi-query]
        Router --> Simple[rag_simple hybrid top-12 + CrossEncoder]
        Simple --> Triage{evaluate_triage evidence-assess}
        Triage -->|sufficient| Gen[generate_answer]
        Triage -->|insufficient| Complex[rag_complex top-60 + MMR]
        Complex --> Eval[evaluate_complex]
        Eval -->|pass| Gen
        Eval -->|fail| Abstain[abstain]
        Gen --> Answer[Answer + sources]
    end
```

The shipped index holds 12 regulatory documents (~7.8k chunks). It is an example corpus —
bring your own documents. Exact counts: [docs/reference/FACTS.md](./docs/reference/FACTS.md#corpus).

---

## REST API

Run the FastAPI backend alongside Streamlit:

```bash
uvicorn api:app --port 8503
```

**`POST /query`** — main RAG pipeline

```bash
curl -X POST http://localhost:8503/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Как часто проводится повторный инструктаж?"}'
```

```json
{
  "answer": "Повторный инструктаж проводится не реже одного раза в 6 месяцев...",
  "passages": [{"text": "...", "source": "2464.pdf", "score": 0.91}],
  "path": "rag_simple → evaluate_triage → generate_answer → END",
  "elapsed_sec": 4.2
}
```

**`POST /retrieve`** — retrieval only (hybrid search, no LLM); **`GET /corpus`** — indexed
documents; **`GET /health`** — liveness (`{"status": "ok"}`).

Full reference with request/response shapes and rate limits:
[docs/reference/api.md](./docs/reference/api.md). Interactive docs: `http://localhost:8503/docs`.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (V7 deterministic graph) |
| LLM | OpenAI (default), Gemini, DeepSeek — configurable per path via `SIMPLE/COMPLEX_LLM_PROVIDER` in `.env` |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Reranking | CrossEncoder (sentence-transformers); FlashRank selectable via `RERANKER_BACKEND` |
| ETL | Docling (PDF/DOCX → chunks) |
| Evaluation | custom LLM-as-judge (faithfulness, answer relevance, correctness) + IR metrics (Hit Rate@k, MRR) |
| Monitoring | Postgres + Grafana (docker compose), events written from inside the graph |
| UI | Streamlit |

---

## Backend abstraction

LLM and vector store are accessed through factory layers (`src/infra/llm_factory.py`, `src/backends/`). Adding a new provider is one function plus one registry entry — pipeline code does not change.

| Layer | Shipped | Configurable via | Roadmap |
|-------|---------|------------------|---------|
| LLM   | OpenAI, Gemini, DeepSeek | `SIMPLE_LLM_PROVIDER` / `COMPLEX_LLM_PROVIDER` | Anthropic |
| Vector store | Chroma | `VECTOR_STORE` | Qdrant, pgvector |
| Embeddings | OpenAI, local (sentence-transformers), hf_api | `EMBEDDING_PROVIDER` | — |

**Fully local embeddings** (LLM still over an API):
```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL_NAME=ai-forever/sbert_large_nlu_ru
```

**Adding a new LLM provider** (example: Anthropic):
1. Add `_create_anthropic_llm(**kwargs)` to `src/infra/llm_factory.py`
2. Register it in `_LLM_PROVIDERS = {..., "anthropic": _create_anthropic_llm}`
3. Set `SIMPLE_LLM_PROVIDER=anthropic` (and/or `COMPLEX_LLM_PROVIDER`) in `.env`

Same pattern for vector stores — implement the `VectorStoreBackend` protocol in `src/backends/`, register it in the factory.

---

## Adapting to your domain

The system ships tuned for Russian regulatory documents, but domain-specific knowledge is isolated and easy to swap.

**Term glossary** (`config/term_glossary.yaml`) — maps informal abbreviations to their official full names so BM25 and vector search can match indexed text. To extend:

```yaml
terms:
  "your abbreviation":
    official: "Full official term from your documents"
    source: "Regulation / standard reference (optional)"
```

No code changes needed — edit the YAML and restart.

**Prompts** (`prompts/`) — Jinja2 templates via `PromptManager`, versioned. Switch the active version via `prompts/registry.yaml`.

**Corpus** — drop your PDFs into `source_docs/` and run `python index.py`. The chunker and embeddings are language-agnostic.

---

## Project status

- ✅ V7 LangGraph pipeline — all nodes, deterministic routing (verifier/rewriter retired — insufficient triage routes straight to rag_complex)
- ✅ Hybrid retrieval — BM25 + semantic, two-stage (simple/complex path)
- ✅ Deterministic sufficiency gate — score-based, no LLM decisions in routing; V8 evidence-assess in production
- ✅ Structured triage gap — triage emits a typed gap and closes it by tail-append before escalating (issue #13)
- ✅ Domain gate — optional pre-retrieval OOS filter via cosine similarity to corpus centroid
- ✅ HybridChunker — structure-aware chunking aligned to document sections/articles
- ✅ Contextual embedding — parent-section heading prepended to each chunk vector
- ✅ Cross-reference expansion — auto-fetches referenced clauses (e.g. "пункт 46") from the same source
- ✅ Multi-query expansion — LLM generates query variants, RRF merge
- ✅ Versioned prompts — Jinja2 templates, registry trimmed to 3 live families; `generate_answer` v8 (anti-sycophancy + value↔condition binding)
- ✅ Offline eval — golden dataset + 90-question practitioner retrieval test set, per-query cost and latency
- ✅ Online monitoring — every query is a row in Postgres (cost, latency, route, tokens, `source`), Grafana dashboard, 👍/👎 under the answer; the whole stack is one `docker compose up`
- ✅ Deployed on a VPS (port 8502, Streamlit)
- 🔄 Value↔condition robustness on multi-value queries (e.g. program-type periodicity)
- 🔄 Chunking rewrite for tables and section headers (modules 06/07)

---

**Author:** Petr Baldaev — [LinkedIn](https://linkedin.com/in/petr-baldaev-b1252b263/) · [GitHub](https://github.com/spqr-86)

[Changelog →](./CHANGELOG.md)
