# Regulatory Compliance RAG

**Production RAG pipeline for Russian regulatory documents (GOST, SNiP, Labour Code) — answers questions with citations or explicitly abstains when uncertain.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Test](https://github.com/spqr-86/safety-incident-analyzer/actions/workflows/evaluation.yml/badge.svg)](https://github.com/spqr-86/safety-incident-analyzer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Correctness: 7.9/10 · Faithfulness: 0.988 · Cost: $0.01/query**

[Russian README →](./README_RU.md)

---

## The problem

Regulatory documents in industrial domains (workplace safety, water treatment, construction) span hundreds of PDFs with cross-references. Manual lookup is slow and error-prone. A hallucinated answer to a compliance question isn't a UX issue — it's a liability.

This project explores how far RAG + deterministic guardrails can go toward reliable Q&A over regulatory corpora.

---

## How it works

```
User query
    ↓
intent_gate          — regex filter, drops noise before retrieval
    ↓
router               — query plan + domain glossary expansion
    ↓
rag_simple           — hybrid retrieval (BM25 + vectors, top-12) + FlashRank rerank
    ↓
evaluate_triage      — deterministic hard gates (no LLM scoring)
    ├── sufficient   → generate_answer (Gemini, thinking_budget=4096)
    ├── borderline   → llm_verifier → rewrite → rag_simple (one retry)
    └── clearly_bad  → rag_complex (top-60 + MMR) → evaluate_complex
                            ├── pass  → generate_answer
                            └── fail  → abstain (explicit refusal)
```

Key design decisions:
- **No LLM routing** — all branching decisions use deterministic score thresholds
- **Abstain > hallucinate** — system refuses to answer when retrieval confidence is low
- **Two-stage retrieval** — fast path handles most queries; slow path activates only when needed

---

## Metrics

| Metric | Value | Target |
|--------|-------|--------|
| Correctness (LLM-as-judge, 0–10) | **7.9** | > 7.5 |
| Faithfulness (no hallucinations) | **0.988** | > 0.85 |
| Answer Relevance | **0.85+** | > 0.85 |
| False-sufficiency rate | **15%** | < 10% |
| Cost per query | **$0.0102** | — |
| Avg latency | 21.9 sec | — |

Eval: 50-question golden dataset, `eval/run_v7_eval.py`. LLM judge: Gemini 2.5 Flash.

---

## Quick start

```bash
git clone https://github.com/spqr-86/safety-incident-analyzer.git
cd safety-incident-analyzer
pip install -r requirements.txt
cp .env.example .env  # add OPENAI_API_KEY + GEMINI_API_KEY
```

Add your PDF/DOCX regulatory documents to `source_docs/`, then:

```bash
python index.py        # index documents → ChromaDB
streamlit run app.py   # open http://localhost:8501
```

---

## Architecture

```mermaid
flowchart TD
    subgraph Ingestion
        Docs[PDF / DOCX] --> Docling[Docling Parser]
        Docling --> Split[Chunking 1500 chars / 400 overlap]
        Split --> Embed[OpenAI Embeddings]
        Embed --> DB[(ChromaDB)]
    end

    subgraph V7 [V7 LangGraph Pipeline]
        Q[Query] --> Gate{intent_gate}
        Gate -->|noise| End[END]
        Gate -->|domain| Router[router + glossary]
        Router --> Simple[rag_simple hybrid top-12 + FlashRank]
        Simple --> Triage{evaluate_triage hard gates}
        Triage -->|sufficient| Gen[generate_answer Gemini]
        Triage -->|borderline| Verifier[llm_verifier]
        Triage -->|clearly_bad| Complex[rag_complex top-60 + MMR]
        Verifier -->|ok| Gen
        Verifier -->|rewrite| Rewriter[rewriter] --> Simple
        Verifier -->|escalate| Complex
        Complex --> Eval[evaluate_complex hard gates]
        Eval -->|pass| Gen
        Eval -->|fail| Abstain[abstain]
        Gen --> Answer[Answer + sources]
    end
```

### GOST corpus (separate index)

108 GOST/SNiP documents → **9,344 chunks** in ChromaDB collection `wta_gosts` (example corpus, not included in repo — bring your own documents).
API endpoint: `POST /query/gosts`. Generation: DeepSeek V3.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangGraph (V7 deterministic graph) |
| LLM | Gemini 2.5 Flash (generation), DeepSeek V3 (GOST) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector store | ChromaDB |
| Reranking | FlashRank Cross-Encoder |
| ETL | Docling (PDF/DOCX → chunks) |
| Evaluation | Ragas + custom LLM-as-judge |
| UI | Streamlit |

---

## Project status

- ✅ V7 LangGraph pipeline — all nodes, deterministic routing
- ✅ Hybrid retrieval — BM25 + semantic, two-stage (simple/complex path)
- ✅ Hard gate thresholds — score-based, no LLM decisions in routing
- ✅ Eval framework — 50-question golden dataset, correctness 7.9/10
- ✅ GOST RAG — 108 docs, 9,344 chunks, separate ChromaDB collection
- ✅ Deployed on VPS (port 8502, Streamlit)
- 🔄 Expanding test dataset
- 🔄 False-sufficiency reduction (target < 10%)

---

**Author:** Petr Baldaev — [LinkedIn](https://linkedin.com/in/petr-baldaev-b1252b263/) · [GitHub](https://github.com/spqr-86)
