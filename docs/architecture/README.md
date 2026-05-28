# Architecture Overview

Regulatory RAG is a retrieval-augmented generation system for Russian normative documents (ГОСТ, ТК РФ, СНиП, СП). It answers regulatory compliance questions by retrieving relevant passages from an indexed corpus and synthesizing an answer with Gemini.

See also: [How V7 pipeline works](./v7-how-it-works.md) · [How triage works](./triage-how-it-works.md)

---

## Pipeline

| Step | Node | What it does |
|------|------|--------------|
| 1 | `intent_gate` | Regex filter: noise queries → END, regulatory queries continue |
| 2 | `router` | Classifies query, builds plan, expands `active_query` via term glossary |
| 3 | `rag_simple` | Hybrid retrieval (vector + BM25, RRF merge), FlashRank rerank, top-12 passages |
| 4 | `evaluate_triage` | Hard gates → `sufficient` / `borderline` / `clearly_bad` |
| 5a | `llm_verifier` | (borderline only) LLM decides: sufficient / rewrite / escalate |
| 5b | `rag_complex` | (clearly_bad / escalated) Deep retrieval top-60 + MMR, merges all attempts |
| 6 | `evaluate_complex` | Hard gates on merged passages; fail → abstain |
| 7 | `generate_answer` | Gemini synthesizes answer from up to 24 final passages |

---

## Codebase Map

| Path | Contents |
|------|----------|
| `src/v7/` | Pipeline nodes (`nodes/`), graph assembly (`graph.py`), state types, hard gates, NLP core, bridge DI adapter |
| `src/infra/` | LLM factory, prompt manager, semantic cache, parsers, shared types |
| `src/indexing/` | Document processor (HybridChunker), Chroma helpers, vector store, applicability retriever |
| `src/backends/` | `VectorStoreBackend` protocol + ChromaDB implementation |
| `config/` | `settings.py` (pydantic-settings), `term_glossary.yaml` |
| `prompts/` | Jinja2 templates + `registry.yaml` |
| `eval/` | `run_v7_eval.py`, metrics modules, gold datasets |
| `scripts/` | `trace_v7.py` (E2E smoke test), `validate_prompts.py`, `measure_cps.py` |
| `tests/` | Unit and integration tests (`pytest -m unit`) |

---

## Adding a New Node

1. Create `src/v7/nodes/<name>.py` — thin orchestrator: read state → call function → write state. Put logic in `nlp_core.py` or `hard_gates.py`, not in the node itself.
2. Register the node in `src/v7/graph.py` (`graph.add_node`, `graph.add_edge`).
3. Write unit tests in `tests/test_<name>.py` using `unittest.mock` for DI dependencies.

---

## Further Reading

- [v7-how-it-works.md](./v7-how-it-works.md) — detailed walkthrough of every node, hard gates, and threshold calibration
- [triage-how-it-works.md](./triage-how-it-works.md) — deep dive into `evaluate_triage` metrics and 3-way routing
