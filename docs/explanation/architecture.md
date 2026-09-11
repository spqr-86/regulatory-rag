# Architecture

Regulatory RAG answers Russian regulatory-compliance questions (ГОСТ, ТК РФ, СНиП, СП,
federal laws) by retrieving passages from an indexed corpus and synthesizing a cited
answer — or **explicitly abstaining** when retrieval confidence is low. The guiding
principle, and *why* the pipeline looks the way it does, is documented separately in
[Design decisions](./design-decisions.md).

All volatile numbers (models, thresholds, metrics, chunk count) live in one place:
[reference/FACTS.md](../reference/FACTS.md). This doc links there rather than repeating
values.

---

## Request flow

The pipeline is a deterministic LangGraph `StateGraph` (`src/v7/graph.py`), entry point
`intent_gate`. Given the same query and index, the path is always the same — there is no
LLM in the routing.

```
intent_gate ─(noise)──────→ END
            └(in-domain)──→ router ─(ambiguous)→ clarify_respond → END
                                   └(ok)────────→ rag_simple → evaluate_triage
                                                                   │
                                  (generate) ─────────────────────┤
                                                                   │            ┌─(generate)→ generate_answer → END
                                  (complex) ─────→ rag_complex → evaluate_complex┤
                                                                                └─(fail)→ abstain → END
```

There is **no** `llm_verifier` or `rewriter` stage — that subgraph was removed (see <!--freshness:ignore-->
[Design decisions §5](./design-decisions.md)). Retrieval nodes emit the terminal
`route_decision` / `route_reason` contract. Generation reads only `final_context`, which
`pack_context` expands, sanitizes, truncates, and validates before the graph commits to
answering.

---

## Nodes

| Node | File | What it does |
|------|------|--------------|
| `intent_gate` | `src/v7/nodes/intent_gate.py` | Regex noise filter **+ domain gate** (cosine of query to corpus centroid; OOS → END). Active when `DOMAIN_GATE_THRESHOLD > 0`. |
| `router` | `src/v7/nodes/router.py` | Classifies the query, builds a retrieval plan, expands `active_query` via the term glossary and multi-query. Short/ambiguous → clarification. |
| `clarify_respond` | `src/v7/nodes/router.py` | Returns a clarification request for under-specified queries, then ends. |
| `rag_simple` | `src/v7/nodes/rag_simple.py` | Fast hybrid retrieval (vector + BM25, RRF merge) → CrossEncoder rerank → top-K passages. |
| `evaluate_triage` | `src/v7/nodes/evaluate_triage.py` | Deterministic sufficiency gate → terminal `route_decision` and `route_reason`: a three-metric hard gate plus a structured `triage_gap` that pulls in cross-referenced clauses before escalating. Enumeration intent forces rag_complex. |
| `rag_complex` | `src/v7/nodes/rag_complex.py` | Deep retrieval (larger top-K + MMR), multiple attempts, merges all. |
| `evaluate_complex` | `src/v7/nodes/evaluate_complex.py` | Hard gates on merged passages; pass → generate, fail → abstain. |
| `generate_answer` | `src/v7/nodes/generate_answer.py` | Synthesizes the answer from final passages via the active prompt template. |
| `abstain` | `src/v7/nodes/abstain.py` | Explicit refusal when retrieval stays poor. |

`visual_enrichment` and `pack_context` are no longer graph nodes. Both evaluators run the
same internal `enrich → pack → validate → decide` pipeline before emitting their terminal
route contract. This makes the evidence in `final_context` exactly the evidence that was
validated for generation.

Node list and thresholds: see [FACTS](../reference/FACTS.md). The LLM provider is
configurable per path; current production values are in FACTS.

---

## Sufficiency gate

Triage runs a **hard gate** (`src/v7/hard_gates.py`, `check_full_triage()`) that takes the
retrieved passages and a plan (thresholds), and checks **three conditions
simultaneously** — all must hold, else `sufficient = False`:

| Condition | Checks | Source of the number |
|---|---|---|
| `above_threshold` | top vector score ≥ threshold | `HARD_GATE_THRESHOLD` / `COMPLEX_THRESHOLD` |
| `enough_evidence` | passage count ≥ min_passages | `MIN_PASSAGES` / `COMPLEX_MIN_PASSAGES` |
| `keyword_overlap_ok` | share of query keywords found in passages ≥ floor | `COMPLEX_MIN_KW_OVERLAP` |

The score is **cosine similarity** from ChromaDB (0–1). BM25 scores are not used for
thresholds (they are unscaled). In `rag_simple`, `top_score` is taken only from the
vector results, not the merged list — see [Design decisions §3](./design-decisions.md)
for why the reranker score must not be used here. Threshold values: [FACTS](../reference/FACTS.md#thresholds).

Triage internals (3-way classification, enumeration handling): [triage.md](./triage.md).

---

## Codebase map

| Path | Contents |
|------|----------|
| `src/v7/` | Pipeline nodes (`nodes/`), graph assembly (`graph.py`), `state_types`, `hard_gates`, `nlp_core`, `domain_gate`, `cross_ref`, `bridge.py` (DI adapter) |
| `src/infra/` | LLM factory, prompt manager, parsers, shared types |
| `src/indexing/` | Document processor (HybridChunker), Chroma helpers, vector store, applicability retriever |
| `src/backends/` | `VectorStoreBackend` protocol + ChromaDB implementation |
| `config/` | `settings.py` (pydantic-settings), `term_glossary.yaml` |
| `prompts/` | Jinja2 templates + `registry.yaml` (3 live families) |
| `eval/` | `run_v7_eval.py`, metrics modules, gold datasets |
| `scripts/` | `trace_v7.py` (E2E smoke test), `measure_cps.py`, `check_docs.py` (docs freshness) |
| `tests/` | Unit and integration tests (`pytest -m unit`) |

## Pluggable backends

LLM and vector store are reached through factory layers (`src/infra/llm_factory.py`,
`src/backends/`). Adding a provider is one file plus one registry entry — pipeline code
does not change. The graph injects LLM and vector-search functions via `bridge.py`
(dependency injection), so swapping Gemini ↔ OpenAI ↔ DeepSeek, or Chroma ↔ another
store, never touches node logic.

---

## Further reading

- [Design decisions](./design-decisions.md) — why deterministic gates, abstain-over-hallucinate, CrossEncoder, and more
- [Triage](./triage.md) — `evaluate_triage` deep dive
- [How-to: add a node](../how-to/add-a-node.md)
- [Reference: FACTS](../reference/FACTS.md) · [data pipeline](../reference/data-pipeline.md)
