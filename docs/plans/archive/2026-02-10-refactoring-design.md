# Refactoring Design: Incremental Cleanup by Layers

**Date**: 2026-02-10
**Approach**: Incremental bottom-up, 5 phases, each phase = working system

---

## Phase 1: Extract Utilities & Remove Duplication

**Goal**: Remove duplicated code, create reusable functions, no logic changes.

### New file: `src/parsers.py`

Extract from `multiagent_rag.py`:
- `parse_json_from_text()` — JSON parsing from LLM responses
- `extract_text_from_content()` — text extraction from mixed content
- `parse_status_block()` — status block parsing
- `parse_search_results()` — search results parsing

### New file: `src/chroma_helpers.py`

Extract from `agent_tools.py`:
- `query_chunks_by_range(vs, source, start, end)` — single method replacing two identical ChromaDB range queries
- `chroma_results_to_documents(result)` — result-to-Document conversion (also duplicated twice)

### Refactor `app.py`

- `render_proof_images(text)` — single function replacing two copies of regex + st.image
- Common error handler for agent/chain modes

### Refactor `final_chain.py`

- `build_retriever(vector_store, bm25, llm, query_expansion=True)` — single factory replacing duplicated ApplicabilityRetriever + reranker creation

### Expected result

~120 lines of duplicated code removed. `multiagent_rag.py` goes from ~587 to ~350 lines. All existing tests pass without changes.

---

## Phase 2: Fix Latency & Inefficiencies

**Goal**: Remove wasted work, cache stable data, speed up hot paths.

### `agent_tools.py` — move load_vector_store() out of loop

Currently called inside nested `for source ... for range ...` loop. Move to top of `search_documents()` — called once per search. Expected: 10-50x speedup for multi-range queries.

### `multiagent_rag.py` — cache glossary

Currently `_load_glossary()` and regex compilation happen on every workflow creation. Fix: `@lru_cache` on glossary load + pre-compiled regex patterns stored as module attribute. Load once, reuse.

### `final_chain.py` — lazy BM25 loading

Currently `vector_store.get()` loads all documents into memory for BM25 index. Fix: do it once in `load_resources()` (already cached via `@st.cache_resource`), reuse same BM25 for agent_retriever instead of rebuilding.

### `app.py` — remove redundant retrieval

Currently does separate `hybrid_retriever.invoke()` after agent already found documents. Fix: pass retrieved sources from agent state directly to UI.

### Expected result

Noticeable latency reduction per query, especially for complex searches with multiple sources.

---

## Phase 3: Remove Global State & Hardcoded Constants

**Goal**: Make code testable, thread-safe, configurable without code changes.

### `agent_tools.py` — replace global state with ToolContext

Currently `_retriever`, `_search_call_count`, `_visual_proof_call_count` are module variables. Breaks tests (state leaks), no parallelism.

Fix: create `ToolContext` class:
```python
class ToolContext:
    retriever: BaseRetriever
    vector_store: Chroma
    search_call_count: int = 0
    visual_proof_call_count: int = 0
```

Pass via LangGraph state or closure. Each workflow invocation gets isolated context.

### Constants to `config/settings.py`

Move all hardcoded values:
- `THINKING_BUDGET = 8192`, `THINKING_VERIFIER = 1024` from `multiagent_rag.py`
- `MAX_AGENT_STEPS = 16`, `MAX_REVISIONS = 1` from `multiagent_rag.py`
- `MAX_SEARCH_CALLS = 2`, `MAX_VISUAL_PROOF_CALLS = 1` from `agent_tools.py`

All become pydantic Settings fields with defaults, overridable via env variables.

### `multiagent_rag.py` — tools via constructor

Instead of hardcoded `tools = [search_documents, visual_proof]`, accept tools as `MultiAgentRAG.__init__(tools=None)`. Default = standard set, but custom tools can be passed for tests and future features.

### Expected result

Code is testable via DI, configurable via env, ready for parallel execution.

---

## Phase 4: Clean LLM Factory & Retriever Architecture

**Goal**: Simplify adding new LLM providers and retriever configurations.

### `llm_factory.py` — provider registry

Replace if/else chains with factory registry:
```python
LLM_PROVIDERS = {
    "openai": _create_openai_llm,
    "gemini": _create_gemini_llm,
}
```

Adding new provider = one function + one dict entry. Same for embeddings. Gemini AFC monkey-patch stays but is isolated inside `_create_gemini_llm()` with comment explaining why.

### `vector_store.py` — deduplicate Chroma init

Extract `_create_chroma_instance(embeddings)` — shared helper for `create_vector_store()` and `load_vector_store()`.

### `applicability_retriever.py` — cache query expansion

Add instance-level dict cache: `self._expansion_cache[query] = expanded_queries`. Saves LLM call (~1-2s) for identical queries within session.

### `load_vector_store()` — singleton

Wrap in `@lru_cache` or lazy module-level variable. One instance per process.

### Expected result

Adding new LLM provider = 1 function instead of editing 3 places. Fewer LLM calls on repeated queries.

---

## Phase 5: Simplify `app.py` & Final Cleanup

**Goal**: Decouple UI layer, remove remaining inefficiencies, final polish.

### `app.py` — separation of concerns

Split into:
- `src/ui_helpers.py` — `render_proof_images(text)`, `render_sources(docs)`, `handle_error(error, mode)`. Pure functions, easy to test.
- `app.py` keeps only Streamlit wiring: layout, sidebar, chat loop. ~150 lines instead of 279.

### `multiagent_rag.py` — pass sources through state

Add `retrieved_sources: list[Document]` to `AgentState`. Agent accumulates found documents during `search_documents`. Output = ready source list, UI displays directly without re-retrieval.

### `file_handler.py` — extract nested functions

`finalize_chunk()` and `update_bbox()` defined inside another function via `nonlocal`. Move to `ChunkBuilder` class methods or module-level functions with explicit parameters.

### Final cleanup

- Remove unused imports (ruff will catch them)
- Check if legacy `final_chain.py` is still needed — if not, mark as deprecated
- `black . && ruff check . --fix` + run all tests

### Expected result

Clean UI layer, sources without extra calls, readable chunking code.

---

## Phase Order & Dependencies

```
Phase 1 (duplication)
  └→ Phase 2 (latency) — uses helpers from Phase 1
       └→ Phase 3 (global state) — cleaner after Phase 2
            └→ Phase 4 (LLM factory) — independent but benefits from Phase 3 patterns
                 └→ Phase 5 (UI + cleanup) — final layer, depends on sources from Phase 3
```

Each phase is a separate branch/PR. System stays working after each merge.
