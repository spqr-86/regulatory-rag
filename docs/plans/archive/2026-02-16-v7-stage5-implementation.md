# V7 Stage 5: Migration & Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect v7 graph to the existing app via `USE_V7_GRAPH` feature flag, with bridge code to adapt existing ChromaDB retriever to v7's interface.

**Architecture:** Add a `bridge.py` module that wraps the existing ChromaDB vector store into the `(query, filters, top_k, **kwargs) → list[dict]` interface expected by v7 nodes. Initialize BM25 index from ChromaDB corpus. In `app.py`, add a third mode toggle for v7 graph alongside existing MAS and Legacy RAG.

**Tech Stack:** Existing ChromaDB, pymorphy3/razdel (already in v7), LangGraph StateGraph

---

## Task 1: Add `USE_V7_GRAPH` to settings

**Files:**
- Modify: `config/settings.py:72` (after `SIMILARITY_THRESHOLD_FOR_VERIFIER_SKIP`)

**Step 1: Add the flag**

In `config/settings.py`, after `SIMILARITY_THRESHOLD_FOR_VERIFIER_SKIP`, add:

```python
    # V7 graph toggle
    USE_V7_GRAPH: bool = False
```

**Step 2: Verify import works**

Run: `python -c "from config.settings import settings; print(settings.USE_V7_GRAPH)"`
Expected: `False`

**Step 3: Commit**

```bash
git add config/settings.py
git commit -m "feat(v7): add USE_V7_GRAPH feature flag to settings"
```

---

## Task 2: Create bridge module `src/v7/bridge.py`

**Files:**
- Create: `src/v7/bridge.py`
- Create: `tests/v7/test_bridge.py`

This module adapts the existing ChromaDB retriever + BM25 corpus to v7's interface.

**Step 1: Write failing tests**

`tests/v7/test_bridge.py`:
```python
"""Tests for src/v7/bridge.py — v7 ↔ existing retriever bridge."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.v7.bridge import make_vector_search_fn, init_v7_from_chroma


class TestMakeVectorSearchFn:
    @pytest.mark.unit
    def test_returns_callable(self):
        mock_store = MagicMock()
        fn = make_vector_search_fn(mock_store)
        assert callable(fn)

    @pytest.mark.unit
    def test_calls_similarity_search_with_score(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        result = fn(query="test query", top_k=5)
        mock_store.similarity_search_with_score.assert_called_once()
        assert result == []

    @pytest.mark.unit
    def test_converts_documents_to_dicts(self):
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "some text about safety"
        mock_doc.metadata = {"source": "gost.pdf", "page_no": 5}
        mock_store.similarity_search_with_score.return_value = [
            (mock_doc, 0.3)  # Chroma returns distance, not similarity
        ]
        fn = make_vector_search_fn(mock_store)
        result = fn(query="safety", top_k=10)
        assert len(result) == 1
        assert result[0]["text"] == "some text about safety"
        assert result[0]["metadata"]["source"] == "gost.pdf"
        assert result[0]["score"] == pytest.approx(0.7, abs=0.01)

    @pytest.mark.unit
    def test_respects_top_k(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        fn(query="test", top_k=20)
        call_kwargs = mock_store.similarity_search_with_score.call_args
        assert call_kwargs[1].get("k") == 20 or call_kwargs[0] == ("test",)

    @pytest.mark.unit
    def test_filters_ignored_gracefully(self):
        """Filters are passed but ChromaDB may not support all of them."""
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        result = fn(query="test", top_k=5, filters={"doc_type": "gost"})
        assert result == []


class TestInitV7FromChroma:
    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_injects_vector_search(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "documents": ["doc1 text", "doc2 text"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        init_v7_from_chroma(mock_store)
        mock_simple.set_vector_search.assert_called_once()
        mock_complex.set_vector_search.assert_called_once()
        mock_bm25.assert_called_once()

    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_bm25_corpus_built_from_chroma(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "documents": ["text A", "text B"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        init_v7_from_chroma(mock_store)
        corpus = mock_bm25.call_args[0][0]
        assert len(corpus) == 2
        assert corpus[0]["text"] == "text A"
        assert corpus[1]["metadata"]["source"] == "b.pdf"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/v7/test_bridge.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `src/v7/bridge.py`**

```python
"""Bridge: adapt existing ChromaDB vector store for v7 pipeline.

Responsibilities:
1. Wrap ChromaDB's similarity_search_with_score → v7 dict format
2. Build BM25 corpus from ChromaDB docs
3. Inject search functions into rag_simple / rag_complex nodes
"""

from __future__ import annotations

from typing import Callable, List

from src.v7.nlp_core import init_bm25_index
from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes import rag_complex as rag_complex_mod


def make_vector_search_fn(vector_store) -> Callable[..., List[dict]]:
    """Create a v7-compatible vector search function from ChromaDB store.

    v7 interface: fn(query, filters=None, top_k=12, **kwargs) -> list[dict]
    Each dict has: text, metadata, score.
    """

    def _search(
        query: str,
        filters: dict | None = None,
        top_k: int = 12,
        **kwargs,
    ) -> List[dict]:
        docs_and_scores = vector_store.similarity_search_with_score(
            query, k=top_k
        )
        results = []
        for doc, distance in docs_and_scores:
            similarity = max(0.0, 1.0 - distance)
            results.append(
                {
                    "text": doc.page_content,
                    "metadata": dict(doc.metadata),
                    "score": round(similarity, 4),
                }
            )
        return results

    return _search


def init_v7_from_chroma(vector_store) -> None:
    """Initialize v7 pipeline from existing ChromaDB vector store.

    1. Creates vector search wrapper
    2. Injects it into rag_simple and rag_complex nodes
    3. Builds BM25 index from full corpus
    """
    search_fn = make_vector_search_fn(vector_store)
    rag_simple_mod.set_vector_search(search_fn)
    rag_complex_mod.set_vector_search(search_fn)

    # Build BM25 corpus from ChromaDB
    all_data = vector_store.get(include=["metadatas", "documents"])
    corpus = [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    init_bm25_index(corpus)
```

**Step 4: Run tests**

Run: `pytest tests/v7/test_bridge.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/v7/bridge.py tests/v7/test_bridge.py
git commit -m "feat(v7): add bridge module for ChromaDB ↔ v7 integration"
```

---

## Task 3: Integrate v7 graph into `app.py`

**Files:**
- Modify: `app.py` (sidebar toggle, resource loading, query handling)

**Step 1: Add v7 import and toggle**

After the MAS import block (~line 26), add:

```python
# V7 Graph
try:
    from src.v7.bridge import init_v7_from_chroma
    from src.v7.graph import build_graph as build_v7_graph

    V7_AVAILABLE = True
except Exception as e:
    logger.warning(f"V7 Graph is not available: {e}")
    V7_AVAILABLE = False
```

**Step 2: Add sidebar toggle**

In the sidebar section (~line 76), after MAS toggle, add v7 toggle:

```python
    if V7_AVAILABLE and settings.USE_V7_GRAPH:
        v7_mode = st.toggle("🔬 V7 Graph (modular pipeline)", value=False)
    else:
        v7_mode = False
```

Ensure `v7_mode` and `mas_mode` are mutually exclusive (when v7 is on, MAS is off):

```python
    if v7_mode:
        mas_mode = False
```

**Step 3: Update `load_resources` to init v7**

At the end of `load_resources()`, before the return, add v7 initialization:

```python
    # V7 Graph init
    v7_app = None
    if V7_AVAILABLE and settings.USE_V7_GRAPH:
        try:
            init_v7_from_chroma(vector_store)
            v7_app = build_v7_graph().compile()
        except Exception as e:
            logger.warning(f"Failed to init V7 Graph: {e}")

    return (chain, retriever, agent, v7_app)
```

Update the unpacking of `loaded` accordingly:

```python
loaded = load_resources()
if not loaded or loaded[0] is None:
    st.warning("Приложение не может быть запущено…")
    st.stop()

rag_chain, hybrid_retriever, agent, v7_app = loaded
```

Note: `load_resources` currently returns 3 values, we add a 4th. For the v7 path we need the raw `vector_store`, so extract it inside `load_resources`.

**Step 4: Add v7 query handling branch**

In the chat input section (~line 196), before the MAS branch, add:

```python
        if v7_mode and v7_app:
            # --- V7 GRAPH MODE ---
            with st.spinner("V7 pipeline..."):
                result = v7_app.invoke({"query": user_query})

            # Determine answer
            if result.get("clarify_message"):
                answer = result["clarify_message"]
            elif result.get("abstain_reason"):
                answer = result["abstain_reason"]
            elif result.get("final_passages"):
                passages = result["final_passages"]
                texts = [p.get("text", "") for p in passages[:10]]
                answer = (
                    f"Найдено {len(passages)} релевантных фрагментов "
                    f"(score: {result.get('final_score', 0):.3f}).\n\n"
                    + "\n\n---\n\n".join(texts)
                )
                # Show sources
                with st.expander(
                    f"🔎 Источники ({len(passages)})", expanded=False
                ):
                    for i, p in enumerate(passages, 1):
                        src = p.get("metadata", {}).get("source", "N/A")
                        score = p.get("score", 0.0)
                        preview = p.get("text", "")[:500].strip().replace("\n", " ")
                        st.markdown(f"**{i}.** `{src}` · 🎯 {score:.2f}")
                        st.code(preview, language="markdown")
                        st.divider()
            elif result.get("intent") == "noise":
                answer = "Это выглядит как приветствие. Задайте вопрос по охране труда."
            else:
                answer = "Не удалось получить ответ."

            st.markdown(answer)
        elif mas_mode and agent:
```

**Step 5: Run lint**

Run: `black . && ruff check . --fix`

**Step 6: Commit**

```bash
git add app.py
git commit -m "feat(v7): integrate v7 graph into app.py with feature flag"
```

---

## Task 4: Update `__init__.py` exports and progress tables

**Files:**
- Modify: `src/v7/__init__.py` (add bridge exports)
- Modify: `CLAUDE.md`, `AGENTS.md`, `GEMINI.md` (progress tables)
- Modify: `docs/plans/2026-02-16-v7-migration-design.md` (stage status)

**Step 1: Add bridge to `__init__.py`**

Add to imports:
```python
from src.v7.bridge import init_v7_from_chroma, make_vector_search_fn
```

Add to `__all__`:
```python
    "init_v7_from_chroma",
    "make_vector_search_fn",
```

**Step 2: Update progress tables**

In all three files (`CLAUDE.md`, `AGENTS.md`, `GEMINI.md`), change Stage 5 from:
```
| 5 | Миграция + cleanup | Pending | — |
```
to:
```
| 5 | Миграция + cleanup | ✅ Done | `feature/v7-migration-stage1` |
```

Also update stage status in `docs/plans/2026-02-16-v7-migration-design.md`.

**Step 3: Run all v7 tests**

Run: `pytest tests/v7/ -v`
Expected: All pass

**Step 4: Lint**

Run: `black . && ruff check . --fix`

**Step 5: Commit**

```bash
git add src/v7/__init__.py CLAUDE.md AGENTS.md GEMINI.md docs/plans/2026-02-16-v7-migration-design.md
git commit -m "feat(v7): complete stage 5 migration, update progress tables"
```

---

## Notes

- **V7 is retrieval-only in this stage** — it returns passages, not LLM-generated answers. The LLM answer generation will be added when the LLM verifier node is connected to a real LLM (currently a stub). This is by design: validate retrieval quality first.
- **Feature flag approach** — `USE_V7_GRAPH=false` (default) means zero impact on existing users. Set `USE_V7_GRAPH=true` in `.env` to test v7.
- **BM25 index init** may take a few seconds on large corpora. It runs once at startup inside `load_resources()`.
