"""Integration tests for src/v7/bridge.py — exercises real ChromaBackend."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_init_v7_from_chroma_with_real_backend(tmp_path, monkeypatch):
    """Smoke test: init_v7_pipeline works against a real ChromaBackend."""
    from langchain_core.documents import Document

    from src.backends.chroma_backend import ChromaBackend
    from src.v7.bridge import init_v7_pipeline

    import src.v7.nlp_core as nlp_core_mod
    import src.v7.nodes.rag_simple as rag_simple_mod

    monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "test_chroma"))
    # Capture module globals before init so monkeypatch restores them after the test.
    # init_v7_pipeline injects into rag_simple AND populates nlp_core._bm25_index —
    # both must be restored to avoid contaminating subsequent unit tests.
    monkeypatch.setattr(rag_simple_mod, "_vector_search", rag_simple_mod._vector_search)
    monkeypatch.setattr(rag_simple_mod, "_reranker_fn", rag_simple_mod._reranker_fn)
    monkeypatch.setattr(rag_simple_mod, "_expand_fn", rag_simple_mod._expand_fn)
    monkeypatch.setattr(nlp_core_mod, "_bm25_index", nlp_core_mod._bm25_index)

    backend = ChromaBackend(load_existing=False)
    backend.create(
        [
            Document(
                page_content="охрана труда",
                metadata={"source": "test.pdf", "chunk_id": 1},
            ),
            Document(
                page_content="инструктаж",
                metadata={"source": "test.pdf", "chunk_id": 2},
            ),
        ]
    )
    # No assertion needed — just verify no exception in iter_all_documents path
    init_v7_pipeline(backend, llm_provider=None)
