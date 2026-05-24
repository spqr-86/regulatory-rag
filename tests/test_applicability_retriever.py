"""Tests for ApplicabilityRetriever query expansion + ensemble behaviour."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import VectorStore

from src.indexing.applicability_retriever import ApplicabilityRetriever


def _make_doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=dict(meta))


@pytest.fixture
def mock_vector_store():
    vs = MagicMock(spec=VectorStore)
    # similarity_search_with_score returns list of (Document, distance)
    vs.similarity_search_with_score.side_effect = lambda q, **kw: [
        (_make_doc(f"vec-{q}-1", chunk_id=f"{q}-1"), 0.2),
        (_make_doc(f"vec-{q}-2", chunk_id=f"{q}-2"), 0.4),
    ]
    return vs


class _FakeBM25(BM25Retriever):
    """Thin BM25 stub that survives pydantic validation and returns a fixed doc."""

    @classmethod
    def make(cls) -> "BM25Retriever":
        bm = cls.from_texts(["bm25-result"])
        return bm

    def _get_relevant_documents(self, query: str, *, run_manager=None):  # type: ignore[override]
        return [_make_doc("bm25-result", chunk_id="bm25-1")]


@pytest.fixture
def mock_bm25():
    return _FakeBM25.make()


@pytest.fixture
def mock_llm():
    # Simulate an LLM that returns three reformulations on three lines.
    return RunnableLambda(
        lambda _: AIMessage(content="первая вариация\nвторая вариация\nтретья вариация")
    )


@patch("src.indexing.applicability_retriever.PromptManager")
def test_generate_queries_uses_line_parser(
    mock_pm_cls, mock_vector_store, mock_bm25, mock_llm
):
    """Three reformulations parsed via LineListOutputParser + original prepended."""
    mock_pm_cls.return_value.render.return_value = "TEMPLATE rendered"

    retriever = ApplicabilityRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=mock_bm25,
        llm=mock_llm,
    )
    queries = retriever._generate_queries("исходный")
    assert queries[0] == "исходный"
    assert "первая вариация" in queries
    assert "вторая вариация" in queries
    assert "третья вариация" in queries
    assert len(queries) <= 4


@patch("src.indexing.applicability_retriever.PromptManager")
def test_generate_queries_cached(mock_pm_cls, mock_vector_store, mock_bm25, mock_llm):
    """Second call for the same query hits the in-memory cache."""
    mock_pm_cls.return_value.render.return_value = "TEMPLATE rendered"
    spy_llm = MagicMock(wraps=mock_llm)
    # Build an llm-like Runnable that records invocations.
    calls = []

    def _llm_fn(_):
        calls.append(1)
        return AIMessage(content="alt1\nalt2\nalt3")

    retriever = ApplicabilityRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=mock_bm25,
        llm=RunnableLambda(_llm_fn),
    )
    _ = retriever._generate_queries("q")
    _ = retriever._generate_queries("q")
    assert len(calls) == 1
    del spy_llm  # silence unused


@patch("src.indexing.applicability_retriever.PromptManager")
def test_generate_queries_fallback_on_error(mock_pm_cls, mock_vector_store, mock_bm25):
    """If LLM/chain explodes, fall back to the original query alone."""
    mock_pm_cls.return_value.render.return_value = "TEMPLATE rendered"

    def _broken(_):
        raise RuntimeError("llm down")

    retriever = ApplicabilityRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=mock_bm25,
        llm=RunnableLambda(_broken),
    )
    assert retriever._generate_queries("оригинал") == ["оригинал"]


@patch("src.indexing.applicability_retriever.PromptManager")
def test_get_relevant_documents_merges_all_subqueries(
    mock_pm_cls, mock_vector_store, mock_bm25, mock_llm
):
    """Result list contains documents fetched for each generated subquery + BM25."""
    mock_pm_cls.return_value.render.return_value = "TEMPLATE rendered"

    retriever = ApplicabilityRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=mock_bm25,
        llm=mock_llm,
    )
    docs = retriever._get_relevant_documents("исходный")

    # vector_store called for original + 3 alternatives = 4 calls
    assert mock_vector_store.similarity_search_with_score.call_count == 4

    contents = {d.page_content for d in docs}
    # At least one doc from every subquery should survive dedup (distinct content).
    assert any(c.startswith("vec-исходный-") for c in contents)
    assert any(c.startswith("vec-первая вариация-") for c in contents)
    assert "bm25-result" in contents

    # similarity_score is populated from (1 - distance).
    for d in docs:
        if d.page_content.startswith("vec-"):
            assert "similarity_score" in d.metadata


@patch("src.indexing.applicability_retriever.PromptManager")
def test_query_expansion_disabled_skips_llm(
    mock_pm_cls, mock_vector_store, mock_bm25, mock_llm
):
    """When query_expansion=False, only the original query goes to vector_store."""
    mock_pm_cls.return_value.render.return_value = "TEMPLATE rendered"

    retriever = ApplicabilityRetriever(
        vector_store=mock_vector_store,
        bm25_retriever=mock_bm25,
        llm=mock_llm,
        query_expansion=False,
    )
    _ = retriever._get_relevant_documents("исходный")
    assert mock_vector_store.similarity_search_with_score.call_count == 1
