"""Tests for ChromaBackend (CARD-2.2)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_chroma import Chroma  # noqa: F401 — used as MagicMock spec
from langchain_core.documents import Document


@pytest.mark.unit
def test_similarity_search_delegates():
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)
        mock_vs.similarity_search_with_score.return_value = [
            (Document(page_content="t"), 0.9)
        ]
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        backend = ChromaBackend()
        results = backend.similarity_search_with_score("q", k=5)

        assert results == [(Document(page_content="t"), 0.9)]
        mock_vs.similarity_search_with_score.assert_called_once_with("q", k=5)


@pytest.mark.unit
def test_iter_all_documents_yields_dicts():
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)
        mock_vs.get.return_value = {
            "documents": ["doc1", "doc2"],
            "metadatas": [{"source": "a"}, {"source": "b"}],
        }
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        docs = list(ChromaBackend().iter_all_documents())

        assert docs == [
            {"text": "doc1", "metadata": {"source": "a"}},
            {"text": "doc2", "metadata": {"source": "b"}},
        ]


@pytest.mark.unit
def test_iter_all_documents_handles_none_metadata():
    """If a doc has metadata=None, iter_all_documents must yield empty dict."""
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)
        mock_vs.get.return_value = {
            "documents": ["doc1"],
            "metadatas": [None],
        }
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        docs = list(ChromaBackend().iter_all_documents())
        assert docs == [{"text": "doc1", "metadata": {}}]


@pytest.mark.unit
def test_count_delegates_to_collection():
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)
        mock_vs._collection = MagicMock()  # instance attr not in spec, set explicitly
        mock_vs._collection.count.return_value = 1973
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        assert ChromaBackend().count() == 1973


@pytest.mark.unit
def test_get_by_filter_wraps_multi_condition_with_and():
    """Multi-key where dict must be wrapped in $and for Chroma."""
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)
        mock_vs.get.return_value = {"documents": [], "metadatas": []}
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        backend = ChromaBackend()
        backend.get_by_filter(
            {
                "source": "doc.pdf",
                "chunk_id": {"$gte": 0, "$lte": 5},
            }
        )

        call_where = mock_vs.get.call_args.kwargs["where"]
        assert "$and" in call_where
        assert {"source": "doc.pdf"} in call_where["$and"]


@pytest.mark.unit
def test_get_by_filter_single_condition_passes_through():
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)
        mock_vs.get.return_value = {"documents": [], "metadatas": []}
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        ChromaBackend().get_by_filter({"source": "doc.pdf"})

        call_where = mock_vs.get.call_args.kwargs["where"]
        assert call_where == {"source": "doc.pdf"}


@pytest.mark.unit
def test_get_by_filter_paginates_all_pages():
    """get_by_filter must return ALL matching docs, not just the first page."""
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)

        def fake_get(where=None, limit=None, offset=0, **kwargs):
            # 5 total docs, page size 2 → pages: [0,1], [2,3], [4], []
            all_docs = [f"d{i}" for i in range(5)]
            page = all_docs[offset : offset + limit]
            return {
                "documents": page,
                "metadatas": [{"source": "big.pdf"}] * len(page),
            }

        mock_vs.get.side_effect = fake_get
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        docs = ChromaBackend().get_by_filter({"source": "big.pdf"}, limit=2)
        assert [d.page_content for d in docs] == ["d0", "d1", "d2", "d3", "d4"]


@pytest.mark.unit
def test_iter_all_documents_paginates():
    """iter_all_documents must batch via offset and return all docs."""
    with patch("src.indexing.vector_store.load_vector_store") as mock_load:
        mock_vs = MagicMock(spec=Chroma)

        def fake_get(limit=None, offset=0, **kwargs):
            all_docs = [f"d{i}" for i in range(3)]
            page = all_docs[offset : offset + limit]
            return {
                "documents": page,
                "metadatas": [{"source": "a"}] * len(page),
            }

        mock_vs.get.side_effect = fake_get
        mock_load.return_value = mock_vs

        from src.backends.chroma_backend import ChromaBackend

        docs = list(ChromaBackend().iter_all_documents())
        assert [d["text"] for d in docs] == ["d0", "d1", "d2"]


@pytest.mark.unit
def test_satisfies_protocol():
    from src.backends.chroma_backend import ChromaBackend
    from src.backends.vector_store import VectorStoreBackend

    with patch("src.indexing.vector_store.load_vector_store"):
        backend = ChromaBackend()
        assert isinstance(backend, VectorStoreBackend)


@pytest.mark.unit
def test_create_populates_vs():
    """ChromaBackend(load_existing=False).create(chunks) must build a new index."""
    from src.backends.chroma_backend import ChromaBackend

    with patch("src.indexing.vector_store.create_vector_store") as mock_create:
        mock_create.return_value = MagicMock()
        backend = ChromaBackend(load_existing=False)
        chunks = [Document(page_content="x", metadata={})]
        result = backend.create(chunks)

        mock_create.assert_called_once_with(chunks)
        assert result is backend  # chainable
