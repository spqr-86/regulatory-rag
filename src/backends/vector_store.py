"""Vector store factory and protocol.

Future backends (Qdrant, pgvector) implement VectorStoreBackend; callers do
not change — they go through get_vector_store_backend().
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

from langchain_core.documents import Document

from config.settings import settings


@runtime_checkable
class VectorStoreBackend(Protocol):
    """Minimum API every backend must implement."""

    def similarity_search_with_score(
        self, query: str, k: int = 10, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Top-k semantic search. Score is similarity (higher=better).

        ``filter`` is an optional Chroma-style metadata filter dict.
        """
        ...

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        """Insert documents. Returns assigned IDs."""
        ...

    def iter_all_documents(self) -> Iterator[dict]:
        """Yield every stored doc as {"text": str, "metadata": dict}.

        Used by BM25 corpus build in src/v7/bridge.py.
        """
        ...

    def count(self) -> int:
        """Total document count."""
        ...

    def get_by_filter(self, where: dict, limit: int = 500) -> list[Document]:
        """Metadata filter query. Returns ALL matching docs (paginated).

        `where` uses Chroma syntax: {"field": value} or
        {"field": {"$gte": N, "$lte": M}}. Backends translate as needed.
        `limit` is the page size, not a hard cap on the result count.
        """
        ...


def get_vector_store_backend(load_existing: bool = True) -> VectorStoreBackend:
    """Return the configured backend.

    Args:
        load_existing: True to load existing index, False to start fresh
            (used by index.py during reindexing).

    Raises:
        ValueError: If VECTOR_STORE points to an unsupported backend.
    """
    backend = settings.VECTOR_STORE.lower()
    if backend == "chroma":
        from src.backends.chroma_backend import ChromaBackend

        return ChromaBackend(load_existing=load_existing)
    raise ValueError(f"Unknown VECTOR_STORE={backend!r}. Currently supported: chroma")
