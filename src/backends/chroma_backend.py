"""Chroma backend — wraps existing src/vector_store.py."""

from __future__ import annotations

from typing import Iterator

from langchain_core.documents import Document


class ChromaBackend:
    """Implements VectorStoreBackend protocol over the legacy Chroma helpers."""

    def __init__(self, load_existing: bool = True) -> None:
        if load_existing:
            from src.indexing.vector_store import load_vector_store

            self._vs = load_vector_store()
        else:
            self._vs = None  # populated by create()

    def create(self, chunks: list[Document]) -> "ChromaBackend":
        """Build a new Chroma index from chunks. Used by index.py."""
        from src.indexing.vector_store import create_vector_store

        self._vs = create_vector_store(chunks)
        return self

    def similarity_search_with_score(
        self, query: str, k: int = 10, filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        return self._vs.similarity_search_with_score(query, k=k, filter=filter)

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        return self._vs.add_texts(texts=texts, metadatas=metadatas or [])

    def iter_all_documents(self, page_size: int = 1000) -> Iterator[dict]:
        # Batch via offset/limit so a large collection is neither truncated by
        # an implicit backend cap nor fully materialised in memory at once.
        offset = 0
        while True:
            data = self._vs.get(
                include=["metadatas", "documents"],
                limit=page_size,
                offset=offset,
            )
            documents = data.get("documents") or []
            if not documents:
                break
            metadatas = data.get("metadatas") or []
            for i, text in enumerate(documents):
                meta = metadatas[i] if i < len(metadatas) else None
                yield {"text": text, "metadata": meta or {}}
            if len(documents) < page_size:
                break
            offset += page_size

    def count(self) -> int:
        return self._vs._collection.count()

    def get_by_filter(self, where: dict, limit: int = 500) -> list[Document]:
        """Return ALL docs matching ``where``.

        ``limit`` is the page size, not a hard cap: results are paginated via
        offset until a page comes back short. Sources with > limit chunks
        (e.g. ТК РФ, > 500) are returned in full instead of being truncated.
        """
        from src.indexing.chroma_helpers import chroma_results_to_documents

        # Chroma needs explicit $and wrapper for multi-condition filters
        if len(where) > 1 or any(isinstance(v, dict) for v in where.values()):
            conditions = []
            for k, v in where.items():
                if isinstance(v, dict):
                    for op, val in v.items():
                        conditions.append({k: {op: val}})
                else:
                    conditions.append({k: v})
            chroma_where = (
                {"$and": conditions} if len(conditions) > 1 else conditions[0]
            )
        else:
            chroma_where = where

        docs: list[Document] = []
        offset = 0
        while True:
            result = self._vs.get(where=chroma_where, limit=limit, offset=offset)
            page = chroma_results_to_documents(result)
            docs.extend(page)
            if len(page) < limit:
                break
            offset += limit
        return docs
