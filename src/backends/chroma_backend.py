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
        self, query: str, k: int = 10
    ) -> list[tuple[Document, float]]:
        return self._vs.similarity_search_with_score(query, k=k)

    def add_texts(
        self, texts: list[str], metadatas: list[dict] | None = None
    ) -> list[str]:
        return self._vs.add_texts(texts=texts, metadatas=metadatas or [])

    def iter_all_documents(self) -> Iterator[dict]:
        data = self._vs.get(include=["metadatas", "documents"])
        for text, meta in zip(data["documents"], data["metadatas"]):
            yield {"text": text, "metadata": meta or {}}

    def count(self) -> int:
        return self._vs._collection.count()

    def get_by_filter(self, where: dict, limit: int = 200) -> list[Document]:
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
        result = self._vs.get(where=chroma_where, limit=limit)
        return chroma_results_to_documents(result)
