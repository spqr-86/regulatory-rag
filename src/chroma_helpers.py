"""Shared helpers for ChromaDB operations — deduplicates query logic."""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def chroma_results_to_documents(result: Optional[dict]) -> List[Document]:
    """Convert raw Chroma .get() result dict to a list of Documents."""
    if not result or not result.get("documents"):
        return []
    docs = []
    # result["documents"] is List[str]
    # result["metadatas"] is List[dict]
    documents = result["documents"]
    metadatas = result.get("metadatas")

    for i, text in enumerate(documents):
        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def query_chunks_by_range(vs, source: str, start: int, end: int) -> List[Document]:
    """Query Chroma for chunks in [start, end] range for a given source.

    Returns Documents sorted by chunk_id. Returns [] on error.

    TODO: When a non-Chroma backend lands, refactor to use
    `backend.get_by_filter({"source": source, "chunk_id": {"$gte": ..., "$lte": ...}})`
    — see src/backends/vector_store.py protocol.
    """
    try:
        result = vs.get(
            where={
                "$and": [
                    {"source": source},
                    {"chunk_id": {"$gte": start}},
                    {"chunk_id": {"$lte": end}},
                ]
            }
        )
        docs = chroma_results_to_documents(result)
        docs.sort(key=lambda x: x.metadata.get("chunk_id", 0))
        return docs
    except Exception as e:
        logger.error(
            "Error querying chunks range %d-%d for %s: %s", start, end, source, e
        )
        return []
