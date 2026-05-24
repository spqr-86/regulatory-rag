"""Shared types and enums for the multi-agent RAG system."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, TypedDict


class RAGStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    PARTIAL = "PARTIAL"


class VerifyStatus(str, Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"


class RouteType(str, Enum):
    CHITCHAT = "chitchat"
    OUT_OF_SCOPE = "out_of_scope"
    RAG = "rag"
    RAG_SIMPLE = "rag_simple"
    RAG_COMPLEX = "rag_complex"


class ChunkInfo(TypedDict):
    content: str
    source: str
    page_no: Optional[int]
    bbox: Optional[List[float]]
    visual_text: Optional[str]
    similarity: Optional[float]
