"""RAG Pipeline v7 — modular implementation."""

from __future__ import annotations

from src.ers_rag.bridge import init_ers_rag_from_chroma, make_vector_search_fn
from src.ers_rag.config import V7Config, v7_config
from src.ers_rag.graph import build_graph
from src.ers_rag.state_types import (
    ALLOWED_FILTER_KEYS,
    MAX_VERIFY_ITERATIONS,
    HardGateResult,
    Intent,
    RAGState,
    RetrievalAttempt,
    RetrievalPlan,
    SufficiencyResult,
    TriageCategory,
    VerificationResult,
    VerifierVerdict,
)

__all__ = [
    "ALLOWED_FILTER_KEYS",
    "build_graph",
    "init_ers_rag_from_chroma",
    "make_vector_search_fn",
    "HardGateResult",
    "Intent",
    "MAX_VERIFY_ITERATIONS",
    "RAGState",
    "RetrievalAttempt",
    "RetrievalPlan",
    "SufficiencyResult",
    "TriageCategory",
    "V7Config",
    "VerificationResult",
    "VerifierVerdict",
    "v7_config",
]
