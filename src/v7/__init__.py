"""RAG Pipeline v7 — modular implementation."""

from src.v7.bridge import init_v7_from_chroma, init_v7_pipeline, make_vector_search_fn
from src.v7.config import V7Config, v7_config
from src.v7.graph import build_graph
from src.v7.state_types import (
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
    "init_v7_pipeline",
    "init_v7_from_chroma",  # deprecated alias, kept for one release
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
