# src/v7/config.py
"""V7 RAG pipeline — configuration.

All thresholds and limits externalized via pydantic-settings.
Env vars use V7_ prefix (e.g. V7_HARD_GATE_THRESHOLD=0.80).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class V7Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="V7_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Simple path (rag_simple) ───────────────────────────────────────────
    # ChromaDB L2→similarity formula: 1/(1+d). Relevant docs score ~0.44–0.56.
    # Acceptance threshold: >= 0.50 = clearly relevant (direct to generate).
    # Soft floor: [0.38, 0.50) = uncertain → borderline → rag_complex.
    # Below 0.38 = garbage → clearly_bad → rag_complex.
    HARD_GATE_THRESHOLD: float = 0.50  # plan.threshold (similarity acceptance gate)
    TRIAGE_SOFT_THRESHOLD: float = 0.38  # plan.borderline_threshold (floor)
    MIN_PASSAGES: int = 5  # plan.min_passages
    # Russian adj/noun lemmas differ ("лестничный"≠"лестница") → keep low
    MIN_KEYWORD_OVERLAP_ACTIVE: float = 0.15  # plan.min_keyword_overlap
    MAX_SINGLE_DOC_RATIO: float = 0.8  # plan.max_single_doc_ratio
    SIMPLE_TOP_K: int = 12  # plan.top_k
    SIMPLE_TIMEOUT_MS: int = 250  # plan.timeout_ms

    # ── Complex path (rag_complex) ────────────────────────────────────────
    COMPLEX_THRESHOLD: float = 0.35  # min threshold for complex (floor, ≤ simple)
    COMPLEX_MIN_PASSAGES: int = 8  # plan.min_passages for slow path
    # Russian morphology: adjective and noun lemmas often differ for the same root.
    # 0.20 is enough to detect real OOS (0-overlap) without blocking domain queries.
    COMPLEX_MIN_KW_OVERLAP: float = 0.20  # plan.min_keyword_overlap for slow path
    COMPLEX_MAX_SINGLE_DOC_RATIO: float = 0.7
    COMPLEX_BORDERLINE_THRESHOLD: float = 0.30
    COMPLEX_TOP_K: int = 60  # plan.top_k for slow path
    RERANK_CANDIDATE_CAP: int = 100  # hard limit on CrossEncoder input; prevents O(n) blowup on large section fetches
    COMPLEX_TIMEOUT_MS: int = 1200

    # ── Retrieval engine ──────────────────────────────────────────────────
    RRF_K: int = 60
    MMR_LAMBDA: float = 0.7

    # ── Keyword overlap (dual) ────────────────────────────────────────────
    MIN_KEYWORD_OVERLAP_ORIGINAL: float = 0.10  # drift detection, even looser

    # ── LLM & Limits ──────────────────────────────────────────────────────
    MAX_REWRITE_ATTEMPTS: int = 2
    MAX_CHUNKS_FOR_LLM: int = 10
    VERIFIER_CONFIDENCE_ANCHOR: float = 0.7  # plan.min_verifier_confidence

    # ── Anti-injection ────────────────────────────────────────────────────
    COVERAGE_DROP_PCT: float = 0.30
    MAX_INPUT_LENGTH: int = 2000
    BLOCKED_PATTERNS: list[str] = [
        "ignore previous",
        "system prompt",
        "you are now",
    ]

    # ── V8 Evidence Assess ─────────────────────────────────────────────────
    V8_ENABLE_EVIDENCE_ASSESS: bool = False
    V8_EVIDENCE_ANSWER_RERANKER_TOP1: float = 0.6
    V8_EVIDENCE_ANSWER_COVERAGE: float = 0.6
    V8_EVIDENCE_ABSTAIN_RERANKER_TOP1: float = 0.2
    V8_EVIDENCE_ABSTAIN_COVERAGE: float = 0.2
    V8_SIMPLE_RERANK_TOP_K: int = 5

    # ── V8 Multi-Query Expand ───────────────────────────────────────────────
    V8_ENABLE_MULTI_QUERY: bool = False
    V8_EXPAND_N: int = 3  # number of query reformulations to generate

    # ── Domain Gate ───────────────────────────────────────────────────────
    DOMAIN_GATE_THRESHOLD: float = 0.0  # cosine similarity floor; 0.0 = disabled


v7_config = V7Config()
