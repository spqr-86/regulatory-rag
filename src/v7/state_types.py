"""V7 RAG pipeline — data contracts and type definitions.

All TypedDicts used across the v7 pipeline are defined here.
This module has zero dependencies on other project code.

Source spec: docs/feature/migration-v7 (lines 182-323).
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Literal, TypedDict
from dataclasses import dataclass

# ─── Data Classes ──────────────────────────────────────────────────────────


@dataclass
class Doc:
    """Represents a document with its text and metadata."""

    id: str
    text: str
    metadata: dict


@dataclass
class ScoredDoc(Doc):
    """Represents a document with an associated score."""

    score: float


# ─── Literal type aliases ────────────────────────────────────────────────────

Intent = Literal["noise", "domain"]
TriageCategory = Literal["sufficient", "borderline", "clearly_bad"]
EvidenceVerdict = Literal["answer", "improve", "abstain"]

NextAfterIntent = Literal["end", "router"]
NextAfterRouter = Literal["rag_simple", "clarify_respond"]
NextAfterTriage = Literal["end", "rag_complex"]
NextAfterEvalComplex = Literal["end", "abstain"]

# ─── Constants ───────────────────────────────────────────────────────────────

ALLOWED_FILTER_KEYS = frozenset({"doc_type", "doc_id", "section", "category", "year"})


# ═══════════════════════════════════════════════════════════════════════════════
# STATE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════


class RetrievalPlan(TypedDict, total=False):
    """Retrieval parameters. Created by router, updated on escalation.

    Retrieval: top_k, rerank, timeout_ms.
    Hard gates: threshold, min_passages, min_keyword_overlap.
    Soft signals: max_single_doc_ratio.
    Borderline zone: borderline_threshold.
    LLM verifier: min_verifier_confidence — minimum LLM confidence
                  to accept the verdict. Below this → ignore verdict, escalate.
    """

    top_k: int
    rerank: bool
    timeout_ms: int
    threshold: float
    min_passages: int
    min_keyword_overlap: float
    max_single_doc_ratio: float
    borderline_threshold: float
    min_verifier_confidence: float
    # v6.1: router-driven signals
    require_multi_doc: bool  # True for comparison queries → diversity = hard gate
    mmr_lambda: float  # dynamic: 0.9-1.0 factoid, 0.5-0.6 overview


class RetrievalAttempt(TypedDict, total=False):
    """Result of a single retrieval attempt. Append-only.

    attempt_plan: plan snapshot at retrieval time (fix #4).
    metrics: per-attempt diagnostics for offline evaluation.
    """

    retrieval_id: str
    stage: Literal["simple", "complex"]
    passages: List[dict]
    top_score: float
    attempt_plan: dict
    metrics: dict


class HardGateResult(TypedDict):
    """Result of hard gates ONLY. No triage, no soft signals.

    Used by evaluate_complex (final check).

    Dual overlap: keyword_overlap_active (on active query),
    keyword_overlap_original (on original — drift detection).
    """

    sufficient: bool
    above_threshold: bool
    enough_evidence: bool
    keyword_overlap_ok: bool
    top_score: float
    passage_count: int
    keyword_overlap_active: float
    keyword_overlap_original: float


class SufficiencyResult(TypedDict):
    """Full check result: hard gates + soft signals + triage.

    Used by evaluate_triage (3-way gate).
    """

    sufficient: bool
    above_threshold: bool
    enough_evidence: bool
    keyword_overlap_ok: bool
    diversity_ok: bool
    escalation_hint: bool
    triage: TriageCategory
    top_score: float
    keyword_overlap_active: float
    keyword_overlap_original: float
    passage_count: int
    unique_docs: int
    max_doc_ratio: float


class EvidenceReport(TypedDict, total=False):
    """V8 evidence assessment report from _evidence_assess."""

    verdict: EvidenceVerdict
    reranker_top1: float
    reranker_top3_mean: float
    coverage_estimate: float
    kw_overlap: float
    passage_count: int


class RAGState(TypedDict, total=False):
    """V7 graph state.

    INPUT:     query (immutable), filters.
    INTERNAL:  intent, plan, retrieval_id, active_query,
               retrieval_attempts, sufficient.
    OUTPUT:    final_passages, final_score, fallback_passages, fallback_score,
               clarify_message, abstain_reason, sufficiency_details.
    UX:        status_message — progress for frontend streaming.
    """

    # INPUT
    query: str
    filters: dict
    # INTERNAL
    intent: Intent
    plan: RetrievalPlan
    retrieval_id: str
    active_query: str
    retrieval_attempts: Annotated[List[RetrievalAttempt], operator.add]
    sufficient: bool
    # OUTPUT
    final_passages: List[dict]
    final_score: float
    fallback_passages: List[dict]
    fallback_score: float
    clarify_message: str
    abstain_reason: str
    sufficiency_details: SufficiencyResult
    answer: str  # synthesised LLM answer (set by generate_answer node)
    evidence_report: EvidenceReport  # V8 evidence assessment; populated only when V8_ENABLE_EVIDENCE_ASSESS=True
    # UX
    status_message: str
