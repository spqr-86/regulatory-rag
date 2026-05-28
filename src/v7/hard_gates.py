"""V7 RAG pipeline — hard gates, triage, anti-injection.

Filtering and scoring logic for retrieval quality assessment.
All thresholds come from plan dict (set by router from v7_config).

Source spec: docs/feature/migration-v7 (lines 330-594, 856-872).
"""

from __future__ import annotations

import re
from typing import List, Optional

from src.v7.nlp_core import compute_doc_diversity, compute_keyword_overlap
from src.v7.state_types import (
    ALLOWED_FILTER_KEYS,
    HardGateResult,
    SufficiencyResult,
    TriageCategory,
)

# ─── validate_filters ─────────────────────────────────────────────────────


def validate_filters(filters: Optional[dict]) -> Optional[dict]:
    """Whitelist validation of filters before passing to retriever.

    Protection against NoSQL injection via arbitrary where-clauses.
    """
    if not filters:
        return filters
    return {k: v for k, v in filters.items() if k in ALLOWED_FILTER_KEYS}


# ─── sanitize_for_llm ─────────────────────────────────────────────────────


def sanitize_for_llm(text: str) -> str:
    """Sanitise text before sending to LLM.

    Removes potential prompt injection patterns.
    """
    injection_patterns = [
        r"(?i)ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
        r"(?i)system\s*:\s*",
        r"(?i)you\s+are\s+now\s+",
        r"(?i)new\s+instructions?\s*:",
        r"(?i)forget\s+(everything|all)",
    ]
    sanitized = text
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized)
    return sanitized


# ─── check_hard_gates ─────────────────────────────────────────────────────


def check_hard_gates(
    original_query: str,
    active_query: str,
    passages: List[dict],
    plan: dict,
) -> HardGateResult:
    """Hard gates ONLY + key metrics. No triage, no soft signals.

    Hard gates (all must be True for sufficient):
      1. top_score >= threshold
      2. passage_count >= min_passages
      3. keyword_overlap (on active_query) >= min_keyword_overlap

    Dual overlap:
      keyword_overlap_active  — on active query (for hard gate).
      keyword_overlap_original — on original query (for drift detection).
    """
    if not passages:
        return HardGateResult(
            sufficient=False,
            above_threshold=False,
            enough_evidence=False,
            keyword_overlap_ok=False,
            top_score=0.0,
            passage_count=0,
            keyword_overlap_active=0.0,
            keyword_overlap_original=0.0,
        )

    top_score = max((p.get("score", 0.0) for p in passages), default=0.0)
    overlap_active = compute_keyword_overlap(active_query, passages)
    overlap_original = compute_keyword_overlap(original_query, passages)

    above_threshold = top_score >= plan.get("threshold", 0.0)
    enough_evidence = len(passages) >= plan.get("min_passages", 1)
    kw_ok = overlap_active >= plan.get("min_keyword_overlap", 0.0)

    return HardGateResult(
        sufficient=all([above_threshold, enough_evidence, kw_ok]),
        above_threshold=above_threshold,
        enough_evidence=enough_evidence,
        keyword_overlap_ok=kw_ok,
        top_score=top_score,
        passage_count=len(passages),
        keyword_overlap_active=round(overlap_active, 3),
        keyword_overlap_original=round(overlap_original, 3),
    )


# ─── check_full_triage ───────────────────────────────────────────────────


def check_full_triage(
    original_query: str,
    active_query: str,
    passages: List[dict],
    plan: dict,
) -> SufficiencyResult:
    """Hard gates + soft signals + 3-way triage label.

    Triage:
      sufficient   — hard gates ok, no escalation_hint.
      borderline   — score in zone (borderline, threshold) OR
                     hard gates ok but escalation_hint (diversity).
      clearly_bad  — score < borderline OR too few passages.
    """
    hard = check_hard_gates(original_query, active_query, passages, plan)

    unique_docs, max_doc_ratio = compute_doc_diversity(passages)
    diversity_ok = max_doc_ratio <= plan.get("max_single_doc_ratio", 1.0)

    # v6.1: if router set require_multi_doc, diversity = hard gate
    require_multi = plan.get("require_multi_doc", False)
    # escalation_hint only when multi-doc is required —
    # for factoid queries (require_multi=False) a single-document answer is normal.
    escalation_hint = not diversity_ok and require_multi

    if require_multi and not diversity_ok:
        hard_sufficient = False
    else:
        hard_sufficient = hard["sufficient"]

    borderline_threshold = plan.get("borderline_threshold", 0.0)

    if hard_sufficient and not escalation_hint:
        triage: TriageCategory = "sufficient"
    elif hard["top_score"] < borderline_threshold or not hard["enough_evidence"]:
        triage = "clearly_bad"
    else:
        triage = "borderline"

    return SufficiencyResult(
        sufficient=hard_sufficient,
        above_threshold=hard["above_threshold"],
        enough_evidence=hard["enough_evidence"],
        keyword_overlap_ok=hard["keyword_overlap_ok"],
        diversity_ok=diversity_ok,
        escalation_hint=escalation_hint,
        triage=triage,
        top_score=hard["top_score"],
        keyword_overlap_active=hard["keyword_overlap_active"],
        keyword_overlap_original=hard["keyword_overlap_original"],
        passage_count=hard["passage_count"],
        unique_docs=unique_docs,
        max_doc_ratio=round(max_doc_ratio, 3),
    )


# ─── _make_sufficiency (helper) ──────────────────────────────────────────


def make_sufficiency(
    hard: HardGateResult,
    passages: List[dict],
    triage: TriageCategory = "sufficient",
    diversity_ok: bool = True,
    escalation_hint: bool = False,
) -> SufficiencyResult:
    """Helper: build SufficiencyResult from HardGateResult + passages.

    Eliminates 13-field copy-paste in evaluate_complex.
    """
    unique_docs, max_doc_ratio = compute_doc_diversity(passages)
    return SufficiencyResult(
        sufficient=hard["sufficient"],
        above_threshold=hard["above_threshold"],
        enough_evidence=hard["enough_evidence"],
        keyword_overlap_ok=hard["keyword_overlap_ok"],
        diversity_ok=diversity_ok,
        escalation_hint=escalation_hint,
        triage=triage,
        top_score=hard["top_score"],
        keyword_overlap_active=hard["keyword_overlap_active"],
        keyword_overlap_original=hard["keyword_overlap_original"],
        passage_count=hard["passage_count"],
        unique_docs=unique_docs,
        max_doc_ratio=round(max_doc_ratio, 3),
    )


# ─── _compute_attempt_metrics ────────────────────────────────────────────


def compute_attempt_metrics(
    original_query: str,
    active_query: str,
    passages: List[dict],
    plan: dict,
) -> tuple[HardGateResult, dict]:
    """Compute hard gates + metrics for a RetrievalAttempt.

    Returns (hard_gate_result, metrics_dict).
    Reused in rag_simple/rag_complex and evaluate nodes.
    """
    hard = check_hard_gates(original_query, active_query, passages, plan)
    unique_docs, max_doc_ratio = compute_doc_diversity(passages)
    metrics = {
        "keyword_overlap_active": hard["keyword_overlap_active"],
        "keyword_overlap_original": hard["keyword_overlap_original"],
        "unique_docs": unique_docs,
        "max_doc_ratio": round(max_doc_ratio, 3),
        "passage_count": len(passages),
    }
    return hard, metrics
