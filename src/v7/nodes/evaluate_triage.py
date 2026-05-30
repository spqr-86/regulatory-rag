"""V7 node: evaluate_triage — 3-way sufficiency gate (V8: evidence-aware)."""

from __future__ import annotations

import re
from typing import Any, Dict, cast

from src.v7.config import v7_config
from src.v7.hard_gates import check_full_triage
from src.v7.state_types import (
    EvidenceReport,
    NextAfterTriage,
    RAGState,
    RetrievalPlan,
)

# Enumeration question patterns — require complete coverage of all categories/conditions.
# For such queries rag_simple may return an incomplete answer even at high top_score.
_ENUMERATION_PATTERNS = [
    r"\bкто\s+проходит\b",
    r"\bкто\s+обязан\b",
    r"\bкакие\s+категори[яи]\b",
    r"\bв\s+каких\s+случаях\b",
    r"\bкогда\s+не\s+требуется\b",
    r"\bкому\s+не\s+требуется\b",
    r"\bкто\s+освобождается\b",
    r"\bперечислите\b",
    r"\bкакие\s+работники\b",
    r"\bкаким\s+работникам\b",
]


# Cross-reference patterns in regulatory documents.
# If retrieved chunks contain many such markers, the answer is likely spread
# across adjacent clauses and requires broader search.
_CROSSREF_PATTERNS = [
    r"\bпункт[а-я]*\s+\d+",
    r"\bподпункт[а-я]*\s+\d+",
    r"\bза\s+исключением\b",
    r"\bв\s+соответствии\s+с\b",
    r"\bуказанн[а-я]+\s+в\b",
    r"\bсогласно\s+пункт",
    r"\bсм\.\s+пункт",
    r"\bприложени[яе]\s+\d+",
]

_CROSSREF_ESCALATION_THRESHOLD = 3  # >= N hits в топ-чанках → escalate


def _count_crossref_hits(passages: list[dict]) -> int:
    """Count total number of crossref pattern matches in the top-5 passages."""
    total = 0
    for p in passages[:5]:
        text = p.get("text", "").lower()
        for pattern in _CROSSREF_PATTERNS:
            if re.search(pattern, text):
                total += 1
    return total


def _has_enumeration_intent(query: str) -> bool:
    """True if the query requires complete enumeration of categories/conditions.

    Such queries are routed to rag_complex even when simple-triage is sufficient,
    because the answer is often spread across multiple document clauses.
    """
    q = query.lower()
    return any(re.search(p, q) for p in _ENUMERATION_PATTERNS)


def _legacy_triage(state: RAGState) -> RAGState:
    """3-way gate: sufficient / borderline / clearly_bad.

    Uses check_full_triage() with plan from attempt_plan snapshot.
    Saves fallback passages when hard gates pass but soft signals escalate.
    """
    attempts = state.get("retrieval_attempts") or []
    if not attempts:
        return {"sufficient": False}

    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state["plan"])
    original_q = state.get("query", "")
    active_q = state.get("active_query", original_q)

    result = check_full_triage(original_q, active_q, last.get("passages", []), plan)

    if result["triage"] == "sufficient":
        passages = last.get("passages", [])

        # Crossref escalation: many cross-references in retrieved chunks indicate
        # the answer is distributed across multiple document sections.
        # Save as fallback and escalate to rag_complex for fuller coverage.
        crossref_hits = _count_crossref_hits(passages)
        if crossref_hits >= _CROSSREF_ESCALATION_THRESHOLD:
            return {
                "sufficient": False,
                "sufficiency_details": result,
                "fallback_passages": passages,
                "fallback_score": result["top_score"],
            }

        # Zero-overlap escalation: none of the original query keywords appear in
        # any retrieved chunk. Topic was found (active_query overlap ok) but the
        # specific answer is missing — escalate to rag_complex for broader search.
        kw_original = result["keyword_overlap_original"]
        if kw_original == 0.0:
            return {
                "sufficient": False,
                "sufficiency_details": result,
                "fallback_passages": passages,
                "fallback_score": result["top_score"],
            }

        return {
            "sufficient": True,
            "final_passages": passages,
            "final_score": result["top_score"],
            "sufficiency_details": result,
        }

    update: Dict[str, Any] = {
        "sufficient": False,
        "sufficiency_details": result,
    }

    # Fallback: hard gates ok, but triage != sufficient (soft signal escalation)
    if result["sufficient"]:
        update["fallback_passages"] = last.get("passages", [])
        update["fallback_score"] = result["top_score"]

    return cast(RAGState, update)


def _evidence_assess(state: RAGState) -> RAGState:
    """V8 evidence-aware triage using FlashRank reranker scores + coverage estimation.

    Verdict logic:
    - "answer":  reranker_top1 >= ANSWER_RERANKER_TOP1 AND coverage >= ANSWER_COVERAGE
    - "abstain": reranker_top1 < ABSTAIN_RERANKER_TOP1 AND coverage < ABSTAIN_COVERAGE
    - "improve": everything else
    """
    attempts = state.get("retrieval_attempts") or []
    if not attempts:
        report = EvidenceReport(
            verdict="abstain",
            reranker_top1=0.0,
            reranker_top3_mean=0.0,
            coverage_estimate=0.0,
            kw_overlap=0.0,
            passage_count=0,
        )
        return {"sufficient": False, "evidence_report": report}

    last = attempts[-1]
    metrics = last.get("metrics") or {}
    passages = last.get("passages") or []
    top_score = last.get("top_score", 0.0)

    reranker_top1: float = float(metrics.get("reranker_top1", 0.0))
    reranker_top3_mean: float = float(metrics.get("reranker_top3_mean", 0.0))
    kw_overlap: float = float(metrics.get("keyword_overlap_active", 0.0))
    passage_count: int = len(passages)

    coverage_estimate: float = kw_overlap * min(passage_count / 10.0, 1.0)

    if (
        reranker_top1 >= v7_config.V8_EVIDENCE_ANSWER_RERANKER_TOP1
        and coverage_estimate >= v7_config.V8_EVIDENCE_ANSWER_COVERAGE
    ):
        verdict = "answer"
    elif (
        reranker_top1 < v7_config.V8_EVIDENCE_ABSTAIN_RERANKER_TOP1
        and coverage_estimate < v7_config.V8_EVIDENCE_ABSTAIN_COVERAGE
    ):
        verdict = "abstain"
    else:
        verdict = "improve"

    report = EvidenceReport(
        verdict=verdict,
        reranker_top1=reranker_top1,
        reranker_top3_mean=reranker_top3_mean,
        coverage_estimate=coverage_estimate,
        kw_overlap=kw_overlap,
        passage_count=passage_count,
    )

    if verdict == "answer":
        return {
            "sufficient": True,
            "final_passages": passages,
            "final_score": top_score,
            "evidence_report": report,
        }

    if verdict == "improve":
        # Save passages as fallback so rag_complex has a starting point if needed.
        return cast(
            RAGState,
            {
                "sufficient": False,
                "evidence_report": report,
                "fallback_passages": passages,
                "fallback_score": top_score,
            },
        )

    # verdict == "abstain": route_after_triage → rag_complex → evaluate_complex → abstain node.
    # rag_complex with poor passages causes evaluate_complex to emit abstain verdict.
    return cast(RAGState, {"sufficient": False, "evidence_report": report})


def evaluate_triage(state: RAGState) -> RAGState:
    """Dispatch to evidence-aware (V8) or legacy triage based on feature flag."""
    if v7_config.V8_ENABLE_EVIDENCE_ASSESS:
        return _evidence_assess(state)
    return _legacy_triage(state)


def route_after_triage(state: RAGState) -> NextAfterTriage:
    if state.get("sufficient"):
        # Enumeration queries require complete coverage across multiple document sections.
        # Force rag_complex even when simple-triage scores are sufficient.
        if _has_enumeration_intent(state.get("query", "")):
            return "rag_complex"
        return "end"

    # Both V8 (evidence_report) and legacy (sufficiency_details) insufficient
    # verdicts route to rag_complex for a broader search attempt. rag_complex →
    # evaluate_complex emits abstain via the abstain node if passages stay poor.
    return "rag_complex"
