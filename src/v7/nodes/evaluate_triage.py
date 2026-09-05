"""V7 node: evaluate_triage — 3-way sufficiency gate (V8: evidence-aware)."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, cast

from src.v7.config import v7_config
from src.v7.cross_ref import _extract_refs
from src.v7.hard_gates import check_full_triage
from src.v7.state_types import (
    EvidenceReport,
    GapRef,
    NextAfterTriage,
    RAGState,
    RetrievalPlan,
    TriageGap,
)

logger = logging.getLogger(__name__)

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


# ─── Structured gap (issue #13) ───────────────────────────────────────────────

# Expander injected at graph build time (see init_v7_pipeline). Signature:
# fn(passages: list[dict], query: str) -> list[dict]. Not injected → the node
# behaves exactly as before, but still reports the gap.
_crossref_expander: Optional[Callable[[List[dict], str], List[dict]]] = None


def set_crossref_expander(fn: Optional[Callable[[List[dict], str], List[dict]]]) -> None:
    """Inject the cross-reference expander. Call once at startup."""
    global _crossref_expander
    _crossref_expander = fn


def _passage_source(passage: dict) -> str:
    return passage.get("metadata", {}).get("source") or passage.get("doc_id", "")


def _ref_present(kind: str, num: str, content: str) -> bool:
    """True if the chunk text structurally *contains* the referenced unit.

    Deliberately narrower than cross_ref._ref_matches_doc: a chunk merely
    naming "пункт 12" is what creates the gap, so a phrase match must not
    count as its resolution. Only a structural heading does.
    """
    v = re.escape(num)
    if kind == "clause":
        return bool(re.search(rf"(?m)^\s*{v}\.(?=\s)", content))
    if kind == "article":
        return bool(
            re.search(rf"(?mi)^\s*(?:стать\w+\s+)?{v}[.\s]", content)
            and re.search(rf"(?i)стать\w+\s+{v}\b", content)
        )
    if kind == "subpara":
        return bool(re.search(rf"(?mi)^\s*{v}\)", content))
    return False


def build_gap(passages: List[dict], resolve_in: Optional[List[dict]] = None) -> TriageGap:
    """Describe what the retrieved text names but does not contain.

    Refs are extracted from the top-5 passages — the same slice that trips
    the crossref escalation — and deduplicated by (doc_id, kind, num).
    Resolution is checked across the whole of `resolve_in` (defaults to
    `passages`) within the same source: a clause sitting at position 9 is
    not a gap. Markers carry no doc_id, so one number named in two documents
    gives two refs and a single marker; the marker stays open while any of
    its refs is unresolved.
    """
    haystack = passages if resolve_in is None else resolve_in

    refs: List[GapRef] = []
    seen: set[tuple[str, str, str]] = set()
    for passage in passages[:5]:
        source = _passage_source(passage)
        for kind, num in _extract_refs(passage.get("text", "")):
            key = (source, kind, num)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"kind": kind, "num": num, "doc_id": source})

    by_source: Dict[str, List[str]] = {}
    for passage in haystack:
        by_source.setdefault(_passage_source(passage), []).append(
            passage.get("text", "")
        )

    resolved: Dict[str, bool] = {}
    for ref in refs:
        marker = f"{ref['kind']}:{ref['num']}"
        present = any(
            _ref_present(ref["kind"], ref["num"], text)
            for text in by_source.get(ref["doc_id"], [])
        )
        resolved[marker] = resolved.get(marker, True) and present

    closed = [m for m, ok in resolved.items() if ok]
    open_ = [m for m, ok in resolved.items() if not ok]

    return {"kind": "unresolved_ref", "refs": refs, "closed": closed, "open": open_}


def _with_gap(update: Dict[str, Any], gap: Optional[TriageGap]) -> RAGState:
    """Attach the gap to a state update when one was computed."""
    if gap is not None:
        update["triage_gap"] = gap
    return cast(RAGState, update)


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
        # the answer is distributed across multiple document sections. Before
        # paying for rag_complex, try to close the gap in place (issue #13, B2):
        # name what is missing, pull it from the same sources, re-check.
        gap: Optional[TriageGap] = None
        crossref_hits = _count_crossref_hits(passages)
        if crossref_hits >= _CROSSREF_ESCALATION_THRESHOLD:
            gap = build_gap(passages)

            expanded: Optional[list] = None
            if gap["open"] and _crossref_expander is not None:
                try:
                    expanded = list(_crossref_expander(passages, active_q))
                except Exception as exc:  # noqa: BLE001 — a live query must not die here
                    logger.warning("triage gap expansion failed: %s", exc)
                    expanded = None

            closed_in_place = False
            if expanded:
                gap = build_gap(passages, resolve_in=expanded)
                if not gap["open"]:
                    # Only check_full_triage is re-run. The crossref counter is
                    # NOT recomputed — the expansion adds the very chunks that
                    # raised it, so re-counting could never let the gap close.
                    recheck = check_full_triage(original_q, active_q, expanded, plan)
                    if recheck["triage"] == "sufficient":
                        passages = expanded
                        result = recheck
                        closed_in_place = True

            if not closed_in_place:
                fallback = expanded if expanded else passages
                return {
                    "sufficient": False,
                    "sufficiency_details": result,
                    "fallback_passages": fallback,
                    "fallback_score": result["top_score"],
                    "triage_gap": gap,
                }
            # Gap closed: fall through to the remaining escalations, which stay
            # in force and are evaluated on the expanded passages.

        # Zero-overlap escalation: none of the original query keywords appear in
        # any retrieved chunk. Topic was found (active_query overlap ok) but the
        # specific answer is missing — escalate to rag_complex for broader search.
        kw_original = result["keyword_overlap_original"]
        if kw_original == 0.0:
            return _with_gap(
                {
                    "sufficient": False,
                    "sufficiency_details": result,
                    "fallback_passages": passages,
                    "fallback_score": result["top_score"],
                },
                gap,
            )

        # Enumeration escalation: even though triage is sufficient, we force
        # rag_complex for queries that require complete enumeration coverage.
        # Save the current passages as fallback so evaluate_complex can fall
        # back to this simple-path result if the complex attempt fails its
        # (stricter) gates — preventing an unnecessary abstain.
        if _has_enumeration_intent(original_q):
            return _with_gap(
                {
                    "sufficient": True,
                    "final_passages": passages,
                    "final_score": result["top_score"],
                    "sufficiency_details": result,
                    "fallback_passages": passages,
                    "fallback_score": result["top_score"],
                },
                gap,
            )

        return _with_gap(
            {
                "sufficient": True,
                "final_passages": passages,
                "final_score": result["top_score"],
                "sufficiency_details": result,
            },
            gap,
        )

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
        original_q = state.get("query", "")
        base: Dict[str, Any] = {
            "sufficient": True,
            "final_passages": passages,
            "final_score": top_score,
            "evidence_report": report,
        }
        # Save fallback so evaluate_complex can recover this result if the
        # subsequent complex attempt (triggered by enumeration routing) fails.
        if _has_enumeration_intent(original_q):
            base["fallback_passages"] = passages
            base["fallback_score"] = top_score
        return cast(RAGState, base)

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
