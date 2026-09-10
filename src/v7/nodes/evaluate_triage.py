"""V7 node: evaluate_triage — 3-way sufficiency gate (hard gate + structured gap)."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, cast

from src.v7.gap import build_gap, has_enumeration_intent, passage_source
from src.v7.hard_gates import check_full_triage
from src.v7.state_types import (
    NextAfterTriage,
    RAGState,
    RetrievalPlan,
    TriageGap,
)

logger = logging.getLogger(__name__)

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


# ─── Structured gap (issue #13) ───────────────────────────────────────────────

# Re-export for backwards compatibility with existing tests and callers.
_has_enumeration_intent = has_enumeration_intent

# Expander injected at graph build time (see init_v7_pipeline). Signature:
# fn(passages: list[dict], query: str) -> list[dict]. Not injected → the node
# behaves exactly as before, but still reports the gap.
_crossref_expander: Optional[Callable[[List[dict], str], List[dict]]] = None


def set_crossref_expander(
    fn: Optional[Callable[[List[dict], str], List[dict]]],
) -> None:
    """Inject the cross-reference expander. Call once at startup."""
    global _crossref_expander
    _crossref_expander = fn


def _merge_new_at_tail(base: List[dict], expanded: List[dict]) -> List[dict]:
    """`base` in its original order, then the passages `expanded` added, in
    theirs. The real cross-ref expander inserts bbox siblings next to their
    parent; handing that order downstream pushes an original chunk out of the
    top-12 the next node reads (issue #30). Merging here makes the triage
    output independent of how the expander arranges its result.
    """
    seen = {(passage_source(p), p.get("text", "")) for p in base}
    tail = [p for p in expanded if (passage_source(p), p.get("text", "")) not in seen]
    return list(base) + tail


def _with_gap(update: Dict[str, Any], gap: Optional[TriageGap]) -> RAGState:
    """Attach the gap to a state update when one was computed."""
    if gap is not None:
        update["triage_gap"] = gap
    return cast(RAGState, update)


def evaluate_triage(state: RAGState) -> RAGState:
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
                except Exception as exc:  # noqa: BLE001 — live query must not die
                    logger.warning("triage gap expansion failed: %s", exc)
                    expanded = None

            # Nothing missing: the counter fired, but every clause it saw is
            # already in the list — there is nothing for rag_complex to find.
            closed_in_place = not gap["open"]
            if expanded:
                gap = build_gap(passages, resolve_in=expanded)
                if not gap["open"]:
                    # Only check_full_triage is re-run. The crossref counter is
                    # NOT recomputed — the expansion adds the very chunks that
                    # raised it, so re-counting could never let the gap close.
                    merged = _merge_new_at_tail(passages, expanded)
                    recheck = check_full_triage(original_q, active_q, merged, plan)
                    if recheck["triage"] == "sufficient":
                        passages = merged
                        result = recheck
                        closed_in_place = True

            if not closed_in_place:
                # The expansion is discarded when it fails to close the gap:
                # the real expander inserts bbox siblings after their parent,
                # so handing on the inflated list pushes relevant chunks out of
                # the top-12 (measured on held-out 05.09.2026).
                return {
                    "sufficient": False,
                    "sufficiency_details": result,
                    "fallback_passages": passages,
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


def route_after_triage(state: RAGState) -> NextAfterTriage:
    if state.get("sufficient"):
        # Enumeration queries require complete coverage across multiple document sections.
        # Force rag_complex even when simple-triage scores are sufficient.
        if _has_enumeration_intent(state.get("query", "")):
            return "rag_complex"
        return "end"

    # Insufficient verdicts route to rag_complex for a broader search attempt.
    # rag_complex → evaluate_complex emits abstain via the abstain node if
    # passages stay poor.
    return "rag_complex"
