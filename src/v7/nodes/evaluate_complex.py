"""V7 node: evaluate_complex — перебор кандидатов через общий конвейер.

Каждый кандидат проходит enrich → pack → validate, побеждает первый,
для которого accept() истинно.
"""

from __future__ import annotations

from typing import Any, Dict, List, cast

import structlog

from src.v7.config import v7_config
from src.v7.contract import OBL_REFS, Candidate, PackResult, RejectedCandidate
from src.v7.decide import accept, terminal_update
from src.v7.nlp_core import merge_all_passages, passage_identity
from src.v7.nodes.visual_enrichment import enrich_passages
from src.v7.pack_context import pack_context
from src.v7.state_types import RAGState, RetrievalPlan
from src.v7.validate import required_obligations, validate_context

logger = structlog.get_logger()

_ORIGIN_REASON = {
    "merged": "complex_sufficient",
    "last_attempt": "complex_sufficient",
    "fallback_snapshot": "complex_fallback_accepted",
}


def _candidates(state: RAGState) -> List[Candidate]:
    """Return merged, last-attempt, then immutable simple snapshot candidates."""
    attempts = state.get("retrieval_attempts") or []
    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state.get("plan") or {})
    active_q = state.get("active_query", state.get("query", ""))

    merged = merge_all_passages(
        attempts,
        top_k=v7_config.FINAL_MERGE_TOP_K,
        mmr_lambda=plan.get("mmr_lambda"),
    )
    snapshot = state.get("fallback_snapshot")
    if snapshot:
        # Escalation is an additive search step. Keep the original simple top-k
        # at the head so a high-scoring direct norm cannot be displaced by a
        # larger complex pool with scores from a different ranking stage.
        anchors = list(snapshot.get("passages") or [])[: v7_config.SIMPLE_TOP_K]
        anchor_ids = {passage_identity(p) for p in anchors}
        novel = [p for p in merged if passage_identity(p) not in anchor_ids]
        if novel:
            merged = (anchors + novel)[: v7_config.FINAL_MERGE_TOP_K]
    out: List[Candidate] = [
        {
            "passages": merged,
            "plan": dict(plan),
            "active_query": active_q,
            "origin": "merged",
            "packed": False,
        },
        {
            "passages": last.get("passages", []),
            "plan": dict(plan),
            "active_query": active_q,
            "origin": "last_attempt",
            "packed": False,
        },
    ]
    if snapshot:
        out.append(cast(Candidate, dict(snapshot)))
    return out


def _prepare(cand: Candidate, cache: Dict[str, Any]) -> PackResult:
    """Enrich and pack a candidate, unless it is an already packed snapshot."""
    if cand.get("packed"):
        return {
            "final_context": list(cand["passages"]),
            "status": cand.get("pack_status", "ok"),
            "dropped": 0,
        }
    enriched = enrich_passages(cand["passages"])
    return pack_context(enriched, cand["active_query"], cand["plan"], cache=cache)


def evaluate_complex(state: RAGState) -> RAGState:
    query = state.get("query", "")
    active_q = state.get("active_query", query)
    required = required_obligations(query)
    attempts = state.get("retrieval_attempts") or []
    plan = cast(RetrievalPlan, state.get("plan") or {})

    if not attempts:
        return cast(
            RAGState,
            terminal_update(
                route="abstain",
                reason="complex_exhausted",
                final_context=[],
                candidate_context=[],
                verdict=None,
                required=required,
                technical_failure=True,
            ),
        )

    snapshot = state.get("fallback_snapshot") or {}
    prior = snapshot.get("passages") if snapshot else None
    candidates = _candidates(state)
    cache: Dict[str, Any] = {}
    rejected: List[RejectedCandidate] = []
    technical = any(a.get("retrieval_error") for a in attempts)

    for i, cand in enumerate(candidates):
        is_last = i == len(candidates) - 1
        packed = _prepare(cand, cache)
        if packed["status"] == "degraded":
            technical = True
        verdict = validate_context(
            packed["final_context"],
            query,
            cand["active_query"],
            cand["plan"],
            required,
            pack_status=packed["status"],
            prior_context=prior,
        )
        logger.info(
            "evaluate_complex.candidate",
            origin=cand["origin"],
            packed=len(packed["final_context"]),
            unmet=verdict["obligations_unmet"],
            pack_status=packed["status"],
        )

        if accept(
            packed["final_context"],
            verdict,
            on_complex=True,
            is_last_candidate=is_last,
        ):
            reason = _ORIGIN_REASON.get(cand["origin"], "complex_sufficient")
            if verdict["obligations_unmet"] == [OBL_REFS]:
                reason = "refs_best_effort"
            elif verdict["obligations_unmet"]:
                reason = "enumeration_best_effort"
            return cast(
                RAGState,
                terminal_update(
                    route="generate",
                    reason=reason,
                    final_context=packed["final_context"],
                    candidate_context=cand["passages"],
                    verdict=verdict,
                    required=required,
                    technical_failure=technical,
                    rejected=cast(List[dict], rejected),
                ),
            )

        rejected.append(
            {
                "origin": cand["origin"],
                "obligations_unmet": list(verdict["obligations_unmet"]),
                "triage": verdict["triage"],
                "passages": len(packed["final_context"]),
                "pack_status": packed["status"],
            }
        )

    empty_verdict = validate_context([], query, active_q, plan, required)
    return cast(
        RAGState,
        terminal_update(
            route="abstain",
            reason="complex_exhausted",
            final_context=[],
            candidate_context=[],
            verdict=empty_verdict,
            required=required,
            technical_failure=technical,
            rejected=cast(List[dict], rejected),
        ),
    )


def route_after_decision(state: RAGState) -> str:
    """Read the terminal route selected by either evaluator."""
    return state.get("route_decision", "abstain")
