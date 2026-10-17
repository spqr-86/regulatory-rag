"""V7 node: evaluate_complex — перебор кандидатов через общий конвейер.

Каждый кандидат проходит enrich → pack → validate, побеждает первый,
для которого строгий accept() истинен; иначе выбирается лучший допустимый.
"""

from __future__ import annotations

# ANCHOR: evaluate the existing candidate queue using graph-bound pack callbacks.
# The cache and candidate snapshots belong to this invocation only.
from typing import Any, Dict, List, cast

import structlog

from src.v7.config import v7_config
from src.v7.contract import OBL_REFS, Candidate, PackResult, RejectedCandidate
from src.v7.decide import accept, terminal_update
from src.v7.nlp_core import mmr_select, passage_identity
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


def _merged_passages(state: RAGState, plan: RetrievalPlan) -> List[dict]:
    """Simple top-k anchors, then MMR over the last complex attempt.

    Escalation is an additive search step: the simple top-k stays at the head
    so a direct norm cannot be displaced. The complex pool is ranked by the
    reranker (see mmr_select); simple scores are not comparable with it, so the
    simple attempt is not pooled. With nothing new from complex the candidate is
    empty: the anchors alone would only repeat the packed fallback snapshot.
    """
    snapshot = state.get("fallback_snapshot") or {}
    anchors = list(snapshot.get("passages") or [])[: v7_config.SIMPLE_TOP_K]
    seen = {passage_identity(p) for p in anchors}
    complex_attempts = [
        a for a in state.get("retrieval_attempts") or [] if a.get("stage") == "complex"
    ]
    last_complex = complex_attempts[-1].get("passages") if complex_attempts else None
    pool: List[dict] = []
    for p in last_complex or []:
        pid = passage_identity(p)
        if pid not in seen:
            seen.add(pid)
            pool.append(p)
    if not pool:
        return []
    room = max(v7_config.FINAL_MERGE_TOP_K - len(anchors), 0)
    return anchors + mmr_select(pool, room, plan.get("mmr_lambda"))


def _candidates(state: RAGState) -> List[Candidate]:
    """Return merged, last-attempt, then immutable simple snapshot candidates."""
    attempts = state.get("retrieval_attempts") or []
    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state.get("plan") or {})
    active_q = state.get("active_query", state.get("query", ""))
    snapshot = state.get("fallback_snapshot")

    merged = _merged_passages(state, plan)
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


def _prepare(
    cand: Candidate,
    cache: Dict[str, Any],
    dependencies=None,
    filters: dict | None = None,
) -> PackResult:
    """Enrich and pack a candidate, unless it is an already packed snapshot."""
    if cand.get("packed"):
        return {
            "final_context": list(cand["passages"]),
            "status": cand.get("pack_status", "ok"),
            "dropped": 0,
        }
    if dependencies is None:
        enriched = enrich_passages(cand["passages"])
        return pack_context(
            enriched,
            cand["active_query"],
            cand["plan"],
            cache=cache,
            filters=filters,
        )
    enriched = enrich_passages(
        cand["passages"], visual_proof_fn=dependencies.visual_proof
    )
    return pack_context(
        enriched,
        cand["active_query"],
        cand["plan"],
        cache=cache,
        crossref_expander=dependencies.crossref_expander,
        filters=filters,
        limits=dependencies.pack_limits,
    )


def evaluate_complex(state: RAGState, *, dependencies=None) -> RAGState:
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

    admissible = []
    for cand in candidates:
        packed = _prepare(cand, cache, dependencies, state.get("filters"))
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
            packed["final_context"], verdict, on_complex=True, is_last_candidate=True
        ):
            admissible.append((cand, packed, verdict))
        if accept(packed["final_context"], verdict):
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

    # Strict acceptance has priority across the entire queue. Only after it
    # fails choose the strongest admissible context, with stable queue ties.
    if admissible:
        cand, packed, verdict = max(
            admissible,
            key=lambda item: (
                item[2]["top_score"],
                len(item[1]["final_context"]),
            ),
        )
        reason = (
            "refs_best_effort"
            if verdict["obligations_unmet"] == [OBL_REFS]
            else "enumeration_best_effort"
        )
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
                rejected=[r for r in rejected if r["origin"] != cand["origin"]],
            ),
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
