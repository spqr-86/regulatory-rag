"""V7 node: evaluate_triage — simple-ветка контракта решения.

enrich → pack → validate → decide (спек 2026-09-09). Своей логики эскалации
у узла больше нет: маршрут выбирает decide_simple, маршрутизацию делает
conditional edge по route_decision.
"""

from __future__ import annotations

from typing import cast

from src.v7.config import v7_config
from src.v7.decide import decide_simple, terminal_update
from src.v7.gap import build_gap, has_enumeration_intent  # noqa: F401 — re-export
from src.v7.gap import has_enumeration_intent as _has_enumeration_intent  # noqa: F401
from src.v7.nodes.visual_enrichment import enrich_passages
from src.v7.pack_context import pack_context
from src.v7.state_types import RAGState, RetrievalPlan
from src.v7.validate import required_obligations, validate_context


def evaluate_triage(state: RAGState) -> RAGState:
    query = state.get("query", "")
    active_q = state.get("active_query", query)
    required = required_obligations(query)
    attempts = state.get("retrieval_attempts") or []

    if not attempts:
        return cast(
            RAGState,
            terminal_update(
                route="abstain",
                reason="retrieval_error",
                final_context=[],
                candidate_context=[],
                verdict=None,
                required=required,
                technical_failure=True,
            ),
        )

    last = attempts[-1]
    plan = cast(RetrievalPlan, last.get("attempt_plan") or state.get("plan") or {})
    raw = last.get("passages", [])
    retrieval_error = bool(last.get("retrieval_error"))

    enriched = enrich_passages(raw)
    packed = pack_context(enriched, active_q, dict(plan))
    verdict = validate_context(
        packed["final_context"],
        query,
        active_q,
        plan,
        required,
        pack_status=packed["status"],
    )
    route, reason = decide_simple(
        packed["final_context"],
        verdict,
        retrieval_error=retrieval_error,
        abstain_on_empty=v7_config.ABSTAIN_ON_EMPTY_EVIDENCE,
    )

    snapshot = None
    if route == "complex":
        snapshot = {
            "passages": packed["final_context"],
            "plan": dict(plan),
            "active_query": active_q,
            "origin": "fallback_snapshot",
            "packed": True,
            "pack_status": packed["status"],
        }

    return cast(
        RAGState,
        terminal_update(
            route=route,
            reason=reason,
            final_context=packed["final_context"],
            candidate_context=enriched,
            verdict=verdict,
            required=required,
            technical_failure=retrieval_error or packed["status"] == "degraded",
            fallback_snapshot=snapshot,
        ),
    )
