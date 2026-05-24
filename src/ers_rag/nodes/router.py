"""V7 node: router — query classification, plan creation."""

from __future__ import annotations

from src.glossary import expand_query_with_glossary
from src.ers_rag.config import v7_config
from src.ers_rag.hard_gates import validate_filters
from src.ers_rag.nodes.utils import make_retrieval_id
from src.ers_rag.state_types import NextAfterRouter, RAGState, RetrievalPlan

# ─── Query classification (keyword-based stub, production: LLM) ──────────

COMPARISON_MARKERS = frozenset(
    {
        "сравни",
        "разница",
        "отличие",
        "коллизия",
        "приоритет",
        "что важнее",
        "противореч",
        "различия",
        "vs",
        "или лучше",
    }
)

FACTOID_MARKERS = frozenset(
    {
        "пункт",
        "п.",
        "таблица",
        "табл.",
        "значение",
        "величина",
        "минимальн",
        "максимальн",
        "не менее",
        "не более",
        "сколько",
    }
)


def _classify_query(q: str) -> tuple[bool, float]:
    """(require_multi_doc, mmr_lambda) based on query type.

    mmr_lambda overrides config default:
      0.5  — comparison (high diversity)
      0.95 — factoid (high precision)
      config default — general
    """
    q_lower = q.lower()
    if any(m in q_lower for m in COMPARISON_MARKERS):
        return True, 0.5
    if any(m in q_lower for m in FACTOID_MARKERS):
        return False, 0.95
    return False, v7_config.MMR_LAMBDA


# ─── Node ─────────────────────────────────────────────────────────────────


def router(state: RAGState) -> RAGState:
    """Reads: query, filters. Writes: plan, retrieval_id, active_query, verify_iteration."""
    q = (state.get("query") or "").strip()
    filters = state.get("filters")

    if len(q) < 8:
        return {
            "clarify_message": (
                "Уточните, пожалуйста, ваш запрос. "
                "Например: какой нормативный документ или тему вы имеете в виду?"
            ),
        }

    require_multi, mmr_lambda = _classify_query(q)

    plan: RetrievalPlan = {
        "top_k": v7_config.SIMPLE_TOP_K,
        "rerank": False,
        "timeout_ms": v7_config.SIMPLE_TIMEOUT_MS,
        "threshold": v7_config.HARD_GATE_THRESHOLD,
        "min_passages": v7_config.MIN_PASSAGES,
        "min_keyword_overlap": v7_config.MIN_KEYWORD_OVERLAP_ACTIVE,
        "max_single_doc_ratio": v7_config.MAX_SINGLE_DOC_RATIO,
        "borderline_threshold": v7_config.TRIAGE_SOFT_THRESHOLD,
        "min_verifier_confidence": v7_config.VERIFIER_CONFIDENCE_ANCHOR,
        "require_multi_doc": require_multi,
        "mmr_lambda": mmr_lambda,
    }

    return {
        "plan": plan,
        "retrieval_id": make_retrieval_id(q, validate_filters(filters)),
        "active_query": expand_query_with_glossary(q),
        "verify_iteration": 0,
        # State cleanup
        "clarify_message": None,
        "abstain_reason": None,
        "fallback_passages": None,
        "fallback_score": None,
    }


def route_after_router(state: RAGState) -> NextAfterRouter:
    return "clarify_respond" if state.get("clarify_message") else "rag_simple"


def clarify_respond(state: RAGState) -> RAGState:
    """Pass-through. clarify_message already in state."""
    return {}
