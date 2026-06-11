"""V7 node: rag_complex — slow path with rerank + MMR."""

from __future__ import annotations

from typing import Callable, List, Optional

from src.v7.config import v7_config
from src.v7.hard_gates import compute_attempt_metrics, validate_filters
from src.v7.nodes.utils import make_retrieval_id
from src.v7.state_types import RAGState, RetrievalAttempt, RetrievalPlan

# ─── Retriever interface (injected, same as rag_simple) ──────────────────


def _default_vector_search(**kwargs) -> List[dict]:
    return []


_vector_search: Callable[..., List[dict]] = _default_vector_search
_rerank_fn: Optional[Callable[[str, List[dict], int], List[dict]]] = None


def set_vector_search(fn: Callable[..., List[dict]]) -> None:
    """Inject vector search implementation for complex retrieval."""
    global _vector_search
    _vector_search = fn


def set_rerank_fn(fn: Callable[[str, List[dict], int], List[dict]]) -> None:
    """Inject FlashRank reranker. Signature: fn(query, passages, top_k) -> passages."""
    global _rerank_fn
    _rerank_fn = fn


_section_fetch_fn: Optional[Callable[[List[dict]], List[dict]]] = None


def set_section_fetch_fn(fn: Callable[[List[dict]], List[dict]]) -> None:
    """Inject section-aware expander. Signature: fn(passages) -> additional_passages."""
    global _section_fetch_fn
    _section_fetch_fn = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def rag_complex(state: RAGState) -> RAGState:
    """Slow path: higher thresholds, rerank + MMR."""
    current_plan = state.get("plan") or {}

    slow_plan: RetrievalPlan = {
        "top_k": v7_config.COMPLEX_TOP_K,
        "rerank": True,
        "timeout_ms": v7_config.COMPLEX_TIMEOUT_MS,
        "threshold": max(v7_config.COMPLEX_THRESHOLD, current_plan.get("threshold", 0)),
        "min_passages": v7_config.COMPLEX_MIN_PASSAGES,
        "min_keyword_overlap": v7_config.COMPLEX_MIN_KW_OVERLAP,
        "max_single_doc_ratio": v7_config.COMPLEX_MAX_SINGLE_DOC_RATIO,
        "borderline_threshold": v7_config.COMPLEX_BORDERLINE_THRESHOLD,
        "min_verifier_confidence": v7_config.VERIFIER_CONFIDENCE_ANCHOR,
        "require_multi_doc": current_plan.get("require_multi_doc", False),
        "mmr_lambda": current_plan.get("mmr_lambda", v7_config.MMR_LAMBDA),
    }

    active_q = state.get("active_query", state.get("query", ""))
    original_q = state.get("query", "")
    safe_filters = validate_filters(state.get("filters"))
    # Prefer the retrieval_id set by the router (computed from original_q).
    # rag_complex uses active_query (glossary-expanded) which can differ from
    # original_q, causing dedup keys to diverge between router and this node.
    rid = state.get("retrieval_id") or make_retrieval_id(active_q, safe_filters)

    # Dedup
    existing = state.get("retrieval_attempts") or []
    if any(
        a.get("retrieval_id") == rid and a.get("stage") == "complex" for a in existing
    ):
        return {}

    passages = _vector_search(
        query=active_q,
        filters=safe_filters,
        top_k=slow_plan["top_k"],
    )

    # Section-aware expansion: fetch all chunks from the same section as the top anchor.
    # Helps for queries where the answer is scattered across multiple paragraphs of one section.
    if _section_fetch_fn is not None and passages:
        extra = _section_fetch_fn(passages)
        if extra:
            seen_texts = {p.get("text", "") for p in passages}
            for p in extra:
                if p.get("text", "") not in seen_texts:
                    passages.append(p)
                    seen_texts.add(p.get("text", ""))

    # Cap candidates before CrossEncoder to prevent O(n) latency blowup.
    # Section-fetch can pull entire large sections (e.g. ТК РФ = 477 chunks).
    # Pre-sort by vector_score so the best candidates survive the cut.
    cap = v7_config.RERANK_CANDIDATE_CAP
    if len(passages) > cap:
        passages = sorted(
            passages,
            key=lambda p: p.get("vector_score", p.get("score", 0.0)),
            reverse=True,
        )[:cap]

    # FlashRank reranking (if injected): reorders by cross-encoder score,
    # but top_score stays anchored to vector similarity (not inflated FlashRank probs).
    if _rerank_fn is not None and passages:
        passages = _rerank_fn(active_q, passages, slow_plan["top_k"])
    top_score = max(
        (p.get("vector_score", p.get("score", 0.0)) for p in passages), default=0.0
    )

    _, metrics = compute_attempt_metrics(original_q, active_q, passages, slow_plan)

    return {
        "plan": slow_plan,
        "retrieval_attempts": [
            RetrievalAttempt(
                retrieval_id=rid,
                stage="complex",
                passages=passages,
                top_score=top_score,
                attempt_plan=dict(slow_plan),
                metrics=metrics,
            )
        ],
        "status_message": f"Расширенный поиск: {len(passages)} фрагментов.",
    }
