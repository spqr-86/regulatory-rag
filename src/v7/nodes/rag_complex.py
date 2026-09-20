"""V7 node: rag_complex — slow path with rerank + MMR."""

from __future__ import annotations

# ANCHOR: existing complex retrieval with per-graph dependencies.
# Input: request state/dependencies. Output: complex attempt under its own plan.

import logging
import inspect
from typing import Callable, List, Optional

from src.v7.config import v7_config
from src.v7.hard_gates import compute_attempt_metrics, validate_filters
from src.v7.nlp_core import bm25_search, rrf_merge
from src.v7.nodes.utils import make_retrieval_id
from src.v7.state_types import RAGState, RetrievalAttempt, RetrievalPlan
from src.v7.reranker import RerankerError
from src.v7.scope_filter import matches_filter

logger = logging.getLogger(__name__)

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


def rag_complex(state: RAGState, *, dependencies=None) -> RAGState:
    """Slow path: higher thresholds, rerank + MMR."""
    vector_search = (
        (dependencies.complex_vector_search or dependencies.vector_search)
        if dependencies is not None
        else _vector_search
    )
    lexical_search = (
        dependencies.bm25_search if dependencies is not None else bm25_search
    )
    section_fetch = (
        dependencies.section_fetch if dependencies is not None else _section_fetch_fn
    )
    rerank = dependencies.rerank if dependencies is not None else _rerank_fn
    current_plan = state.get("plan") or {}

    slow_plan: RetrievalPlan = {
        "top_k": v7_config.COMPLEX_TOP_K,
        "rerank": True,
        "timeout_ms": v7_config.COMPLEX_TIMEOUT_MS,
        # The slow path has its own acceptance floor. Inheriting the simple
        # threshold couples two calibration controls and can make escalation
        # stricter precisely when the simple gate is tightened.
        "threshold": v7_config.COMPLEX_THRESHOLD,
        "min_passages": v7_config.COMPLEX_MIN_PASSAGES,
        "min_keyword_overlap": v7_config.COMPLEX_MIN_KW_OVERLAP,
        "min_keyword_overlap_original": v7_config.MIN_KEYWORD_OVERLAP_ORIGINAL,
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

    # Hybrid candidate pool: dense + BM25 → RRF, same recipe as rag_simple.
    # A dense-only pool was the reason the slow path scored WORSE than the fast
    # one on Hit Rate@12 (baseline 04.09.2026): the reranker sorts well, but in
    # 22 of 26 misses the right chunk never entered the pool, while BM25 found
    # it in 19 of them. The cross-encoder re-scores everything anyway, so what
    # matters here is membership, not the order this merge produces.
    retrieval_error = False
    try:
        vector_results = vector_search(
            query=active_q,
            filters=safe_filters,
            top_k=slow_plan["top_k"],
        )
        if safe_filters:
            vector_results = [
                p
                for p in vector_results
                if matches_filter(p.get("metadata") or {}, safe_filters)
            ]
    except Exception as exc:  # noqa: BLE001 — failure is represented in graph state
        logger.warning("rag_complex: retrieval failed: %s", exc)
        retrieval_error = True
        vector_results = []

    try:
        bm25_results = (
            lexical_search(
                query=active_q,
                filters=safe_filters,
                top_k=slow_plan["top_k"],
            )
            if not retrieval_error
            else []
        )
        if safe_filters:
            bm25_results = [
                p
                for p in bm25_results
                if matches_filter(p.get("metadata") or {}, safe_filters)
            ]
    except Exception as exc:  # a missing BM25 index must not kill the slow path
        logger.warning("bm25_search failed, complex pool stays dense-only: %s", exc)
        bm25_results = []

    if bm25_results:
        passages = rrf_merge(
            vector_results,
            bm25_results,
            top_k=len(vector_results) + len(bm25_results),
        )
    else:
        passages = list(vector_results)

    # Section-aware expansion: fetch all chunks from the same section as the top anchor.
    # Helps for queries where the answer is scattered across multiple paragraphs of one section.
    if section_fetch is not None and passages:
        parameters = inspect.signature(section_fetch).parameters.values()
        accepts_filters = "filters" in inspect.signature(
            section_fetch
        ).parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters)
        extra = (
            section_fetch(passages, filters=safe_filters)
            if accepts_filters
            else section_fetch(passages)
        )
        if safe_filters:
            extra = [
                p
                for p in extra
                if matches_filter(p.get("metadata") or {}, safe_filters)
            ]
        if extra:
            seen_texts = {p.get("text", "") for p in passages}
            for p in extra:
                if p.get("text", "") not in seen_texts:
                    passages.append(p)
                    seen_texts.add(p.get("text", ""))

    # Cap candidates before CrossEncoder to prevent O(n) latency blowup.
    # Section-fetch can pull entire large sections (e.g. ТК РФ = 477 chunks).
    # Pre-sort by vector_score so the best candidates survive the cut.
    # The cut keeps the fused order: retrieved candidates first, section-fetched
    # filler after. Sorting by vector_score here would drop every BM25-only
    # candidate first (BM25 passages carry no vector_score) and undo the merge.
    cap = v7_config.RERANK_CANDIDATE_CAP
    if len(passages) > cap:
        passages = passages[:cap]

    # FlashRank reranking (if injected): reorders by cross-encoder score,
    # but top_score stays anchored to vector similarity (not inflated FlashRank probs).
    rerank_error = None
    if rerank is not None and passages:
        try:
            passages = rerank(active_q, passages, slow_plan["top_k"])
        except (RerankerError, TimeoutError) as exc:
            logger.warning("rag_complex: reranker failed: %s", exc)
            rerank_error = type(exc).__name__
            retrieval_error = True
            passages = []
    # Anchored to the dense list only: the BM25 `score` is a squashed BM25 value,
    # not a similarity, and must not lift the threshold gate.
    top_score = max((p.get("vector_score", 0.0) for p in vector_results), default=0.0)

    _, metrics = compute_attempt_metrics(original_q, active_q, passages, slow_plan)
    if rerank_error:
        metrics["rerank_error"] = rerank_error

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
                retrieval_error=retrieval_error,
            )
        ],
        "status_message": f"Расширенный поиск: {len(passages)} фрагментов.",
    }
