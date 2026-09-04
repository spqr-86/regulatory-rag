"""V7 node: rag_simple — fast hybrid retrieval (vector + BM25 → RRF)."""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

from src.v7.config import v7_config
from src.v7.hard_gates import compute_attempt_metrics, validate_filters
from src.v7.nlp_core import bm25_search, passage_identity, rrf_merge
from src.v7.state_types import RAGState, RetrievalAttempt

logger = logging.getLogger(__name__)

# ─── Retriever interface (injected at graph build time) ──────────────────


# Default: stub that returns empty. Production: Chroma/Qdrant vector search.
def _default_vector_search(**kwargs) -> List[dict]:
    return []


_vector_search: Callable[..., List[dict]] = _default_vector_search
_reranker_fn: Optional[Callable] = None
_expand_fn: Optional[Callable] = None


def set_vector_search(fn: Callable[..., List[dict]]) -> None:
    """Inject vector search implementation. Call once at startup."""
    global _vector_search
    _vector_search = fn


def set_reranker(fn: Callable) -> None:
    """Inject reranker for V8 evidence assess light rerank. Signature: fn(query, passages, top_k) -> passages."""
    global _reranker_fn
    _reranker_fn = fn


def set_expand_fn(fn: Optional[Callable]) -> None:
    """Inject query expander for V8 multi-query. Signature: fn(query: str, n: int) -> list[str]."""
    global _expand_fn
    _expand_fn = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def rag_simple(state: RAGState) -> RAGState:
    """Fast hybrid retrieval: vector + BM25 → RRF merge."""
    plan = state["plan"]
    rid = state["retrieval_id"]
    active_q = state.get("active_query", state["query"])
    original_q = state.get("query", "")
    safe_filters = validate_filters(state.get("filters"))

    # Dedup: skip if already executed for this retrieval_id + stage
    existing = state.get("retrieval_attempts") or []
    if any(
        a.get("retrieval_id") == rid and a.get("stage") == "simple" for a in existing
    ):
        return {}

    # V8 Multi-Query Expand: generate alternative query reformulations
    extra_queries: List[str] = []
    if _expand_fn is not None and v7_config.V8_ENABLE_MULTI_QUERY:
        try:
            extra_queries = _expand_fn(active_q, n=v7_config.V8_EXPAND_N) or []
        except Exception as exc:
            logger.warning("expand_fn failed, falling back to single query: %s", exc)
            extra_queries = []

    all_queries = [active_q] + extra_queries

    # Run vector + BM25 for each query; collect per-query result lists for RRF
    all_vector_lists: List[List[dict]] = []
    all_bm25_lists: List[List[dict]] = []

    for q in all_queries:
        v_res = _vector_search(query=q, filters=safe_filters, top_k=plan["top_k"])
        b_res = bm25_search(query=q, filters=safe_filters, top_k=plan["top_k"])
        all_vector_lists.append(v_res)
        all_bm25_lists.append(b_res)

    # top_score anchored to original query only (threshold gate must not be
    # inflated by low-relevance passages from expanded queries)
    vector_results = all_vector_lists[0]
    top_score = max((p.get("score", 0.0) for p in vector_results), default=0.0)

    # RRF merge across all query result lists
    passages = rrf_merge(*all_vector_lists, *all_bm25_lists, top_k=plan["top_k"])
    if not passages:
        passages = sorted(
            vector_results, key=lambda x: x.get("score", 0.0), reverse=True
        )

    # BM25 guarantee: ensure top-3 BM25 results from each query variant are always
    # included. Handles cases where BM25 finds exact-match factoid chunks that
    # vector search misses (e.g. "как часто" → "не реже чем один раз в пять лет").
    # Identity, not the bare chunk_id: the latter is a per-source counter and
    # collides across documents, so a BM25 hit would be dropped as a duplicate
    # of a same-numbered chunk from an unrelated file. rrf_merge keys the same way.
    existing_ids = {passage_identity(p) for p in passages}
    for bm25_list in all_bm25_lists:
        for p in bm25_list[:3]:
            pid = passage_identity(p)
            if pid not in existing_ids:
                passages.append(p)
                existing_ids.add(pid)

    _, metrics = compute_attempt_metrics(original_q, active_q, passages, plan)
    metrics["retrieval_type"] = "hybrid_rrf"

    # V8 Evidence Assess: light rerank to populate reranker scores in metrics
    if _reranker_fn is not None and v7_config.V8_ENABLE_EVIDENCE_ASSESS and passages:
        top_k = v7_config.V8_SIMPLE_RERANK_TOP_K
        try:
            reranked = _reranker_fn(active_q, passages[:top_k], top_k)
            if reranked:
                reranker_scores = [p.get("score", 0.0) for p in reranked]
                metrics["reranker_top1"] = reranked[0].get("score", 0.0)
                metrics["reranker_top3_mean"] = sum(reranker_scores[:3]) / max(
                    len(reranker_scores[:3]), 1
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker failed, skipping V8 rerank scores: %s", exc)

    return {
        "retrieval_attempts": [
            RetrievalAttempt(
                retrieval_id=rid,
                stage="simple",
                passages=passages,
                top_score=top_score,
                attempt_plan=dict(plan),
                metrics=metrics,
            )
        ],
        "status_message": f"Found {len(passages)} fragments (hybrid search).",
    }
