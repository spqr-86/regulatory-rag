"""V7 node: rag_simple — fast hybrid retrieval (vector + BM25 → RRF)."""

from __future__ import annotations

# ANCHOR: existing hybrid algorithm with graph-bound search callbacks.
# Input: request state and optional dependencies. Output: one simple attempt.
import logging
from typing import Callable, List, Optional

from src.v7.config import v7_config
from src.v7.hard_gates import compute_attempt_metrics, validate_filters
from src.v7.nlp_core import bm25_search, passage_identity, rrf_merge
from src.v7.scope_filter import matches_filter
from src.v7.state_types import RAGState, RetrievalAttempt
from src.v7.usage import LLM_USAGE_KEY, stamp_stage, unpack

logger = logging.getLogger(__name__)

# ─── Retriever interface (injected at graph build time) ──────────────────


# Default: stub that returns empty. Production: Chroma/Qdrant vector search.
def _default_vector_search(**kwargs) -> List[dict]:
    return []


_vector_search: Callable[..., List[dict]] = _default_vector_search
_expand_fn: Optional[Callable] = None


def set_vector_search(fn: Callable[..., List[dict]]) -> None:
    """Inject vector search implementation. Call once at startup."""
    global _vector_search
    _vector_search = fn


def set_expand_fn(fn: Optional[Callable]) -> None:
    """Inject query expander for V8 multi-query. Signature: fn(query: str, n: int) -> list[str]."""
    global _expand_fn
    _expand_fn = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def rag_simple(state: RAGState, *, dependencies=None) -> RAGState:
    """Fast hybrid retrieval: vector + BM25 → RRF merge."""
    vector_search = (
        dependencies.vector_search if dependencies is not None else _vector_search
    )
    lexical_search = (
        dependencies.bm25_search if dependencies is not None else bm25_search
    )
    expand_fn = dependencies.expand if dependencies is not None else _expand_fn
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
    expand_usage: List[dict] = []
    if expand_fn is not None and v7_config.V8_ENABLE_MULTI_QUERY:
        try:
            expanded, expand_usage = unpack(
                expand_fn(active_q, n=v7_config.V8_EXPAND_N)
            )
            extra_queries = expanded or []
        except Exception as exc:
            logger.warning("expand_fn failed, falling back to single query: %s", exc)
            extra_queries = []
            expand_usage = []

    all_queries = [active_q] + extra_queries

    # Run vector + BM25 for each query; collect per-query result lists for RRF
    all_vector_lists: List[List[dict]] = []
    all_bm25_lists: List[List[dict]] = []
    retrieval_error = False

    try:
        for q in all_queries:
            v_res = vector_search(query=q, filters=safe_filters, top_k=plan["top_k"])
            b_res = lexical_search(query=q, filters=safe_filters, top_k=plan["top_k"])
            if safe_filters:
                v_res = [
                    p
                    for p in v_res
                    if matches_filter(p.get("metadata") or {}, safe_filters)
                ]
                b_res = [
                    p
                    for p in b_res
                    if matches_filter(p.get("metadata") or {}, safe_filters)
                ]
            all_vector_lists.append(v_res)
            all_bm25_lists.append(b_res)
    except Exception as exc:  # noqa: BLE001 — failure is represented in graph state
        logger.warning("rag_simple: retrieval failed: %s", exc)
        retrieval_error = True
        all_vector_lists, all_bm25_lists = [[]], [[]]

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

    return {
        "retrieval_attempts": [
            RetrievalAttempt(
                retrieval_id=rid,
                stage="simple",
                passages=passages,
                top_score=top_score,
                attempt_plan=dict(plan),
                metrics=metrics,
                retrieval_error=retrieval_error,
            )
        ],
        "status_message": f"Found {len(passages)} fragments (hybrid search).",
        LLM_USAGE_KEY: stamp_stage(expand_usage, "simple"),
    }
