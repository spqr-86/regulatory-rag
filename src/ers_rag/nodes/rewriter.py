"""V7 node: rewriter — query reformulation with doc identifier protection."""

from __future__ import annotations

from typing import Callable, List

from src.ers_rag.nodes.utils import extract_doc_identifiers, make_retrieval_id
from src.ers_rag.state_types import RAGState

# ─── LLM rewrite interface (stub by default) ─────────────────────────────


def _stub_rewrite(
    original_query: str,
    active_query: str,
    rewrite_hint: str,
    missing_aspects: List[str],
) -> str:
    """Stub rewriter. Production: LLM with doc protection prompt."""
    protected_ids = extract_doc_identifiers(original_query)

    aspects = ", ".join(missing_aspects) if missing_aspects else ""
    rewritten = f"{original_query} ({aspects})" if aspects else original_query

    # Safety net: guarantee doc identifiers are preserved
    for doc_id in protected_ids:
        if doc_id not in rewritten:
            rewritten = f"{rewritten} [{doc_id}]"

    return rewritten


_rewrite_fn: Callable[..., str] = _stub_rewrite


def set_rewrite_fn(fn: Callable[..., str]) -> None:
    """Inject production LLM rewriter. Call once at startup."""
    global _rewrite_fn
    _rewrite_fn = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def rewriter(state: RAGState) -> RAGState:
    """Reformulate query based on LLM feedback. Loops back to rag_simple."""
    verification = state.get("verification") or {}
    original_q = state.get("query", "")
    active_q = state.get("active_query", original_q)

    new_query = _rewrite_fn(
        original_query=original_q,
        active_query=active_q,
        rewrite_hint=verification.get("rewrite_hint", ""),
        missing_aspects=verification.get("missing_aspects", []),
    )

    return {
        "active_query": new_query,
        "retrieval_id": make_retrieval_id(new_query, state.get("filters")),
        "verify_iteration": state.get("verify_iteration", 0) + 1,
    }
