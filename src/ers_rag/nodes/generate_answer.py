"""V7 node: generate_answer — LLM synthesis from final_passages."""

from __future__ import annotations

from typing import Callable, List, Optional

from src.ers_rag.state_types import RAGState

# ─── Generate interface (stub by default, inject production LLM) ──────────


def _stub_generate(
    query: str,
    active_query: str,
    passages: List[dict],
) -> str:
    """Rule-based stub for testing. Production: LLM call via bridge."""
    if not passages:
        return ""
    texts = "\n\n".join(p.get("text", "") for p in passages[:10])
    return texts


_generate_fn: Optional[Callable[[str, str, List[dict]], str]] = None


def set_generate_fn(fn: Optional[Callable[[str, str, List[dict]], str]]) -> None:
    """Inject LLM generation function. Pass None to restore stub."""
    global _generate_fn
    _generate_fn = fn


# ─── Node ─────────────────────────────────────────────────────────────────


def generate_answer(state: RAGState) -> RAGState:
    """Synthesise final answer from retrieved passages.

    Reads:  query, active_query, final_passages.
    Writes: answer.
    """
    query = state.get("query", "")
    active_query = state.get("active_query", query)
    passages = state.get("final_passages") or []

    fn = _generate_fn if _generate_fn is not None else _stub_generate
    answer = fn(query, active_query, passages)

    return {"answer": answer}
