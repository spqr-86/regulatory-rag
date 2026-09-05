"""V7 node: generate_answer — LLM synthesis from final_passages.

The node dispatches between two injected LLM-backed generators based on which
retrieval path produced the final passages:

* simple path (rag_simple → evaluate_triage → visual_enrichment → here)
  uses ``_generate_fn_simple`` — typically a cheaper model.
* complex path (… → rag_complex → evaluate_complex → visual_enrichment → here)
  uses ``_generate_fn_complex`` — the higher-quality model.

Path is detected from ``state["retrieval_attempts"]``: if the latest attempt
has ``stage == "complex"`` we treat the request as complex.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from src.v7.state_types import RAGState
from src.v7.usage import LLM_USAGE_KEY, stamp_stage, unpack

# ─── Generate interface (stub by default, inject production LLM) ──────────

# The injected fn may return the answer text alone (stub / legacy) or
# ``(text, usage)`` — see src.v7.usage.unpack.
GenerateFn = Callable[[str, str, List[dict]], Any]


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


_generate_fn_simple: Optional[GenerateFn] = None
_generate_fn_complex: Optional[GenerateFn] = None


def set_generate_fn(fn: Optional[GenerateFn]) -> None:
    """Inject a single LLM generation function for both paths.

    Back-compat shim — prefer :func:`set_generate_fns` to split models per
    path. Pass ``None`` to restore the stub on both paths.
    """
    global _generate_fn_simple, _generate_fn_complex
    _generate_fn_simple = fn
    _generate_fn_complex = fn


def set_generate_fns(
    simple: Optional[GenerateFn],
    complex_: Optional[GenerateFn],
) -> None:
    """Inject distinct generation functions for simple and complex paths."""
    global _generate_fn_simple, _generate_fn_complex
    _generate_fn_simple = simple
    _generate_fn_complex = complex_


def _last_stage(state: RAGState) -> str:
    attempts = state.get("retrieval_attempts") or []
    if not attempts:
        return "simple"
    stage = attempts[-1].get("stage", "simple")
    return stage if stage in ("simple", "complex") else "simple"


# ─── Node ─────────────────────────────────────────────────────────────────


def generate_answer(state: RAGState) -> RAGState:
    """Synthesise final answer from retrieved passages.

    Reads:  query, active_query, final_passages, retrieval_attempts.
    Writes: answer, llm_usage (token usage of the generation call, tagged with
            the retrieval path — the runner needs cost split by path).
    """
    query = state.get("query", "")
    active_query = state.get("active_query", query)
    passages = state.get("final_passages") or []

    stage = _last_stage(state)
    fn: Optional[GenerateFn] = (
        _generate_fn_complex if stage == "complex" else _generate_fn_simple
    )
    if fn is None:
        fn = _stub_generate
    answer, usages = unpack(fn(query, active_query, passages))

    return {"answer": answer, LLM_USAGE_KEY: stamp_stage(usages, stage)}
