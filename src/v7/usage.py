"""Per-call LLM token accounting for the v7 pipeline.

Roadmap step 4a: token counts extracted from the provider response used to go
into the log only — the runner never saw them, so a run reported quality
without price. These helpers carry usage from ``bridge`` through the graph
state (key :data:`LLM_USAGE_KEY`) up to the eval runner, which converts tokens
to money (pricing lives in ``eval/pricing.py``: the pipeline counts tokens,
the runner counts dollars).

Contract for injected LLM functions: they may return either the bare value
(old contract, stubs and tests) or ``(value, usage)``. :func:`unpack` accepts
both, so nothing breaks when an fn is not usage-aware.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, TypedDict

# Key under which usage records accumulate in RAGState.
LLM_USAGE_KEY = "llm_usage"


class LLMUsage(TypedDict, total=False):
    """One LLM call's token usage.

    ``node``: which pipeline node made the call ("generate", "expand").
    ``stage``: which retrieval path the call belongs to ("simple", "complex") —
    stamped by the node, since only it knows the path.
    """

    model: str
    node: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    # How many passages the call put in its prompt. Only the generate call sets
    # it, and only it can: cross-reference expansion happens inside the fn,
    # after the graph state was written (issue #22).
    n_passages: int


def _as_int(value: Any) -> int:
    """None / missing / non-numeric → 0: a missing count must not poison sums."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def usage_from_response(response: Any, model: str, node: str) -> LLMUsage:
    """Read token counts off a LangChain response.

    Providers disagree on key names: LangChain normalises to
    ``usage_metadata.input_tokens/output_tokens``, OpenAI's raw payload uses
    ``prompt_tokens/completion_tokens``, and some models surface it only under
    ``response_metadata["token_usage"]``. All three are read; absent counts are
    zero, never None.
    """
    meta = getattr(response, "usage_metadata", None) or {}
    if not meta:
        meta = (getattr(response, "response_metadata", None) or {}).get(
            "token_usage"
        ) or {}

    prompt = meta.get("input_tokens", meta.get("prompt_tokens"))
    completion = meta.get("output_tokens", meta.get("completion_tokens"))

    return {
        "model": model,
        "node": node,
        "prompt_tokens": _as_int(prompt),
        "completion_tokens": _as_int(completion),
    }


def unpack(result: Any) -> Tuple[Any, List[LLMUsage]]:
    """Split ``(value, usage)`` from an injected fn; pass a bare value through.

    A two-element tuple whose second element is a usage dict (or list of them)
    is treated as the usage-aware contract. Anything else is the value itself.
    """
    if isinstance(result, tuple) and len(result) == 2:
        value, raw = result
        if isinstance(raw, dict):
            return value, [raw]
        if isinstance(raw, list) and all(isinstance(u, dict) for u in raw):
            return value, list(raw)
    return result, []


def stamp_stage(usages: Iterable[LLMUsage], stage: str) -> List[LLMUsage]:
    """Return copies of ``usages`` tagged with the retrieval path."""
    return [{**u, "stage": stage} for u in usages]


def sum_usage(usages: Iterable[LLMUsage]) -> Dict[str, Dict[str, int]]:
    """Total prompt/completion tokens per model — the input to cost maths."""
    totals: Dict[str, Dict[str, int]] = {}
    for u in usages:
        bucket = totals.setdefault(
            u.get("model", "unknown"), {"prompt_tokens": 0, "completion_tokens": 0}
        )
        bucket["prompt_tokens"] += _as_int(u.get("prompt_tokens"))
        bucket["completion_tokens"] += _as_int(u.get("completion_tokens"))
    return totals
