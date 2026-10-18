"""LangSmith target: runs the v7 graph on a dataset example."""

# ANCHOR: Final v7 state → evaluator outputs, usage totals and priced cost.
# make_target measures elapsed_sec around the telemetry runner, without extra calls.

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from eval.pricing import cost_for_usages
from eval.run_retrieval_eval import extract_chunk_ids
from src.v7.context_format import format_generation_context


def state_to_outputs(state: dict) -> dict:
    """Map the final graph state to the `outputs` shape the evaluators expect.

    A domain-gate refusal leaves only {"answer": ""}: no route, no passages, no LLM calls.
    LangSmith cannot price the model (OpenRouter), so cost is computed here from the
    token usage in the state, with the project's rate card.
    `context` is the text the generator got (it reads final_context, like here):
    the citation judge checks the answer's references against it.
    """
    passages = state.get("final_passages") or []
    usages = state.get("llm_usage") or []
    priced = cost_for_usages(usages)
    unknown_usage = any(u.get("usage_unknown") for u in usages)
    attempts = state.get("retrieval_attempts") or []
    sources = (p.get("metadata", {}).get("source") for p in passages)
    return {
        "answer": state.get("answer") or "",
        "route_decision": state.get("route_decision"),
        "route_reason": state.get("route_reason"),
        "abstain_reason": state.get("abstain_reason"),
        "generation_failed": bool(state.get("generation_failed")),
        "generation_error": state.get("generation_error") or {},
        "technical_failure": bool(state.get("technical_failure")),
        "path": (
            "complex"
            if any(a.get("stage") == "complex" for a in attempts)
            else "simple"
        ),
        "retrieval_attempts": attempts,
        "usage": usages,
        "usage_unknown": unknown_usage,
        "cost_incomplete": unknown_usage or bool(priced["unpriced_models"]),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "chunk_ids": extract_chunk_ids(passages),
        # dict.fromkeys: unique sources in order of first appearance
        "sources": list(dict.fromkeys(s for s in sources if s)),
        "cost_usd": priced["cost_usd"],
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "reasoning_tokens": sum(u.get("reasoning_tokens", 0) for u in usages),
        # A model without a rate card counts as $0; keep it visible.
        "unpriced_models": priced["unpriced_models"],
        "context": format_generation_context(
            (state.get("final_context") or []) if "final_context" in state else passages
        ),
        **_parent_summary(state),
    }


def _parent_summary(state: dict) -> dict:
    """Sum replacement diagnostics over every packing pass (empty unless the flag ran)."""
    stats = state.get("parent_context_stats") or []
    if not stats:
        return {}
    reasons = sorted(
        {s["unavailable_reason"] for s in stats if s.get("unavailable_reason")}
    )
    return {
        "parent_replaced": sum(s.get("replaced", 0) for s in stats),
        "parent_children_kept": sum(s.get("children", 0) for s in stats),
        "parent_deduplicated": sum(s.get("deduplicated", 0) for s in stats),
        "parent_truncated": sum(s.get("truncated", 0) for s in stats),
        "parent_slots_lost": sum(s.get("slots_lost", 0) for s in stats),
        "parent_unavailable": reasons,
    }


def make_target(run_id: str | None = None, checkpoint=None) -> Callable[[dict], dict]:
    """Build the v7 graph once and return a target: {"question"} -> outputs.

    Heavy imports are local so that importing this module (unit tests of
    state_to_outputs) needs no settings, API keys or vector store.
    Exceptions are not caught: client.evaluate(error_handling="log") records
    them as an error on the row.
    """
    from config.settings import settings
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import build_full_v7_runtime
    from src.v7.graph import build_graph
    from src.v7.runner import default_writer, run_query

    vector_store = get_vector_store_backend(load_existing=True)
    graph = build_graph(runtime=build_full_v7_runtime(vector_store)).compile()
    writer = default_writer()
    # One id per experiment ties its telemetry rows together.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"eval-{stamp}-{uuid.uuid4().hex[:4]}"

    def target(inputs: dict) -> dict:
        start = time.perf_counter()
        query_id = str(uuid.uuid4())
        try:
            state, query_id = run_query(
                graph,
                inputs["question"],
                source="eval",
                writer=writer,
                run_id=run_id,
                query_id=query_id,
            )
        except Exception as exc:
            if checkpoint:
                checkpoint(
                    {
                        "query_id": query_id,
                        "run_id": run_id,
                        "inputs": inputs,
                        "target_error": type(exc).__name__,
                        "cost_incomplete": True,
                        "usage_unknown": True,
                        "elapsed_sec": time.perf_counter() - start,
                    }
                )
            raise
        outputs = state_to_outputs(state)
        outputs["query_id"] = query_id
        outputs["run_id"] = run_id
        outputs["case_id"] = inputs.get("case_id")
        outputs["example_id"] = inputs.get("example_id")
        outputs["elapsed_sec"] = time.perf_counter() - start
        # Embedding calls are outside llm_usage. Never present accounted LLM cost
        # as the full invoice when a paid embedding provider may have been used.
        outputs["untracked_cost_components"] = (
            [] if settings.EMBEDDING_PROVIDER == "local" else ["embedding"]
        )
        outputs["cost_incomplete"] |= bool(outputs["untracked_cost_components"])
        if checkpoint:
            checkpoint({"inputs": inputs, "outputs": outputs})
        return outputs

    return target
