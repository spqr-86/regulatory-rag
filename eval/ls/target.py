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
    sources = (p.get("metadata", {}).get("source") for p in passages)
    return {
        "answer": state.get("answer") or "",
        "route_decision": state.get("route_decision"),
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
        "parent_unavailable": reasons,
    }


def make_target() -> Callable[[dict], dict]:
    """Build the v7 graph once and return a target: {"question"} -> outputs.

    Heavy imports are local so that importing this module (unit tests of
    state_to_outputs) needs no settings, API keys or vector store.
    Exceptions are not caught: client.evaluate(error_handling="log") records
    them as an error on the row.
    """
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import build_full_v7_runtime
    from src.v7.graph import build_graph
    from src.v7.runner import default_writer, run_query

    vector_store = get_vector_store_backend(load_existing=True)
    graph = build_graph(runtime=build_full_v7_runtime(vector_store)).compile()
    writer = default_writer()
    # One id per experiment ties its telemetry rows together.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"eval-{stamp}-{uuid.uuid4().hex[:4]}"

    def target(inputs: dict) -> dict:
        start = time.perf_counter()
        state, _query_id = run_query(
            graph, inputs["question"], source="eval", writer=writer, run_id=run_id
        )
        outputs = state_to_outputs(state)
        outputs["elapsed_sec"] = time.perf_counter() - start
        return outputs

    return target
