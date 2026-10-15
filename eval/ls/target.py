"""LangSmith target: runs the v7 graph on a dataset example."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from eval.run_retrieval_eval import extract_chunk_ids


def state_to_outputs(state: dict) -> dict:
    """Map the final graph state to the `outputs` shape the evaluators expect.

    A domain-gate refusal leaves only {"answer": ""}: no route, no passages.
    """
    passages = state.get("final_passages") or []
    sources = (p.get("metadata", {}).get("source") for p in passages)
    return {
        "answer": state.get("answer") or "",
        "route_decision": state.get("route_decision"),
        "chunk_ids": extract_chunk_ids(passages),
        # dict.fromkeys: unique sources in order of first appearance
        "sources": list(dict.fromkeys(s for s in sources if s)),
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
        state, _query_id = run_query(
            graph, inputs["question"], source="eval", writer=writer, run_id=run_id
        )
        return state_to_outputs(state)

    return target
