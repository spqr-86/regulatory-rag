"""Retrieval-only entry point into the existing V7 graph."""

# ANCHOR: fresh state in, terminal graph result out; no second routing policy.
# Non-ready results never expose unaccepted candidates as final evidence.

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Literal

from src.v7.graph import build_graph
from src.v7.runtime import V7Runtime
from src.v7.reranker import SharedReranker


@dataclass(frozen=True)
class ScopedRetrievalResult:
    final_context: list[dict]
    outcome: Literal["ready", "empty", "insufficient", "clarification", "failed"]
    route: str | None
    reason: str | None
    attempts: list[dict]
    elapsed_ms: float
    technical_failure: bool
    clarification: str | None = None


def retrieve_context(
    question: str,
    *,
    runtime: V7Runtime,
    filters: dict | None = None,
    deadline: float | None = None,
) -> ScopedRetrievalResult:
    """Invoke the shared graph with one isolated request state.

    ``deadline`` bounds shared reranker waiting/loading; the full Department
    admission/cancellation budget is implemented at the service boundary.
    """
    started = time.monotonic()
    if deadline is not None and isinstance(runtime.rerank, SharedReranker):
        runtime = replace(runtime, rerank=partial(runtime.rerank, deadline=deadline))
    state = (
        build_graph(runtime=runtime, retrieval_only=True)
        .compile()
        .invoke({"query": question, "filters": copy.deepcopy(filters)})
    )
    technical = bool(state.get("technical_failure"))
    route, reason = state.get("route_decision"), state.get("route_reason")
    if state.get("clarify_message"):
        outcome = "clarification"
    elif route == "generate":
        outcome = "ready"
    elif technical:
        outcome = "failed"
    elif reason == "empty_pool" or not any(
        a.get("passages") for a in state.get("retrieval_attempts", [])
    ):
        outcome = "empty"
    else:
        outcome = "insufficient"
    return ScopedRetrievalResult(
        final_context=(
            copy.deepcopy(state.get("final_context", [])) if outcome == "ready" else []
        ),
        outcome=outcome,
        route=route,
        reason=reason,
        attempts=copy.deepcopy(state.get("retrieval_attempts", [])),
        elapsed_ms=(time.monotonic() - started) * 1000,
        technical_failure=technical,
        clarification=state.get("clarify_message"),
    )
