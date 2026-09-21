"""Retrieval-only entry point into the existing V7 graph."""

# ANCHOR: fresh state in, terminal graph result out; no second routing policy.
# Non-ready results never expose unaccepted candidates as final evidence.

from __future__ import annotations

import copy
from concurrent.futures import Future
import threading
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Callable, Literal

from src.v7.graph import build_graph
from src.v7.hard_gates import validate_scope_filters
from src.v7.runtime import V7Runtime
from src.v7.reranker import SharedReranker
from src.v7.nodes.router import router


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


class RequestEmbeddingMemo:
    """Request-local single-flight memo for exact prepared query embeddings."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._futures: dict[tuple[str, str], Future[list[float]]] = {}
        self._computations = 0
        self._api_attempts = 0
        self._cache_hits = 0

    def get_or_compute(self, *, space: str, text: str, embed) -> list[float]:
        key = (space, text)
        with self._lock:
            future = self._futures.get(key)
            if future is None:
                future = Future()
                self._futures[key] = future
                owner = True
                self._computations += 1
            else:
                owner = False
                self._cache_hits += 1
        if owner:
            try:
                with self._lock:
                    self._api_attempts += 1
                future.set_result(embed(text))
            except Exception as exc:
                future.set_exception(exc)
        return future.result()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "computations": self._computations,
                "api_attempts": self._api_attempts,
                "cache_hits": self._cache_hits,
            }


def bind_request_embedding(
    runtime: V7Runtime,
    *,
    memo: RequestEmbeddingMemo,
    space: str,
    embed: Callable[[str], list[float]],
) -> V7Runtime:
    """Bind one request-local embedding memo to all dense searches in a runtime.

    Separate runtimes may share ``memo`` for concurrent corpus branches. Search
    results remain corpus/filter specific; only the exact prepared query vector
    is reused. Generic runtimes that do not opt in keep text-search semantics.
    """

    def with_embedding(search: Callable) -> Callable:
        def _search(*, query: str, **kwargs):
            embedding = memo.get_or_compute(space=space, text=query, embed=embed)
            return search(query=query, embedding=embedding, **kwargs)

        return _search

    vector_search = with_embedding(runtime.vector_search)
    complex_search = (
        with_embedding(runtime.complex_vector_search)
        if runtime.complex_vector_search is not None
        else vector_search
    )
    return replace(
        runtime,
        vector_search=vector_search,
        complex_vector_search=complex_search,
    )


def retrieve_context(
    question: str,
    *,
    runtime: V7Runtime,
    filters: dict | None = None,
    strict_scope: bool = False,
    deadline: float | None = None,
    require_multi_doc: bool | None = None,
) -> ScopedRetrievalResult:
    """Invoke the shared graph with one isolated request state.

    ``deadline`` bounds shared reranker waiting/loading; the full Department
    admission/cancellation budget is implemented at the service boundary.
    """
    started = time.monotonic()
    if strict_scope:
        filters = validate_scope_filters(filters or {})
    if deadline is not None and isinstance(runtime.rerank, SharedReranker):
        runtime = replace(runtime, rerank=partial(runtime.rerank, deadline=deadline))
    overrides = None
    if require_multi_doc is not None:

        def scoped_router(state):
            update = router(state)
            if plan := update.get("plan"):
                update = {
                    **update,
                    "plan": {**plan, "require_multi_doc": require_multi_doc},
                }
            return update

        overrides = {"router": scoped_router}
    state = (
        build_graph(overrides, runtime=runtime, retrieval_only=True)
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
