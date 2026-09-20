"""Retrieval-only graph and dependency isolation gates for issue #56."""

# ANCHOR: same real evaluators, fixed backends, no embedding/model clients.
# Verify parity, no generation, and isolation across interleaved graph instances.

from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.v7.config import v7_config
from src.v7.graph import build_graph
from tests.v7.test_retrieval_characterization import FIXTURE, signature

pytestmark = pytest.mark.unit
CASES = json.loads(FIXTURE.read_text())["cases"]


def fail_generation(*args, **kwargs):
    pytest.fail("retrieval-only invoked generation")


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["name"])
def test_retrieval_only_matches_original_terminal_context(case, monkeypatch):
    from src.v7.retrieval import retrieve_context
    from src.v7.runtime import V7Runtime

    for key, value in case["config"].items():
        monkeypatch.setattr(v7_config, key, value)

    def vector_search(query, filters=None, top_k=12):
        if case["error"]:
            raise RuntimeError("recorded backend failure")
        return copy.deepcopy(case["simple" if top_k == 12 else "complex"])

    runtime = V7Runtime(vector_search=vector_search)
    state = (
        build_graph(
            {"generate_answer": fail_generation, "intent_gate": fail_generation},
            runtime=runtime,
            retrieval_only=True,
        )
        .compile()
        .invoke({"query": case["query"]})
    )
    assert signature(state) == case["expected"]
    result = retrieve_context(case["query"], runtime=runtime)
    expected_outcome = {
        "simple_accepted": "ready",
        "escalation_accepted": "ready",
        "simple_fallback": "ready",
        "empty": "empty",
        "retrieval_error": "failed",
        "clarification": "clarification",
    }[case["name"]]
    assert result.outcome == expected_outcome
    assert result.final_context == case["expected"]["final_context"]
    assert result.elapsed_ms >= 0


def test_bound_graphs_keep_search_pack_and_generation_callbacks(monkeypatch):
    from src.v7.runtime import V7Runtime
    from src.v7.nodes import rag_simple, rag_complex
    from src.v7 import nlp_core

    barrier = Barrier(2)
    calls = []

    def runtime_for(source):
        def vector(query, **kwargs):
            if query.startswith("parallel"):
                barrier.wait(timeout=5)
            return []

        def bm25(query, **kwargs):
            return [
                {
                    "text": f"медосмотр водителей {i}",
                    "chunk_id": i,
                    "metadata": {"source": source},
                    "doc_id": source,
                    "score": 0.8,
                    "vector_score": 0.8,
                }
                for i in range(10)
            ]

        def crossref(passages, query):
            calls.append((source, "pack"))
            assert all(p["metadata"]["source"] == source for p in passages)
            return passages

        def generate(query, active, passages):
            assert all(p["metadata"]["source"] == source for p in passages)
            return source

        return V7Runtime(
            vector_search=vector,
            bm25_search=bm25,
            crossref_expander=crossref,
            generate_simple=generate,
            generate_complex=generate,
        )

    a = build_graph(runtime=runtime_for("A")).compile()
    b = build_graph(runtime=runtime_for("B")).compile()
    monkeypatch.setattr(rag_simple, "_vector_search", fail_generation)
    monkeypatch.setattr(rag_complex, "_vector_search", fail_generation)
    monkeypatch.setattr(nlp_core, "_bm25_index", object())
    for graph, source in ((a, "A"), (b, "B"), (a, "A")):
        assert graph.invoke({"query": "медосмотр водителей"})["answer"] == source
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(g.invoke, {"query": "parallel медосмотр водителей"})
            for g in (a, b)
        ]
        assert [f.result(timeout=10)["answer"] for f in futures] == ["A", "B"]
    assert {source for source, _ in calls} == {"A", "B"}


def test_retrieval_context_can_disable_per_corpus_multi_doc_requirement():
    from src.v7.retrieval import retrieve_context
    from src.v7.runtime import V7Runtime

    result = retrieve_context(
        "Сравни требования закона и ЛНА",
        runtime=V7Runtime(),
        require_multi_doc=False,
    )
    assert result.attempts[0]["attempt_plan"]["require_multi_doc"] is False
