"""Tests for the simple branch terminal-decision evaluator."""

from __future__ import annotations

import pytest

from src.v7 import pack_context as pc
from src.v7.nodes.evaluate_triage import evaluate_triage

PLAN = {
    "threshold": 0.4,
    "min_passages": 2,
    "min_keyword_overlap": 0.1,
    "borderline_threshold": 0.3,
    "max_single_doc_ratio": 1.0,
}


@pytest.fixture(autouse=True)
def _no_expander():
    pc.set_crossref_expander(None)
    yield
    pc.set_crossref_expander(None)


def _p(i, text, score=0.8, source="doc.pdf"):
    return {
        "chunk_id": i,
        "text": text,
        "vector_score": score,
        "metadata": {"source": source},
    }


def _state(passages, query="медосмотр водителей", plan=PLAN, error=False):
    return {
        "query": query,
        "active_query": query,
        "plan": plan,
        "retrieval_attempts": [
            {
                "stage": "simple",
                "passages": passages,
                "attempt_plan": plan,
                "retrieval_error": error,
            }
        ],
    }


def test_sufficient_context_generates():
    ps = [
        _p(1, "медосмотр водителей проводится ежегодно"),
        _p(2, "медосмотр водителей обязателен"),
    ]
    out = evaluate_triage(_state(ps))
    assert out["route_decision"] == "generate"
    assert out["route_reason"] == "context_sufficient"
    assert out["obligations_unmet"] == []
    assert out["final_context"] and out["final_passages"] == out["final_context"]
    assert out["sufficient"] is True
    assert out["technical_failure"] is False


def test_no_attempts_is_retrieval_error():
    out = evaluate_triage({"query": "q", "plan": PLAN, "retrieval_attempts": []})
    assert out["route_decision"] == "abstain"
    assert out["route_reason"] == "retrieval_error"
    assert out["technical_failure"] is True
    assert out["sufficient"] is False


def test_failed_attempt_is_retrieval_error_not_empty_pool():
    out = evaluate_triage(_state([], error=True))
    assert out["route_reason"] == "retrieval_error"
    assert out["technical_failure"] is True


def test_enumeration_query_escalates():
    ps = [
        _p(1, "кто проходит медосмотр: а) водители"),
        _p(2, "б) машинисты медосмотр"),
    ]
    out = evaluate_triage(_state(ps, query="кто проходит медосмотр"))
    assert out["route_decision"] == "complex"
    assert out["route_reason"] == "enumeration_intent"
    assert "enumeration_complete" in out["obligations_unmet"]


def test_open_reference_escalates():
    ps = [
        _p(1, "медосмотр водителей в соответствии с пунктом 15"),
        _p(2, "медосмотр водителей ежегодно"),
    ]
    out = evaluate_triage(_state(ps))
    assert out["route_decision"] == "complex"
    assert out["route_reason"] == "refs_unresolved"


def test_empty_pool_abstains():
    out = evaluate_triage(_state([]))
    assert out["route_decision"] == "abstain"
    assert out["route_reason"] == "empty_pool"
    assert out["final_context"] == []


def test_degraded_pack_is_technical_failure_but_still_escalates():
    def boom(passages, query):
        raise RuntimeError("expander down")

    pc.set_crossref_expander(boom)
    ps = [
        _p(1, "медосмотр водителей ежегодно"),
        _p(2, "медосмотр водителей обязателен"),
    ]
    out = evaluate_triage(_state(ps))
    assert out["technical_failure"] is True
    assert out["route_decision"] == "complex"
    assert out["route_reason"] == "refs_unresolved"


def test_snapshot_is_packed_and_carries_its_own_plan():
    ps = [
        _p(1, "кто проходит медосмотр: а) водители"),
        _p(2, "б) машинисты медосмотр"),
    ]
    out = evaluate_triage(_state(ps, query="кто проходит медосмотр"))
    snap = out["fallback_snapshot"]
    assert snap["plan"] == PLAN
    assert snap["origin"] == "fallback_snapshot"
    assert snap["packed"] is True
    assert snap["passages"] == out["final_context"]


def test_generate_branch_clears_the_snapshot():
    ps = [
        _p(1, "медосмотр водителей ежегодно"),
        _p(2, "медосмотр водителей обязателен"),
    ]
    out = evaluate_triage(_state(ps))
    assert out["fallback_snapshot"] == {}
