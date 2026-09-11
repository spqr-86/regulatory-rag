"""Tests for the complex branch terminal-decision evaluator."""

from __future__ import annotations

import pytest

from src.v7 import pack_context as pc
from src.v7.contract import COMPLEX_REASONS, OBL_REFS
from src.v7.nodes.evaluate_complex import evaluate_complex, route_after_decision

PLAN = {
    "threshold": 0.35,
    "min_passages": 2,
    "min_keyword_overlap": 0.1,
    "borderline_threshold": 0.30,
    "max_single_doc_ratio": 1.0,
}


@pytest.fixture(autouse=True)
def _no_expander():
    pc.set_crossref_expander(None)
    yield
    pc.set_crossref_expander(None)


def _p(i, text, score=0.7, source="doc.pdf"):
    return {
        "chunk_id": i,
        "text": text,
        "vector_score": score,
        "metadata": {"source": source},
    }


def _snapshot(passages, query="медосмотр водителей", plan=PLAN):
    return {
        "passages": passages,
        "plan": dict(plan),
        "active_query": query,
        "origin": "fallback_snapshot",
        "packed": True,
        "pack_status": "ok",
    }


def _state(complex_passages, query="медосмотр водителей", snapshot=None, error=False):
    state = {
        "query": query,
        "active_query": query,
        "plan": PLAN,
        "retrieval_attempts": [
            {"stage": "simple", "passages": [], "attempt_plan": PLAN},
            {
                "stage": "complex",
                "passages": complex_passages,
                "attempt_plan": PLAN,
                "retrieval_error": error,
            },
        ],
    }
    if snapshot is not None:
        state["fallback_snapshot"] = snapshot
    return state


def test_merged_candidate_accepted():
    ps = [
        _p(1, "медосмотр водителей ежегодно"),
        _p(2, "медосмотр водителей обязателен"),
    ]
    out = evaluate_complex(_state(ps))
    assert out["route_decision"] == "generate"
    assert out["route_reason"] == "complex_sufficient"
    assert out["final_context"]


def test_merged_candidate_keeps_simple_evidence_ahead_of_complex_noise(monkeypatch):
    """Escalation may add evidence, but it must not displace the direct simple norm."""
    monkeypatch.setattr(pc.v7_config, "SIMPLE_TOP_K", 2)
    monkeypatch.setattr(pc.v7_config, "FINAL_MERGE_TOP_K", 4)
    direct = _p(1, "медосмотр водителей проводится ежегодно", score=0.4)
    simple = [direct, _p(2, "медосмотр водителей обязателен", score=0.4)]
    complex_noise = [
        _p(i, f"медосмотр водителей посторонний фрагмент {i}", score=0.99)
        for i in range(10, 16)
    ]
    snap = _snapshot(simple)
    state = _state(complex_noise, snapshot=snap)
    state["retrieval_attempts"][0]["passages"] = simple

    out = evaluate_complex(state)

    assert out["route_decision"] == "generate"
    assert out["route_reason"] == "complex_sufficient"
    assert out["final_context"][0]["chunk_id"] == direct["chunk_id"]


def test_exhausted_queue_abstains_with_a_verdict_on_the_returned_context():
    out = evaluate_complex(_state([]))
    assert out["route_decision"] == "abstain"
    assert out["route_reason"] == "complex_exhausted"
    assert out["final_context"] == []
    assert out["sufficiency_details"].get("passage_count", 0) == 0
    assert out["final_score"] == 0.0


def test_rejected_candidates_are_recorded_for_telemetry():
    ps = [_p(1, "совершенно посторонний текст"), _p(2, "ещё посторонний текст")]
    out = evaluate_complex(_state(ps))
    assert out["route_decision"] == "abstain"
    assert out["rejected_candidates"]
    assert {"origin", "obligations_unmet"} <= set(out["rejected_candidates"][0])


def test_technical_failure_uses_complex_exhausted_not_retrieval_error():
    out = evaluate_complex({"query": "q", "plan": PLAN, "retrieval_attempts": []})
    assert out["route_reason"] == "complex_exhausted"
    assert out["route_reason"] in COMPLEX_REASONS
    assert out["technical_failure"] is True


def test_fallback_snapshot_checked_under_its_own_plan():
    strict = {**PLAN, "min_passages": 99}
    snap = _snapshot(
        [
            _p(9, "медосмотр водителей ежегодно"),
            _p(10, "медосмотр водителей обязателен"),
        ]
    )
    state = _state([], snapshot=snap)
    state["retrieval_attempts"][-1]["attempt_plan"] = strict
    state["plan"] = strict
    out = evaluate_complex(state)
    assert out["route_decision"] == "generate"
    assert out["route_reason"] == "complex_fallback_accepted"


def test_packed_snapshot_is_not_expanded_again():
    calls = []
    pc.set_crossref_expander(lambda ps, q: (calls.append(q), list(ps))[1])
    snap = _snapshot(
        [
            _p(1, "медосмотр водителей ежегодно"),
            _p(2, "медосмотр водителей обязателен"),
        ]
    )
    evaluate_complex(_state([], snapshot=snap))
    assert calls == []


def test_best_effort_only_for_the_last_candidate_of_the_full_queue():
    query = "кто проходит медосмотр"
    snap = _snapshot([], query=query)
    ps = [_p(1, "а) водители медосмотр"), _p(2, "б) машинисты медосмотр")]
    state = _state(ps, query=query, snapshot=snap)
    out = evaluate_complex(state)
    assert (
        out["route_reason"] != "enumeration_best_effort"
        or out["route_decision"] == "abstain"
    )


def test_enumeration_best_effort_on_the_last_candidate():
    query = "кто проходит медосмотр"
    ps = [_p(1, "а) водители медосмотр"), _p(2, "б) машинисты медосмотр")]
    snap = _snapshot(ps, query=query)
    out = evaluate_complex(_state(ps, query=query, snapshot=snap))
    assert out["route_decision"] == "generate"
    assert out["route_reason"] == "enumeration_best_effort"
    assert out["obligations_unmet"] == ["enumeration_complete"]


def test_refs_best_effort_on_the_last_candidate():
    snap = _snapshot(
        [
            _p(1, "медосмотр водителей по пункту 15"),
            _p(2, "медосмотр водителей обязателен"),
        ]
    )
    state = _state([], snapshot=snap)

    out = evaluate_complex(state)

    assert out["route_decision"] == "generate"
    assert out["route_reason"] == "refs_best_effort"
    assert out["obligations_unmet"] == [OBL_REFS]


def test_route_after_decision_reads_the_decision():
    assert route_after_decision({"route_decision": "generate"}) == "generate"
    assert route_after_decision({}) == "abstain"
