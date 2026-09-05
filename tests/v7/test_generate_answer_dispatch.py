"""Tests for generate_answer split dispatch (simple vs complex path).

Cheap model handles the simple RAG path; full-quality model handles the
complex path. Dispatch is decided by the latest retrieval_attempts.stage.
"""

from __future__ import annotations

import pytest

from src.v7.nodes import generate_answer as mod


@pytest.fixture(autouse=True)
def _reset_fns():
    mod.set_generate_fns(None, None)
    yield
    mod.set_generate_fns(None, None)


@pytest.mark.unit
def test_simple_stage_calls_simple_fn():
    calls = {"simple": 0, "complex": 0}

    def simple_fn(q, aq, ps):
        calls["simple"] += 1
        return "SIMPLE"

    def complex_fn(q, aq, ps):
        calls["complex"] += 1
        return "COMPLEX"

    mod.set_generate_fns(simple=simple_fn, complex_=complex_fn)
    state = {
        "query": "q",
        "active_query": "q",
        "final_passages": [{"text": "p"}],
        "retrieval_attempts": [{"stage": "simple", "retrieval_id": "r1"}],
    }
    out = mod.generate_answer(state)
    assert out["answer"] == "SIMPLE"
    assert calls == {"simple": 1, "complex": 0}


@pytest.mark.unit
def test_complex_stage_calls_complex_fn():
    calls = {"simple": 0, "complex": 0}

    def simple_fn(q, aq, ps):
        calls["simple"] += 1
        return "SIMPLE"

    def complex_fn(q, aq, ps):
        calls["complex"] += 1
        return "COMPLEX"

    mod.set_generate_fns(simple=simple_fn, complex_=complex_fn)
    state = {
        "query": "q",
        "active_query": "q",
        "final_passages": [{"text": "p"}],
        "retrieval_attempts": [
            {"stage": "simple", "retrieval_id": "r1"},
            {"stage": "complex", "retrieval_id": "r2"},
        ],
    }
    out = mod.generate_answer(state)
    assert out["answer"] == "COMPLEX"
    assert calls == {"simple": 0, "complex": 1}


@pytest.mark.unit
def test_set_generate_fn_backcompat_applies_to_both():
    """The legacy single-fn setter must still work and route both paths."""

    def shared(q, aq, ps):
        return "SHARED"

    mod.set_generate_fn(shared)
    for stage in ("simple", "complex"):
        state = {
            "query": "q",
            "active_query": "q",
            "final_passages": [{"text": "p"}],
            "retrieval_attempts": [{"stage": stage, "retrieval_id": "r"}],
        }
        assert mod.generate_answer(state)["answer"] == "SHARED"


@pytest.mark.unit
def test_no_attempts_defaults_to_simple_path():
    mod.set_generate_fns(
        simple=lambda q, aq, ps: "SIMPLE",
        complex_=lambda q, aq, ps: "COMPLEX",
    )
    state = {
        "query": "q",
        "active_query": "q",
        "final_passages": [{"text": "p"}],
    }
    assert mod.generate_answer(state)["answer"] == "SIMPLE"
