"""Retrieval failures are represented in state instead of killing the graph."""

from __future__ import annotations

import pytest

from src.v7.nodes import rag_complex as rc
from src.v7.nodes import rag_simple as rs


def _boom(**kwargs):
    raise RuntimeError("chroma down")


@pytest.mark.unit
def test_simple_search_failure_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(rs, "_vector_search", _boom)
    state = {
        "query": "медосмотр",
        "active_query": "медосмотр",
        "retrieval_id": "r1",
        "plan": {"top_k": 5, "threshold": 0.5, "min_passages": 2},
        "filters": {},
        "retrieval_attempts": [],
    }

    out = rs.rag_simple(state)

    attempt = out["retrieval_attempts"][0]
    assert attempt["retrieval_error"] is True
    assert attempt["passages"] == []


@pytest.mark.unit
def test_complex_search_failure_is_recorded_not_raised(monkeypatch):
    monkeypatch.setattr(rc, "_vector_search", _boom)
    state = {
        "query": "медосмотр",
        "active_query": "медосмотр",
        "retrieval_id": "r2",
        "plan": {"top_k": 5, "threshold": 0.5},
        "filters": {},
        "retrieval_attempts": [],
    }

    out = rc.rag_complex(state)

    attempt = out["retrieval_attempts"][0]
    assert attempt["retrieval_error"] is True
    assert attempt["passages"] == []
