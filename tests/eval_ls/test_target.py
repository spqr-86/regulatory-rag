"""Unit tests for eval/ls/target.py — state mapping only, no graph."""

from __future__ import annotations

import pytest

from eval.ls.target import state_to_outputs

pytestmark = pytest.mark.unit


def _p(source, cid):
    return {
        "text": "t",
        "metadata": {"source": source, "chunk_id": cid},
        "chunk_id": cid,
        "doc_id": source,
    }


def test_maps_answer_route_and_ordered_chunk_ids():
    state = {
        "answer": "Ответ",
        "route_decision": "generate",
        "final_passages": [
            _p("782н.pdf", 431),
            _p("2464.pdf", 107),
            _p("782н.pdf", 431),
        ],
    }
    o = state_to_outputs(state)
    assert o["answer"] == "Ответ"
    assert o["route_decision"] == "generate"
    assert o["chunk_ids"] == ["782н.pdf#431", "2464.pdf#107"]
    assert o["sources"] == ["782н.pdf", "2464.pdf"]


def test_domain_gate_state_without_route():
    o = state_to_outputs({"answer": ""})
    assert o == {"answer": "", "route_decision": None, "chunk_ids": [], "sources": []}
