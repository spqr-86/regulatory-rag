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
    assert o == {
        "answer": "",
        "route_decision": None,
        "chunk_ids": [],
        "sources": [],
        "cost_usd": 0.0,
        "unpriced_models": [],
        "context": "",
    }


def test_cost_from_llm_usage():
    state = {
        "answer": "x",
        "llm_usage": [
            {
                "model": "deepseek/deepseek-v4.1-flash",
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
            }
        ],
    }
    o = state_to_outputs(state)
    assert o["cost_usd"] == pytest.approx(0.15 + 0.60)
    assert o["unpriced_models"] == []


def test_unknown_model_is_reported_not_priced():
    state = {"llm_usage": [{"model": "no/such-model", "prompt_tokens": 10}]}
    o = state_to_outputs(state)
    assert o["cost_usd"] == 0.0
    assert o["unpriced_models"] == ["no/such-model"]


def test_context_is_what_the_generator_saw():
    # The generator reads final_context; the citation judge needs the same text.
    from src.v7.context_format import format_generation_context

    ctx = [_p("2464.pdf", 107)]
    state = {"answer": "a", "final_context": ctx, "final_passages": [_p("x.pdf", 1)]}
    assert state_to_outputs(state)["context"] == format_generation_context(ctx)


def test_context_falls_back_to_final_passages():
    from src.v7.context_format import format_generation_context

    passages = [_p("2464.pdf", 107)]
    state = {"answer": "a", "final_passages": passages}
    assert state_to_outputs(state)["context"] == format_generation_context(passages)
