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
        "completion_tokens": 0,
        "reasoning_tokens": 0,
        "unpriced_models": [],
        "context": "",
        "route_reason": None,
        "abstain_reason": None,
        "generation_failed": False,
        "generation_error": {},
        "technical_failure": False,
        "path": "simple",
        "retrieval_attempts": [],
        "usage": [],
        "usage_unknown": False,
        "cost_incomplete": False,
        "prompt_tokens": 0,
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


def test_outputs_sum_tokens_across_calls():
    outputs = state_to_outputs(
        {
            "llm_usage": [
                {"completion_tokens": 10, "reasoning_tokens": 7},
                {"completion_tokens": 20, "reasoning_tokens": 5},
                {},
            ]
        }
    )
    assert outputs["completion_tokens"] == 30
    assert outputs["reasoning_tokens"] == 12


def test_target_measures_elapsed_without_extra_graph_calls(monkeypatch):
    from unittest.mock import MagicMock

    import eval.ls.target as target_module
    import src.backends.vector_store as store_module
    import src.v7.bridge as bridge_module
    import src.v7.graph as graph_module
    import src.v7.runner as runner_module

    graph = MagicMock()
    build_graph = MagicMock(return_value=graph)
    runner = MagicMock(
        return_value=(
            {
                "answer": "Ответ",
                "llm_usage": [{"completion_tokens": 12, "reasoning_tokens": 8}],
            },
            "query-id",
        )
    )
    monkeypatch.setattr(store_module, "get_vector_store_backend", MagicMock())
    monkeypatch.setattr(bridge_module, "build_full_v7_runtime", MagicMock())
    monkeypatch.setattr(graph_module, "build_graph", build_graph)
    monkeypatch.setattr(runner_module, "default_writer", MagicMock())
    monkeypatch.setattr(runner_module, "run_query", runner)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(target_module.time, "perf_counter", lambda: next(ticks))
    outputs = target_module.make_target()({"question": "Вопрос"})
    assert outputs["elapsed_sec"] == 2.5
    assert outputs["completion_tokens"] == 12
    assert outputs["reasoning_tokens"] == 8
    runner.assert_called_once()
    assert runner.call_args.args == (graph.compile.return_value, "Вопрос")


def test_parent_context_stats_are_summed_into_outputs():
    state = {
        "answer": "a",
        "parent_context_stats": [
            {
                "replaced": 3,
                "children": 1,
                "deduplicated": 2,
                "truncated": 2,
                "slots_lost": 3,
                "unavailable_reason": None,
            },
            {
                "replaced": 0,
                "children": 4,
                "deduplicated": 0,
                "truncated": 1,
                "slots_lost": 4,
                "unavailable_reason": "stale",
            },
        ],
    }
    o = state_to_outputs(state)
    assert (
        o["parent_replaced"],
        o["parent_children_kept"],
        o["parent_deduplicated"],
    ) == (3, 5, 2)
    assert o["parent_unavailable"] == ["stale"]
    assert o["parent_truncated"] == 3
    assert o["parent_slots_lost"] == 7


def test_no_parent_keys_when_flag_off():
    assert "parent_replaced" not in state_to_outputs({"answer": "a"})
