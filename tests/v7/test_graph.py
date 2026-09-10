"""Tests for src/v7/graph.py — Stage 4."""

from __future__ import annotations

import pytest

from src.v7.graph import build_graph
from src.v7.state_types import RAGState


class TestBuildGraph:
    @pytest.mark.unit
    def test_graph_compiles(self):
        """build_graph() returns a compilable StateGraph."""
        g = build_graph()
        app = g.compile()
        assert app is not None

    @pytest.mark.unit
    def test_graph_with_overrides(self):
        """build_graph(overrides) accepts node replacements."""

        def mock_intent_gate(state: RAGState) -> RAGState:
            return {"intent": "noise"}

        g = build_graph({"intent_gate": mock_intent_gate})
        app = g.compile()
        assert app is not None

    @pytest.mark.unit
    def test_noise_query_ends(self):
        """Noise query → intent_gate returns noise → END."""
        app = build_graph().compile()
        result = app.invoke({"query": "hi"})
        assert result["intent"] == "noise"
        # Should not have plan or retrieval_attempts
        assert "plan" not in result or result.get("plan") is None

    @pytest.mark.unit
    def test_short_query_clarifies(self):
        """Short domain query → router → clarify_respond → END."""
        app = build_graph().compile()
        result = app.invoke({"query": "что?"})
        assert result.get("clarify_message") is not None

    @pytest.mark.unit
    def test_domain_query_full_cycle(self):
        """Domain query passes through full graph cycle."""
        app = build_graph().compile()
        result = app.invoke(
            {"query": "Требования к ограждениям лестничных клеток по ГОСТ"}
        )
        # Should have gone through router at minimum
        assert result.get("plan") is not None
        assert result.get("active_query") is not None
        # Should have retrieval_attempts (even if empty from stubs)
        assert "retrieval_attempts" in result

    @pytest.mark.unit
    def test_override_replaces_node(self):
        """Override a node and verify it's used."""
        call_count = {"n": 0}

        def counting_router(state: RAGState) -> RAGState:
            call_count["n"] += 1
            return {
                "clarify_message": "mock clarify",
            }

        app = build_graph({"router": counting_router}).compile()
        result = app.invoke({"query": "Требования к ограждениям лестниц"})
        assert call_count["n"] == 1
        assert result.get("clarify_message") == "mock clarify"

    @pytest.mark.unit
    def test_result_has_abstain_on_empty_retrieval(self):
        """With stub retrievers (empty), should reach abstain."""
        app = build_graph().compile()
        result = app.invoke({"query": "Требования к ограждениям лестничных клеток"})
        # With empty retrieval stubs, graph should reach abstain
        # (empty passages → clearly_bad → rag_complex → evaluate_complex → abstain)
        assert (
            result.get("abstain_reason") is not None
            or result.get("sufficient") is not None
        )


def test_graph_has_no_visual_enrichment_node():
    assert "visual_enrichment" not in build_graph().nodes


def test_graph_compiles():
    assert build_graph().compile() is not None


def test_route_after_triage_is_gone():
    import src.v7.nodes.evaluate_triage as et

    assert not hasattr(et, "route_after_triage")
