"""Tests for router node."""

from __future__ import annotations

import pytest

from src.v7.nodes.router import (
    _classify_query,
    clarify_respond,
    route_after_router,
    router,
)


class TestClassifyQuery:
    @pytest.mark.unit
    def test_comparison_query(self):
        require_multi, mmr_lambda = _classify_query("сравни ГОСТ и СП")
        assert require_multi is True
        assert mmr_lambda == 0.5

    @pytest.mark.unit
    def test_factoid_query(self):
        require_multi, mmr_lambda = _classify_query("минимальная высота ограждения")
        assert require_multi is False
        assert mmr_lambda == 0.95

    @pytest.mark.unit
    def test_default_query(self):
        require_multi, mmr_lambda = _classify_query("требования к лестницам")
        assert require_multi is False
        assert mmr_lambda == 0.7


class TestRouter:
    @pytest.mark.unit
    def test_short_query_clarifies(self):
        result = router({"query": "что?"})
        assert "clarify_message" in result

    @pytest.mark.unit
    def test_normal_query_creates_plan(self):
        result = router({"query": "Требования к ограждениям лестничных клеток"})
        assert "plan" in result
        assert "retrieval_id" in result
        assert "active_query" in result

    @pytest.mark.unit
    def test_plan_has_all_fields(self):
        result = router({"query": "Высота ограждения по ГОСТ 12.1.004"})
        plan = result["plan"]
        expected_keys = {
            "top_k",
            "rerank",
            "timeout_ms",
            "threshold",
            "min_passages",
            "min_keyword_overlap",
            "max_single_doc_ratio",
            "borderline_threshold",
            "min_verifier_confidence",
            "require_multi_doc",
            "mmr_lambda",
        }
        assert set(plan.keys()) == expected_keys

    @pytest.mark.unit
    def test_state_cleanup(self):
        result = router({"query": "Требования к ограждениям лестничных клеток"})
        assert result.get("clarify_message") is None
        assert result.get("abstain_reason") is None

    @pytest.mark.unit
    def test_comparison_sets_require_multi_doc(self):
        result = router({"query": "сравни требования ГОСТ и СП к ограждениям"})
        assert result["plan"]["require_multi_doc"] is True

    @pytest.mark.unit
    def test_active_query_expanded_with_glossary_term(self):
        """active_query carries the official term so retrieval matches corpus
        chunks that use the full name, not the short label the user typed.
        """
        result = router(
            {"query": "Минимальная продолжительность обучения по программе Б?"}
        )
        assert "безопасным методам" in result["active_query"]

    @pytest.mark.unit
    def test_active_query_unchanged_without_glossary_term(self):
        query = "Как проводится расследование несчастного случая?"
        result = router({"query": query})
        assert result["active_query"] == query


class TestRouteAfterRouter:
    @pytest.mark.unit
    def test_clarify(self):
        assert route_after_router({"clarify_message": "уточните"}) == "clarify_respond"

    @pytest.mark.unit
    def test_normal(self):
        assert route_after_router({}) == "rag_simple"


class TestClarifyRespond:
    @pytest.mark.unit
    def test_passthrough(self):
        assert clarify_respond({}) == {}
