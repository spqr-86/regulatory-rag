"""Tests for v7 state type contracts."""

from __future__ import annotations

import operator


class TestLiteralTypes:
    """Verify all Literal type aliases exist and have correct values."""

    def test_intent_values(self):
        from src.v7.state_types import Intent

        val: Intent = "noise"
        assert val == "noise"
        val2: Intent = "domain"
        assert val2 == "domain"

    def test_triage_category_values(self):
        from src.v7.state_types import TriageCategory

        val: TriageCategory = "sufficient"
        assert val in ("sufficient", "borderline", "clearly_bad")

    def test_routing_literal_types_exist(self):
        from src.v7.state_types import (
            NextAfterIntent,
        )

        assert NextAfterIntent is not None


class TestRetrievalPlan:
    def test_empty_plan_is_valid(self):
        from src.v7.state_types import RetrievalPlan

        plan: RetrievalPlan = {}
        assert isinstance(plan, dict)

    def test_plan_with_fields(self):
        from src.v7.state_types import RetrievalPlan

        plan: RetrievalPlan = {
            "top_k": 10,
            "rerank": True,
            "threshold": 0.65,
            "require_multi_doc": False,
            "mmr_lambda": 0.9,
        }
        assert plan["top_k"] == 10
        assert plan["rerank"] is True

    def test_plan_has_expected_keys(self):
        from src.v7.state_types import RetrievalPlan

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
        assert set(RetrievalPlan.__annotations__) == expected_keys


class TestRetrievalAttempt:
    def test_attempt_with_fields(self):
        from src.v7.state_types import RetrievalAttempt

        attempt: RetrievalAttempt = {
            "retrieval_id": "abc123",
            "stage": "simple",
            "passages": [{"text": "test"}],
            "top_score": 0.85,
            "attempt_plan": {"top_k": 10},
            "metrics": {"overlap": 0.5},
        }
        assert attempt["stage"] == "simple"
        assert attempt["top_score"] == 0.85

    def test_attempt_has_expected_keys(self):
        from src.v7.state_types import RetrievalAttempt

        expected_keys = {
            "retrieval_id",
            "stage",
            "passages",
            "top_score",
            "attempt_plan",
            "metrics",
        }
        assert set(RetrievalAttempt.__annotations__) == expected_keys


class TestHardGateResult:
    def test_full_result(self):
        from src.v7.state_types import HardGateResult

        result: HardGateResult = {
            "sufficient": True,
            "above_threshold": True,
            "enough_evidence": True,
            "keyword_overlap_ok": True,
            "top_score": 0.78,
            "passage_count": 5,
            "keyword_overlap_active": 0.6,
            "keyword_overlap_original": 0.4,
        }
        assert result["sufficient"] is True
        assert result["top_score"] == 0.78

    def test_has_expected_keys(self):
        from src.v7.state_types import HardGateResult

        expected_keys = {
            "sufficient",
            "above_threshold",
            "enough_evidence",
            "keyword_overlap_ok",
            "top_score",
            "passage_count",
            "keyword_overlap_active",
            "keyword_overlap_original",
        }
        assert set(HardGateResult.__annotations__) == expected_keys


class TestSufficiencyResult:
    def test_full_result(self):
        from src.v7.state_types import SufficiencyResult

        result: SufficiencyResult = {
            "sufficient": True,
            "above_threshold": True,
            "enough_evidence": True,
            "keyword_overlap_ok": True,
            "diversity_ok": True,
            "escalation_hint": False,
            "triage": "sufficient",
            "top_score": 0.78,
            "keyword_overlap_active": 0.6,
            "keyword_overlap_original": 0.4,
            "passage_count": 5,
            "unique_docs": 3,
            "max_doc_ratio": 0.4,
        }
        assert result["triage"] == "sufficient"
        assert result["unique_docs"] == 3

    def test_has_expected_keys(self):
        from src.v7.state_types import SufficiencyResult

        expected_keys = {
            "sufficient",
            "above_threshold",
            "enough_evidence",
            "keyword_overlap_ok",
            "diversity_ok",
            "escalation_hint",
            "triage",
            "top_score",
            "keyword_overlap_active",
            "keyword_overlap_original",
            "passage_count",
            "unique_docs",
            "max_doc_ratio",
        }
        assert set(SufficiencyResult.__annotations__) == expected_keys


class TestRAGState:
    def test_minimal_state(self):
        from src.v7.state_types import RAGState

        state: RAGState = {"query": "тест", "filters": {}}
        assert state["query"] == "тест"

    def test_has_all_sections(self):
        from src.v7.state_types import RAGState

        annotations = set(RAGState.__annotations__)
        assert "query" in annotations
        assert "filters" in annotations
        assert "intent" in annotations
        assert "plan" in annotations
        assert "retrieval_id" in annotations
        assert "active_query" in annotations
        assert "retrieval_attempts" in annotations
        assert "sufficient" in annotations
        assert "final_passages" in annotations
        assert "final_score" in annotations
        assert "fallback_passages" in annotations
        assert "fallback_score" in annotations
        assert "clarify_message" in annotations
        assert "abstain_reason" in annotations
        assert "sufficiency_details" in annotations
        assert "status_message" in annotations

    def test_retrieval_attempts_uses_operator_add(self):
        from src.v7.state_types import RAGState
        import typing

        hints = typing.get_type_hints(RAGState, include_extras=True)
        attempts_hint = hints["retrieval_attempts"]
        assert hasattr(attempts_hint, "__metadata__")
        assert operator.add in attempts_hint.__metadata__


class TestConstants:
    def test_allowed_filter_keys(self):
        from src.v7.state_types import ALLOWED_FILTER_KEYS

        assert "doc_type" in ALLOWED_FILTER_KEYS
        assert "doc_id" in ALLOWED_FILTER_KEYS
        assert isinstance(ALLOWED_FILTER_KEYS, (set, frozenset))
