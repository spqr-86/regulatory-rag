"""Tests for llm_verifier node."""

from __future__ import annotations

import pytest

from src.v7.nodes.llm_verifier import (
    llm_verifier,
    route_after_verifier,
    set_verify_fn,
)
from src.v7.state_types import VerificationResult


def _make_state(top_score=0.7, iteration=0, min_confidence=0.5):
    # passage text must contain query keywords so stub keyword-overlap check passes.
    return {
        "query": "ограждение лестница",
        "active_query": "ограждение лестница",
        "retrieval_attempts": [
            {
                "passages": [
                    {"text": "ограждение лестничных маршей", "score": top_score}
                ],
                "top_score": top_score,
            }
        ],
        "verify_iteration": iteration,
        "plan": {"min_verifier_confidence": min_confidence},
    }


class TestLlmVerifier:
    @pytest.mark.unit
    def test_sufficient_on_high_score(self):
        result = llm_verifier(_make_state(top_score=0.7))
        assert result["sufficient"] is True
        assert result["verification"]["verdict"] == "sufficient"

    @pytest.mark.unit
    def test_rewrite_on_medium_score(self):
        result = llm_verifier(_make_state(top_score=0.45))
        assert result["sufficient"] is False
        assert result["verification"]["verdict"] == "rewrite"

    @pytest.mark.unit
    def test_escalate_on_low_score(self):
        result = llm_verifier(_make_state(top_score=0.2))
        assert result["sufficient"] is False
        assert result["verification"]["verdict"] == "escalate"

    @pytest.mark.unit
    def test_max_iterations_forces_escalate(self):
        """Rewrite at max iterations → forced escalate."""
        result = llm_verifier(_make_state(top_score=0.45, iteration=2))
        assert result["verification"]["verdict"] == "escalate"

    @pytest.mark.unit
    def test_confidence_gate(self):
        """Low confidence → forced escalate even if verdict was sufficient."""
        result = llm_verifier(_make_state(top_score=0.7, min_confidence=0.95))
        # Stub returns confidence=0.85, which is < 0.95
        assert result["verification"]["verdict"] == "escalate"

    @pytest.mark.unit
    def test_no_attempts(self):
        result = llm_verifier({"retrieval_attempts": [], "plan": {}})
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_sufficient_attempt_without_passages_key_no_keyerror(self):
        """Regression M5: sufficient branch used last['passages'] directly.

        An attempt dict missing the 'passages' key must not KeyError when the
        verdict is sufficient — it should degrade to an empty passage list.
        """

        def _always_sufficient(**_kwargs):
            return VerificationResult(
                verdict="sufficient",
                reason="stub",
                missing_aspects=[],
                confidence=0.99,
            )

        set_verify_fn(_always_sufficient)
        try:
            state = {
                "query": "ограждение",
                "active_query": "ограждение",
                # attempt has no 'passages' key
                "retrieval_attempts": [{"top_score": 0.8}],
                "verify_iteration": 0,
                "plan": {"min_verifier_confidence": 0.0},
            }
            result = llm_verifier(state)
            assert result["sufficient"] is True
            assert result["final_passages"] == []
        finally:
            from src.v7.nodes.llm_verifier import _stub_verify

            set_verify_fn(_stub_verify)


class TestRouteAfterVerifier:
    @pytest.mark.unit
    def test_sufficient(self):
        assert route_after_verifier({"sufficient": True}) == "end"

    @pytest.mark.unit
    def test_rewrite(self):
        state = {"sufficient": False, "verification": {"verdict": "rewrite"}}
        assert route_after_verifier(state) == "rewriter"

    @pytest.mark.unit
    def test_escalate(self):
        state = {"sufficient": False, "verification": {"verdict": "escalate"}}
        assert route_after_verifier(state) == "rag_complex"
