"""Tests for abstain node."""

from __future__ import annotations

import pytest

from src.v7.nodes.abstain import abstain


class TestAbstain:
    @pytest.mark.unit
    def test_basic_abstain(self):
        state = {
            "query": "Требования к ограждениям",
            "active_query": "Требования к ограждениям",
            "retrieval_attempts": [{"stage": "simple"}],
            "sufficiency_details": {
                "above_threshold": False,
                "enough_evidence": True,
                "keyword_overlap_ok": True,
                "top_score": 0.3,
                "passage_count": 5,
                "keyword_overlap_active": 0.5,
                "keyword_overlap_original": 0.4,
            },
        }
        result = abstain(state)
        assert "abstain_reason" in result
        assert "ниже порога" in result["abstain_reason"]

    @pytest.mark.unit
    def test_fallback_reason(self):
        """No details, no verification → generic message."""
        state = {"query": "test", "retrieval_attempts": []}
        result = abstain(state)
        assert "контекст недостаточен" in result["abstain_reason"]
