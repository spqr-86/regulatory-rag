"""Tests for evaluate_complex node."""

from __future__ import annotations

import pytest

from src.v7.nodes.evaluate_complex import evaluate_complex, route_after_eval_complex

PLAN = {
    "threshold": 0.5,
    "min_passages": 2,
    "min_keyword_overlap": 0.2,
    "max_single_doc_ratio": 1.0,
}


class TestEvaluateComplex:
    @pytest.mark.unit
    def test_sufficient_from_merged(self):
        attempts = [
            {
                "passages": [
                    {
                        "chunk_id": "c1",
                        "text": "ограждение лестница",
                        "score": 0.7,
                        "vector_score": 0.7,
                        "doc_id": "d1",
                    },
                    {
                        "chunk_id": "c2",
                        "text": "ограждение балкон",
                        "score": 0.6,
                        "vector_score": 0.6,
                        "doc_id": "d2",
                    },
                ],
                "attempt_plan": PLAN,
            },
            {
                "passages": [
                    {
                        "chunk_id": "c3",
                        "text": "лестница нормы ограждение",
                        "score": 0.55,
                        "vector_score": 0.55,
                        "doc_id": "d3",
                    },
                ],
                "attempt_plan": PLAN,
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": attempts,
        }
        result = evaluate_complex(state)
        assert result["sufficient"] is True
        assert len(result["final_passages"]) == 3

    @pytest.mark.unit
    def test_fallback_used(self):
        """When merged and last fail, fallback passages should be used."""
        attempts = [
            {
                "passages": [
                    {
                        "chunk_id": "c1",
                        "text": "нерелевантное",
                        "score": 0.1,
                        "doc_id": "d1",
                    },
                ],
                "attempt_plan": PLAN,
            },
        ]
        fallback = [
            {
                "chunk_id": "f1",
                "text": "ограждение лестница",
                "score": 0.7,
                "vector_score": 0.7,
                "doc_id": "d1",
            },
            {
                "chunk_id": "f2",
                "text": "ограждение балкон",
                "score": 0.6,
                "vector_score": 0.6,
                "doc_id": "d2",
            },
        ]
        state = {
            "query": "ограждение",
            "active_query": "ограждение",
            "retrieval_attempts": attempts,
            "fallback_passages": fallback,
            "fallback_score": 0.7,
        }
        result = evaluate_complex(state)
        assert result["sufficient"] is True
        assert result["final_passages"] == fallback

    @pytest.mark.unit
    def test_full_failure(self):
        attempts = [
            {
                "passages": [{"text": "нерелевантное", "score": 0.1, "doc_id": "d1"}],
                "attempt_plan": PLAN,
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": attempts,
        }
        result = evaluate_complex(state)
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_no_attempts(self):
        result = evaluate_complex({"retrieval_attempts": []})
        assert result["sufficient"] is False


class TestRouteAfterEvalComplex:
    @pytest.mark.unit
    def test_sufficient_end(self):
        assert route_after_eval_complex({"sufficient": True}) == "end"

    @pytest.mark.unit
    def test_fail_abstain(self):
        assert route_after_eval_complex({"sufficient": False}) == "abstain"


class TestEnumerationEscalationNoAbstain:
    """C2+C3 integration: triage=sufficient + enumeration → rag_complex escalation.

    Pipeline flow being tested:
    1. evaluate_triage returns sufficient=True for an enumeration query and
       additionally stores fallback_passages (C2 fix).
    2. route_after_triage sends the state to rag_complex (existing logic).
    3. rag_complex returns complex passages that fail the complex-path gates.
    4. evaluate_complex checks fallback_passages against the *simple* plan (C3
       fix) rather than the stricter complex plan — and returns sufficient=True.
    """

    # Simple plan mirrors what rag_simple stores as attempt_plan.
    SIMPLE_PLAN = {
        "threshold": 0.50,
        "min_passages": 5,
        "min_keyword_overlap": 0.15,
        "max_single_doc_ratio": 0.8,
        "borderline_threshold": 0.38,
        "require_multi_doc": False,
    }

    # Complex plan has stricter thresholds; fallback would fail these gates.
    COMPLEX_PLAN = {
        "threshold": 0.35,
        "min_passages": 8,
        "min_keyword_overlap": 0.20,
        "max_single_doc_ratio": 0.7,
        "borderline_threshold": 0.30,
        "require_multi_doc": False,
    }

    @pytest.mark.unit
    def test_pipeline_does_not_abstain_when_fallback_passes_simple_gates(self):
        """evaluate_complex should NOT abstain when fallback passes simple-plan gates.

        Scenario:
        - Simple attempt succeeded (triage=sufficient, enumeration query) and
          stored fallback_passages alongside final_passages (C2).
        - Complex attempt returned passages that fail complex-plan hard gates.
        - evaluate_complex must evaluate fallback_passages against the simple
          plan (C3) and return sufficient=True — no abstain.
        """
        # Passages that pass simple gates (min_passages=5, keyword_overlap>=0.15)
        # but would fail complex gates (min_passages=8).
        fallback_passages = [
            {
                "chunk_id": f"f{i}",
                "text": "какие документы нужны для оформления охраны труда",
                "score": 0.7,
                "vector_score": 0.7,
                "doc_id": f"d{i % 3 + 1}",
            }
            for i in range(6)  # 6 passages: passes simple (>=5), fails complex (>=8)
        ]
        # Complex attempt passages — too few and low relevance to pass any gate.
        complex_passages = [
            {
                "chunk_id": "cx1",
                "text": "нерелевантный текст",
                "score": 0.15,
                "vector_score": 0.15,
                "doc_id": "dx1",
            },
        ]

        state = {
            "query": "какие документы нужны для оформления охраны труда",
            "active_query": "какие документы нужны для оформления охраны труда",
            "retrieval_attempts": [
                {
                    "stage": "simple",
                    "passages": fallback_passages,
                    "top_score": 0.7,
                    "attempt_plan": self.SIMPLE_PLAN,
                    "retrieval_id": "rid_simple",
                },
                {
                    "stage": "complex",
                    "passages": complex_passages,
                    "top_score": 0.15,
                    "attempt_plan": self.COMPLEX_PLAN,
                    "retrieval_id": "rid_complex",
                },
            ],
            "fallback_passages": fallback_passages,
            "fallback_score": 0.7,
        }

        result = evaluate_complex(state)

        assert result["sufficient"] is True, (
            "Pipeline must not abstain when fallback_passages pass simple-plan gates. "
            "C3 fix ensures fallback is evaluated against simple_plan, not complex_plan."
        )
        assert result["final_passages"] == fallback_passages

    @pytest.mark.unit
    def test_fallback_evaluated_against_simple_plan_not_complex(self):
        """Verify simple_plan is chosen over complex_plan for fallback gate check.

        Constructs a state where fallback passes simple gates but would fail
        complex gates (min_passages=8 vs 5). Without C3 fix the result would
        be sufficient=False; with the fix it should be sufficient=True.
        """
        # 5 passages — passes simple min_passages=5, fails complex min_passages=8
        fallback_passages = [
            {
                "chunk_id": f"f{i}",
                "text": "какие категории работников проходят инструктаж",
                "score": 0.65,
                "vector_score": 0.65,
                "doc_id": f"d{i % 2 + 1}",
            }
            for i in range(5)
        ]
        low_score_complex = [
            {"chunk_id": "cx1", "text": "нерелевантно", "score": 0.1, "doc_id": "d9"}
        ]

        state = {
            "query": "какие категории работников проходят инструктаж",
            "active_query": "какие категории работников проходят инструктаж",
            "retrieval_attempts": [
                {
                    "stage": "simple",
                    "passages": fallback_passages,
                    "top_score": 0.65,
                    "attempt_plan": self.SIMPLE_PLAN,
                    "retrieval_id": "rid1",
                },
                {
                    "stage": "complex",
                    "passages": low_score_complex,
                    "top_score": 0.1,
                    "attempt_plan": self.COMPLEX_PLAN,
                    "retrieval_id": "rid2",
                },
            ],
            "fallback_passages": fallback_passages,
            "fallback_score": 0.65,
        }

        result = evaluate_complex(state)
        # With C3 fix: simple plan used → min_passages=5 → 5 passages is enough.
        assert result["sufficient"] is True
        assert result["final_passages"] == fallback_passages


class TestFinalMergeTopK:
    """Потолок итоговой выдачи — конфиг, а не число в коде (issue #10)."""

    @pytest.mark.unit
    def test_default_keeps_current_behaviour(self):
        from src.v7.config import v7_config

        assert v7_config.FINAL_MERGE_TOP_K == 24

    @pytest.mark.unit
    def test_merge_uses_configured_top_k(self, monkeypatch):
        from src.v7 import config as config_module
        from src.v7.nodes import evaluate_complex as node_module

        captured: dict = {}

        def fake_merge(attempts, top_k, mmr_lambda=None):
            captured["top_k"] = top_k
            return []

        monkeypatch.setattr(node_module, "merge_all_passages", fake_merge)
        monkeypatch.setattr(config_module.v7_config, "FINAL_MERGE_TOP_K", 7)

        evaluate_complex(
            {
                "query": "ограждение лестница",
                "active_query": "ограждение лестница",
                "retrieval_attempts": [{"passages": [], "attempt_plan": PLAN}],
            }
        )

        assert captured["top_k"] == 7
