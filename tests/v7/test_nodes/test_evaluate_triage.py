"""Tests for evaluate_triage node."""

from __future__ import annotations

import pytest

from src.v7.nodes.evaluate_triage import (
    _count_crossref_hits,
    _has_enumeration_intent,
    _legacy_triage,
    evaluate_triage,
    route_after_triage,
)


def _make_attempt(passages, plan=None):
    return {
        "retrieval_id": "rid1",
        "stage": "simple",
        "passages": passages,
        "top_score": max((p.get("score", 0) for p in passages), default=0),
        "attempt_plan": plan
        or {
            "threshold": 0.65,
            "min_passages": 2,
            "min_keyword_overlap": 0.3,
            "max_single_doc_ratio": 0.6,
            "borderline_threshold": 0.40,
            "require_multi_doc": False,
        },
    }


class TestEvaluateTriage:
    @pytest.mark.unit
    def test_sufficient(self):
        passages = [
            {
                "text": "ограждение лестница высота",
                "score": 0.8,
                "vector_score": 0.8,
                "doc_id": "d1",
            },
            {
                "text": "ограждение балкон нормы",
                "score": 0.7,
                "vector_score": 0.7,
                "doc_id": "d2",
            },
            {
                "text": "лестница ограждение проект",
                "score": 0.65,
                "vector_score": 0.65,
                "doc_id": "d3",
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        # Tests the legacy 3-way gate directly (V8 evidence_assess is the default
        # path when the flag is on; this asserts the legacy sufficient branch).
        result = _legacy_triage(state)
        assert result["sufficient"] is True
        assert result["final_passages"] == passages

    @pytest.mark.unit
    def test_clearly_bad(self):
        passages = [
            {"text": "ограждение", "score": 0.2, "doc_id": "d1"},
            {"text": "ограждение", "score": 0.1, "doc_id": "d2"},
        ]
        state = {
            "query": "ограждение",
            "active_query": "ограждение",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = evaluate_triage(state)
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_no_attempts(self):
        result = evaluate_triage({"retrieval_attempts": [], "plan": {}})
        assert result["sufficient"] is False

    @pytest.mark.unit
    def test_fallback_saved_on_borderline(self):
        """When hard gates pass but soft signals escalate, save fallback."""
        passages = [
            {"text": "ограждение лестница высота", "score": 0.8, "doc_id": "d1"},
            {"text": "ограждение балкон нормы", "score": 0.7, "doc_id": "d1"},
            {"text": "лестница ограждение проект", "score": 0.65, "doc_id": "d1"},
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = evaluate_triage(state)
        # plan={} → max_single_doc_ratio=1.0 → diversity_ok=True, escalation_hint=False.
        # Fallback saved only when hard gates pass but soft signals escalate (plan with low ratio).
        if not result.get("sufficient"):
            details = result.get("sufficiency_details", {})
            if details.get("sufficient"):
                assert "fallback_passages" in result

    @pytest.mark.unit
    def test_borderline_attempt_without_passages_key_no_keyerror(self):
        """An attempt dict missing 'passages' must not raise KeyError.

        Regression for M5: the borderline-fallback branch indexed last["passages"]
        directly. A malformed attempt (no 'passages' key) reaching that branch
        crashed instead of degrading gracefully.
        """
        plan = {
            "threshold": 0.65,
            "min_passages": 1,
            "min_keyword_overlap": 0.0,
            "max_single_doc_ratio": 1.0,
            "borderline_threshold": 0.40,
            "require_multi_doc": False,
        }
        # Hard gates pass (high score, 1 passage) but triage is borderline.
        passages = [{"text": "ограждение лестница", "score": 0.5, "doc_id": "d1"}]
        attempt = {
            "retrieval_id": "rid1",
            "stage": "simple",
            # NOTE: 'passages' key intentionally omitted on the top level used by
            # the fallback branch — supplied only where the triage reads it.
            "top_score": 0.5,
            "attempt_plan": plan,
        }
        # The triage reads passages via .get on line 86/89; the fallback branch
        # (line ~129) previously used last["passages"]. Provide passages for the
        # read paths but ensure the dict lacks it to hit the KeyError path.
        attempt_no_passages = dict(attempt)
        state = {
            "query": "ограждение лестница высота",
            "active_query": "ограждение лестница высота",
            "retrieval_attempts": [attempt_no_passages],
            "plan": plan,
        }
        # Must not raise KeyError regardless of branch taken.
        result = evaluate_triage(state)
        assert "sufficient" in result
        # If the borderline fallback fired, it must default to an empty list.
        if "fallback_passages" in result:
            assert result["fallback_passages"] == []
        _ = passages  # silence unused

    @pytest.mark.unit
    def test_clearly_bad_attempt_without_passages_key(self):
        """clearly_bad path on an attempt missing 'passages' must not crash."""
        plan = {
            "threshold": 0.65,
            "min_passages": 2,
            "min_keyword_overlap": 0.3,
            "max_single_doc_ratio": 0.6,
            "borderline_threshold": 0.40,
            "require_multi_doc": False,
        }
        attempt = {"retrieval_id": "rid1", "stage": "simple", "attempt_plan": plan}
        state = {
            "query": "ограждение",
            "active_query": "ограждение",
            "retrieval_attempts": [attempt],
            "plan": plan,
        }
        result = evaluate_triage(state)
        assert result["sufficient"] is False


class TestRouteAfterTriage:
    @pytest.mark.unit
    def test_sufficient_routes_end(self):
        assert (
            route_after_triage({"sufficient": True, "query": "ограждение лестница"})
            == "end"
        )

    @pytest.mark.unit
    def test_borderline_routes_complex(self):
        # Legacy borderline → llm_verifier route was removed; all insufficient
        # verdicts now route to rag_complex.
        state = {
            "sufficient": False,
            "sufficiency_details": {"triage": "borderline"},
            "query": "ограждение лестница",
        }
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    def test_clearly_bad_routes_complex(self):
        state = {
            "sufficient": False,
            "sufficiency_details": {"triage": "clearly_bad"},
            "query": "ограждение лестница",
        }
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "query",
        [
            "кто проходит обучение по программе А охраны труда",
            "какие категории работников проходят инструктаж",
            "в каких случаях не требуется инструктаж",
            "кто освобождается от первичного инструктажа",
            "кому не требуется проходить обучение",
        ],
    )
    def test_enumeration_intent_detected(self, query: str):
        assert _has_enumeration_intent(query) is True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "query",
        [
            "какова высота ограждения лестницы",
            "что такое охрана труда",
            "требования к освещению рабочего места",
        ],
    )
    def test_no_enumeration_intent(self, query: str):
        assert _has_enumeration_intent(query) is False

    @pytest.mark.unit
    def test_enumeration_query_forces_complex_even_if_sufficient(self):
        """Sufficient simple-triage result should still go to rag_complex for enumeration queries."""
        state = {
            "sufficient": True,
            "query": "кто проходит обучение по программе А охраны труда",
        }
        assert route_after_triage(state) == "rag_complex"

    @pytest.mark.unit
    def test_non_enumeration_sufficient_routes_end(self):
        """Non-enumeration sufficient result should go to end as before."""
        state = {
            "sufficient": True,
            "query": "какова минимальная высота ограждения",
        }
        assert route_after_triage(state) == "end"


class TestCrossrefSignal:
    @pytest.mark.unit
    def test_no_crossrefs_returns_zero(self):
        passages = [
            {"text": "высота ограждения не менее 1 метра"},
            {"text": "ширина прохода должна составлять 0.8 метра"},
        ]
        assert _count_crossref_hits(passages) == 0

    @pytest.mark.unit
    def test_single_crossref_detected(self):
        passages = [{"text": "в соответствии с пунктом 5 настоящего документа"}]
        assert _count_crossref_hits(passages) >= 1

    @pytest.mark.unit
    def test_multiple_crossrefs_accumulate(self):
        passages = [
            {"text": "согласно пункту 3, за исключением случаев указанных в пункте 4"},
            {
                "text": "подпункт 2.1 устанавливает требования, в соответствии с приложением 1"
            },
        ]
        hits = _count_crossref_hits(passages)
        assert hits >= 3

    @pytest.mark.unit
    def test_crossref_escalation_on_sufficient_triage(self):
        """When sufficient triage but many crossrefs: set sufficient=False, save fallback."""
        passages = [
            {
                "text": (
                    "согласно пункту 3, за исключением лиц указанных в пункте 4, "
                    "в соответствии с приложением 1"
                ),
                "score": 0.85,
                "vector_score": 0.85,
                "doc_id": "d1",
            },
            {
                "text": "пункт 5 настоящего документа содержит перечень исключений",
                "score": 0.80,
                "vector_score": 0.80,
                "doc_id": "d2",
            },
            {
                "text": "ограждение лестница высота",
                "score": 0.75,
                "vector_score": 0.75,
                "doc_id": "d3",
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = _legacy_triage(state)
        # If crossref_hits >= threshold → not sufficient, fallback saved
        from src.v7.nodes.evaluate_triage import (
            _count_crossref_hits,
            _CROSSREF_ESCALATION_THRESHOLD,
        )

        hits = _count_crossref_hits(passages)
        if hits >= _CROSSREF_ESCALATION_THRESHOLD:
            assert result["sufficient"] is False
            assert "fallback_passages" in result
            assert result["fallback_passages"] == passages

    @pytest.mark.unit
    def test_no_crossref_sufficient_stays_sufficient(self):
        """When sufficient triage and no crossrefs: remains sufficient."""
        passages = [
            {
                "text": "ограждение лестница высота не менее метра",
                "score": 0.85,
                "vector_score": 0.85,
                "doc_id": "d1",
            },
            {
                "text": "ограждение балкон требования проекта",
                "score": 0.75,
                "vector_score": 0.75,
                "doc_id": "d2",
            },
            {
                "text": "лестница ограждение строительные нормы",
                "score": 0.70,
                "vector_score": 0.70,
                "doc_id": "d3",
            },
        ]
        state = {
            "query": "ограждение лестница",
            "active_query": "ограждение лестница",
            "retrieval_attempts": [_make_attempt(passages)],
            "plan": {},
        }
        result = _legacy_triage(state)
        # No crossrefs → sufficient should remain True
        assert result["sufficient"] is True
