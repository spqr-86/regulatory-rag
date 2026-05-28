"""Unit tests for eval/metrics.py — inversion and citation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from eval.metrics import (
    compute_citation_doc_match,
    compute_citation_in_retrieval,
    compute_citation_rate,
    compute_inversion_detected,
    compute_inversion_rate,
    parse_citations,
)


class TestComputeInversionDetected:
    def test_empty_must_not_contain_returns_false(self):
        assert (
            compute_inversion_detected("", "повторный инструктаж раз в 6 месяцев")
            is False
        )

    def test_none_like_empty_string(self):
        assert compute_inversion_detected("   ", "любой ответ") is False

    def test_pattern_not_in_answer_returns_false(self):
        assert (
            compute_inversion_detected(
                "раз в год|ежегодно", "инструктаж раз в 6 месяцев"
            )
            is False
        )

    def test_pattern_in_answer_returns_true(self):
        assert (
            compute_inversion_detected(
                "раз в год|ежегодно", "инструктаж проводится раз в год"
            )
            is True
        )

    def test_second_pattern_matches(self):
        assert (
            compute_inversion_detected("раз в год|ежегодно", "тренировки ежегодно")
            is True
        )

    def test_case_insensitive(self):
        assert compute_inversion_detected("РАЗ В ГОД", "инструктаж раз в год") is True

    def test_empty_answer_returns_false(self):
        assert compute_inversion_detected("раз в год", "") is False

    def test_single_pattern(self):
        assert (
            compute_inversion_detected("8 часов", "программа Б — не менее 16 часов")
            is False
        )
        assert (
            compute_inversion_detected("8 часов", "программа Б — не менее 8 часов")
            is True
        )

    def test_partial_match(self):
        # "30%" appears in the answer among other numbers
        assert compute_inversion_detected("30%", "СИЗ: не менее 30% практики") is True

    def test_spaces_in_pattern_are_stripped(self):
        assert compute_inversion_detected(" раз в год | ежегодно ", "раз в год") is True


class TestComputeInversionRate:
    def test_no_results(self):
        assert compute_inversion_rate([]) == 0.0

    def test_no_checkable_rows(self):
        results = [
            {"must_not_contain": "", "inversion_detected": False},
            {"must_not_contain": "", "inversion_detected": False},
        ]
        assert compute_inversion_rate(results) == 0.0

    def test_all_clean(self):
        results = [
            {"must_not_contain": "раз в год", "inversion_detected": False},
            {"must_not_contain": "8 часов", "inversion_detected": False},
        ]
        assert compute_inversion_rate(results) == 0.0

    def test_one_inversion(self):
        results = [
            {"must_not_contain": "раз в год", "inversion_detected": True},
            {"must_not_contain": "8 часов", "inversion_detected": False},
            {"must_not_contain": "", "inversion_detected": False},
        ]
        # 1 inversion out of 2 checkable rows (third row without must_not_contain is excluded)
        assert compute_inversion_rate(results) == pytest.approx(0.5)

    def test_all_inversions(self):
        results = [
            {"must_not_contain": "раз в год", "inversion_detected": True},
            {"must_not_contain": "8 часов", "inversion_detected": True},
        ]
        assert compute_inversion_rate(results) == 1.0


# ── Citation metrics ──────────────────────────────────────────────────────────

_ANSWER_WITH_CITATIONS = (
    "Повторный инструктаж проводится раз в 6 месяцев [Фрагмент 1: ТК РФ, п. 212.1]. "
    "Стажировка — не менее 2 смен [Фрагмент 3: Приказ 2464, п. 8.2]. "
    "Это важное требование."
)
_ANSWER_NO_CITATIONS = (
    "Повторный инструктаж проводится раз в 6 месяцев. Стажировка не менее 2 смен."
)

_PASSAGES = [
    {
        "metadata": {"source": "Трудовой кодекс РФ - Система Охрана труда.pdf"},
        "text": "...",
    },
    {"metadata": {"source": "2464.pdf"}, "text": "..."},
    {"metadata": {"source": "Приказ 2464 об обучении.pdf"}, "text": "..."},
]


class TestParseCitations:
    def test_no_citations(self):
        assert parse_citations("Просто текст без ссылок.") == []

    def test_single_with_section(self):
        result = parse_citations("[Фрагмент 2: Документ X, п. 5.1]")
        assert len(result) == 1
        assert result[0]["n"] == 2
        assert result[0]["doc"] == "Документ X"
        assert result[0]["section"] == "5.1"

    def test_single_without_section(self):
        result = parse_citations("[Фрагмент 4: Документ Y, без пункта]")
        assert len(result) == 1
        assert result[0]["n"] == 4
        assert result[0]["doc"] == "Документ Y"
        assert result[0]["section"] == ""

    def test_multiple_citations(self):
        result = parse_citations(_ANSWER_WITH_CITATIONS)
        assert len(result) == 2
        assert result[0]["n"] == 1
        assert result[1]["n"] == 3

    def test_empty_answer(self):
        assert parse_citations("") == []


class TestCitationRate:
    def test_all_sentences_cited(self):
        answer = (
            "Требование А [Фрагмент 1: Doc, п. 1]. "
            "Требование Б [Фрагмент 2: Doc, п. 2]."
        )
        rate = compute_citation_rate(answer)
        assert rate == pytest.approx(1.0)

    def test_no_sentences_cited(self):
        rate = compute_citation_rate(_ANSWER_NO_CITATIONS)
        assert rate == 0.0

    def test_partial(self):
        # 2 out of 3 sentences have citations
        rate = compute_citation_rate(_ANSWER_WITH_CITATIONS)
        assert 0.0 < rate < 1.0

    def test_empty_answer(self):
        assert compute_citation_rate("") == 0.0


class TestCitationInRetrieval:
    def test_all_in_range(self):
        # citations 1 and 3, passages has 3 elements → both valid
        rate = compute_citation_in_retrieval(_ANSWER_WITH_CITATIONS, _PASSAGES)
        assert rate == pytest.approx(1.0)

    def test_citation_out_of_range(self):
        answer = "Текст [Фрагмент 10: Doc, п. 1]."
        rate = compute_citation_in_retrieval(
            answer, _PASSAGES
        )  # passages len=3, N=10 → out of range
        assert rate == 0.0

    def test_no_citations(self):
        assert compute_citation_in_retrieval(_ANSWER_NO_CITATIONS, _PASSAGES) == 0.0

    def test_empty_passages(self):
        rate = compute_citation_in_retrieval(_ANSWER_WITH_CITATIONS, [])
        assert rate == 0.0


class TestCitationDocMatch:
    def test_doc_matches_source(self):
        # Fragment 1: "Трудовой кодекс" → source[0] = "Трудовой кодекс РФ..."
        answer = "Текст [Фрагмент 1: Трудовой кодекс, п. 1]."
        rate = compute_citation_doc_match(answer, _PASSAGES)
        assert rate == pytest.approx(1.0)

    def test_doc_matches_by_number(self):
        # Fragment 2: "2464" → source[1] = "2464.pdf"
        answer = "Текст [Фрагмент 2: 2464, п. 3]."
        rate = compute_citation_doc_match(answer, _PASSAGES)
        assert rate == pytest.approx(1.0)

    def test_doc_mismatch(self):
        # Fragment 1: "ПБ" → source[0] = "Трудовой кодекс РФ..." — no match
        answer = "Текст [Фрагмент 1: ПБ, п. 1]."
        rate = compute_citation_doc_match(answer, _PASSAGES)
        assert rate == 0.0

    def test_no_citations(self):
        assert compute_citation_doc_match(_ANSWER_NO_CITATIONS, _PASSAGES) == 0.0

    def test_out_of_range_citation_skipped(self):
        answer = "Текст [Фрагмент 99: Doc, п. 1]."
        rate = compute_citation_doc_match(answer, _PASSAGES)
        assert rate == 0.0
