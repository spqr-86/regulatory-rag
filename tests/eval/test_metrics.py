"""Unit tests for eval/metrics.py — deterministic completeness and abstain metrics."""

from __future__ import annotations

import pytest

from eval.metrics import (
    compute_completeness,
    compute_abstain_rate,
    compute_false_abstain_rate,
    compute_correct_abstain_rate,
    compute_retrieval_stats,
    extract_key_phrases,
)

# ── extract_key_phrases ────────────────────────────────────────────────────────


class TestExtractKeyPhrases:
    def test_returns_set(self):
        phrases = extract_key_phrases("повторный инструктаж раз в 6 месяцев")
        assert isinstance(phrases, set)

    def test_non_empty_for_meaningful_text(self):
        phrases = extract_key_phrases("инструктаж по охране труда проводится ежегодно")
        assert len(phrases) > 0

    def test_empty_string(self):
        phrases = extract_key_phrases("")
        assert phrases == set()

    def test_lemmatizes_russian(self):
        # "инструктажи" → lemma "инструктаж", "проводятся" → "проводить" (lemmatization)
        phrases = extract_key_phrases("инструктажи проводятся ежегодно")
        assert "инструктаж" in phrases

    def test_filters_short_words(self):
        phrases = extract_key_phrases("по за на от до")
        assert len(phrases) == 0


# ── compute_completeness ───────────────────────────────────────────────────────


class TestComputeCompleteness:
    def test_perfect_match_returns_1(self):
        gt = "повторный инструктаж проводится раз в шесть месяцев"
        answer = "повторный инструктаж проводится не реже одного раза в шесть месяцев"
        score = compute_completeness(gt, answer)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_empty_answer_returns_0(self):
        gt = "инструктаж проводится ежегодно"
        score = compute_completeness(gt, "")
        assert score == 0.0

    def test_empty_ground_truth_returns_1(self):
        # No key phrases → nothing to check → completeness = 1.0
        score = compute_completeness("", "любой ответ")
        assert score == 1.0

    def test_partial_match(self):
        gt = "программа А охрана труда руководители специалисты комиссия"
        answer = "по программе А обучаются руководители"
        score = compute_completeness(gt, answer)
        assert 0.0 < score < 1.0

    def test_no_match_returns_0(self):
        gt = "противопожарные тренировки раз в полгода эвакуация"
        answer = "стажировка составляет не менее двух смен"
        score = compute_completeness(gt, answer)
        # Может быть не строго 0, но должна быть очень маленькой
        assert score < 0.3

    def test_score_is_float_between_0_and_1(self):
        gt = "обучение охрана труда программа Б шестнадцать часов"
        answer = "обучение по программе Б составляет не менее 16 часов"
        score = compute_completeness(gt, answer)
        assert 0.0 <= score <= 1.0

    def test_case_insensitive(self):
        gt = "Инструктаж Охрана Труда"
        answer = "инструктаж охрана труда"
        score = compute_completeness(gt, answer)
        assert score == pytest.approx(1.0, abs=0.05)


# ── compute_abstain_rate ───────────────────────────────────────────────────────


class TestComputeAbstainRate:
    def _make_results(self, answers: list[str]) -> list[dict]:
        return [{"answer": a} for a in answers]

    def test_all_answered(self):
        results = self._make_results(["ответ 1", "ответ 2", "ответ 3"])
        assert compute_abstain_rate(results) == 0.0

    def test_all_abstained(self):
        results = self._make_results(["", "", ""])
        assert compute_abstain_rate(results) == 1.0

    def test_half_abstained(self):
        results = self._make_results(["ответ", "", "ответ", ""])
        assert compute_abstain_rate(results) == pytest.approx(0.5)

    def test_empty_list(self):
        assert compute_abstain_rate([]) == 0.0

    def test_uses_abstained_flag_if_present(self):
        results = [
            {"answer": "ответ", "abstained": False},
            {"answer": "", "abstained": True},
        ]
        assert compute_abstain_rate(results) == pytest.approx(0.5)


# ── compute_false_abstain_rate ─────────────────────────────────────────────────


class TestComputeFalseAbstainRate:
    def test_no_domain_queries(self):
        results = [{"answer": "ответ", "is_oos": False}]
        assert compute_false_abstain_rate(results) == 0.0

    def test_false_abstain_detected(self):
        results = [
            {"answer": "", "is_oos": False},  # domain + abstained → false abstain
            {"answer": "ответ", "is_oos": False},
        ]
        rate = compute_false_abstain_rate(results)
        assert rate == pytest.approx(0.5)

    def test_oos_abstain_not_counted(self):
        results = [
            {"answer": "", "is_oos": True},  # OOS + abstained → correct, not false
            {"answer": "ответ", "is_oos": False},
        ]
        assert compute_false_abstain_rate(results) == 0.0

    def test_empty_list(self):
        assert compute_false_abstain_rate([]) == 0.0


# ── compute_correct_abstain_rate ───────────────────────────────────────────────


class TestComputeCorrectAbstainRate:
    def test_all_oos_abstained(self):
        results = [
            {"answer": "", "is_oos": True},
            {"answer": "", "is_oos": True},
        ]
        assert compute_correct_abstain_rate(results) == 1.0

    def test_oos_answered(self):
        results = [
            {"answer": "как приготовить борщ", "is_oos": True},
        ]
        assert compute_correct_abstain_rate(results) == 0.0

    def test_no_oos_queries(self):
        results = [{"answer": "ответ", "is_oos": False}]
        assert compute_correct_abstain_rate(results) == 1.0  # no OOS → treated as 100%

    def test_empty_list(self):
        assert compute_correct_abstain_rate([]) == 1.0


# ── compute_retrieval_stats ───────────────────────────────────────────────────


class TestComputeRetrievalStats:
    def _make_results(
        self, top_scores: list[float], passage_counts: list[int]
    ) -> list[dict]:
        return [
            {"top_score": ts, "passage_count": pc}
            for ts, pc in zip(top_scores, passage_counts)
        ]

    def test_avg_top_score(self):
        results = self._make_results([0.8, 0.6], [5, 3])
        stats = compute_retrieval_stats(results)
        assert stats["avg_top_score"] == pytest.approx(0.7, abs=0.01)

    def test_avg_passage_count(self):
        results = self._make_results([0.8, 0.6], [4, 6])
        stats = compute_retrieval_stats(results)
        assert stats["avg_passage_count"] == pytest.approx(5.0, abs=0.01)

    def test_empty_list(self):
        stats = compute_retrieval_stats([])
        assert stats["avg_top_score"] == 0.0
        assert stats["avg_passage_count"] == 0.0

    def test_missing_fields_handled(self):
        results = [{"other_field": "value"}]
        stats = compute_retrieval_stats(results)
        assert "avg_top_score" in stats
        assert "avg_passage_count" in stats
