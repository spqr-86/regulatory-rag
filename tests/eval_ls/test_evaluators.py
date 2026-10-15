"""Unit tests for eval/ls/evaluators.py — pure functions."""

from __future__ import annotations

import pytest

from eval.ls.evaluators import (
    doc_cited,
    doc_number,
    forbidden,
    oos_handled,
    retrieval_hit,
    summarize,
    summarize_cost,
)

pytestmark = pytest.mark.unit

IN_SCOPE = {
    "ground_truth": "не менее 2 смен",
    "relevant_chunk_ids": ["2464.pdf#107", "782н.pdf#431"],
    "must_not_contain": "",
    "forbidden_claims": "",
    "oos_type": "",
}
OOS = {**IN_SCOPE, "relevant_chunk_ids": [], "oos_type": "out_of_scope"}
FALSE_PREMISE = {**OOS, "oos_type": "false_premise"}


def out(answer="", route="generate", chunks=()):
    return {"answer": answer, "route_decision": route, "chunk_ids": list(chunks)}


@pytest.mark.parametrize(
    "cid, num",
    [("2464.pdf#107", "2464"), ("782н.pdf#431", "782н"), ("223н.pdf#298", "223н")],
)
def test_doc_number(cid, num):
    assert doc_number(cid) == num


class TestRetrievalHit:
    def test_hit_reports_first_rank(self):
        r = retrieval_hit(out(chunks=["x#1", "782н.pdf#431", "2464.pdf#107"]), IN_SCOPE)
        assert r == {"key": "retrieval_hit", "score": 1, "comment": "rank=2"}

    def test_miss(self):
        r = retrieval_hit(out(chunks=["x#1"]), IN_SCOPE)
        assert r["score"] == 0 and r["comment"] == "miss"

    def test_empty_answer_no_chunks_is_miss(self):
        assert retrieval_hit(out(route=None), IN_SCOPE)["score"] == 0

    def test_not_applicable_to_oos(self):
        assert retrieval_hit(out(chunks=["2464.pdf#107"]), OOS)["score"] is None


class TestDocCited:
    def test_number_in_answer(self):
        r = doc_cited(out("Согласно п. 53 Правил № 2464 ..."), IN_SCOPE)
        assert r["score"] == 1

    def test_case_insensitive_letter(self):
        assert doc_cited(out("по приказу № 782Н"), IN_SCOPE)["score"] == 1

    def test_not_cited(self):
        assert doc_cited(out("Стажировка не менее 2 смен."), IN_SCOPE)["score"] == 0

    def test_empty_answer(self):
        assert doc_cited(out(""), IN_SCOPE)["score"] == 0

    def test_number_inside_other_number_does_not_count(self):
        assert doc_cited(out("пункт 24640"), IN_SCOPE)["score"] == 0

    def test_not_applicable_to_oos(self):
        assert doc_cited(out("№ 2464"), OOS)["score"] is None


class TestForbidden:
    def test_empty_pattern_not_applicable(self):
        assert forbidden(out("что угодно"), IN_SCOPE)["score"] is None

    def test_pattern_found_is_zero(self):
        ref = {**IN_SCOPE, "must_not_contain": "раз в год|ежегодно"}
        assert forbidden(out("Проводится Ежегодно."), ref)["score"] == 0

    def test_pattern_absent_is_one(self):
        ref = {**IN_SCOPE, "must_not_contain": "раз в год|ежегодно"}
        assert forbidden(out("Раз в три года."), ref)["score"] == 1


class TestOosHandled:
    def test_abstain_route(self):
        assert oos_handled(out("какой-то текст", route="abstain"), OOS)["score"] == 1

    def test_empty_answer_from_domain_gate(self):
        assert oos_handled(out("", route=None), OOS)["score"] == 1

    def test_refusal_text(self):
        assert oos_handled(out("Не могу ответить: вопрос вне темы."), OOS)["score"] == 1

    def test_answered_oos_is_zero(self):
        assert oos_handled(out("Варите свёклу 40 минут."), OOS)["score"] == 0

    def test_false_premise_refuted(self):
        r = oos_handled(out("Это неверно: не реже раза в 3 года."), FALSE_PREMISE)
        assert r["score"] == 1

    def test_not_applicable_to_in_scope(self):
        assert oos_handled(out("ответ"), IN_SCOPE)["score"] is None


def test_summarize_rates_and_mrr():
    rows = [
        {"retrieval_hit": {"score": 1, "comment": "rank=1"}, "doc_cited": {"score": 1}},
        {"retrieval_hit": {"score": 1, "comment": "rank=4"}, "doc_cited": {"score": 0}},
        {
            "retrieval_hit": {"score": 0, "comment": "miss"},
            "doc_cited": {"score": None},
        },
        {"retrieval_hit": {"score": None, "comment": "oos"}},
    ]
    s = summarize(rows)
    assert s["retrieval_hit"] == pytest.approx(2 / 3)
    assert s["doc_cited"] == pytest.approx(1 / 2)
    assert s["mrr"] == pytest.approx((1 + 0.25 + 0) / 3)
    assert s["n"]["retrieval_hit"] == 3


def test_summarize_cost_total_mean_and_unpriced():
    outputs = [
        {"cost_usd": 0.01, "unpriced_models": []},
        {"cost_usd": 0.03, "unpriced_models": ["a/b"]},
        {},  # failed example: no outputs
    ]
    s = summarize_cost(outputs)
    assert s["total_usd"] == pytest.approx(0.04)
    assert s["mean_usd"] == pytest.approx(0.02)
    assert s["n"] == 2
    assert s["unpriced_models"] == ["a/b"]


def test_summarize_cost_empty():
    s = summarize_cost([])
    assert s["total_usd"] == 0 and s["mean_usd"] is None
