"""Tests for the department Q&A flow: two scoped searches → prompt → checked answer."""

from __future__ import annotations

import pytest

from src.department_qa.contract import Basis, ModelAnswer
from src.department_qa.service import answer_question
from src.v7.scope_filter import build_scope_filters


def _passage(text, source, **meta):
    return {"text": text, "score": 0.7, "metadata": {"source": source, **meta}}


EXT_HITS = [_passage("Огнетушители осматривают по паспорту.", "ppr.pdf", title="ППР")]
INT_HITS = [
    _passage("Осмотр раз в квартал.", "pril3.md", title="Инструкция", chunk_id=4)
]


class FakeSearch:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, query, filters=None, top_k=8):
        self.calls.append(filters)
        if self.fail:
            raise RuntimeError("chroma down")
        return EXT_HITS if filters["source_type"] == "external" else INT_HITS


def _model(answer: ModelAnswer):
    prompts = []

    def call(prompt: str) -> ModelAnswer:
        prompts.append(prompt)
        return answer

    call.prompts = prompts
    return call


GOOD = ModelAnswer(
    answer="Осмотр ежеквартально.",
    external_basis=[Basis(statement="по паспорту", evidence_ids=["ext_001"])],
    internal_basis=[Basis(statement="раз в квартал", evidence_ids=["int_001"])],
)


@pytest.mark.unit
def test_two_scoped_searches_and_answered_response():
    search, model = FakeSearch(), _model(GOOD)
    result = answer_question("Как часто?", "unit_1", search, model, snapshot_id="s1")

    assert search.calls == list(build_scope_filters("unit_1"))
    assert "[ext_001]" in model.prompts[0] and "[int_001]" in model.prompts[0]
    assert result.status == "answered"
    assert result.snapshot_id == "s1"
    assert [e.id for e in result.evidence] == ["ext_001", "int_001"]
    assert result.evidence[1].text == "Осмотр раз в квартал."
    assert result.evidence[1].chunk_id == 4


@pytest.mark.unit
def test_retrieval_error_is_failed_not_empty():
    model = _model(GOOD)
    result = answer_question("q", "unit_1", FakeSearch(fail=True), model)
    assert (result.status, result.reason_codes) == ("failed", ["retrieval_failed"])
    assert model.prompts == []


@pytest.mark.unit
def test_generation_error_is_failed_without_passages_as_answer():
    def broken(prompt):
        raise ValueError("schema validation failed")

    result = answer_question("q", "unit_1", FakeSearch(), broken)
    assert (result.status, result.reason_codes) == ("failed", ["generation_failed"])
    assert result.answer == ""


@pytest.mark.unit
def test_citation_invalid_hides_basis():
    bad = GOOD.model_copy(
        update={"internal_basis": [Basis(statement="x", evidence_ids=["int_999"])]}
    )
    result = answer_question("q", "unit_1", FakeSearch(), _model(bad))
    assert (result.status, result.reason_codes) == ("failed", ["citation_invalid"])
    assert result.external_basis == [] and result.internal_basis == []
