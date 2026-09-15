"""Tests for the department answer contract: citation check and status rule (spec §8, §9)."""

from __future__ import annotations

import pytest

from src.department_qa.contract import (
    Basis,
    Evidence,
    ModelAnswer,
    check_citations,
    decide,
)

EVIDENCE = {
    "ext_001": Evidence(
        id="ext_001", level="external", text="п. 60 ППР", source="ppr.pdf"
    ),
    "int_001": Evidence(
        id="int_001", level="internal", text="осмотр раз в квартал", source="pril3.md"
    ),
}


def _answer(ext=("ext_001",), internal=("int_001",), **kw) -> ModelAnswer:
    return ModelAnswer(
        answer="черновик",
        external_basis=(
            [Basis(statement="закон", evidence_ids=list(ext))] if ext else []
        ),
        internal_basis=(
            [Basis(statement="ЛНА", evidence_ids=list(internal))] if internal else []
        ),
        **kw,
    )


@pytest.mark.unit
def test_valid_citations_pass():
    assert check_citations(_answer(), EVIDENCE) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer, problem",
    [
        (_answer(ext=("ext_404",)), "unknown"),
        (_answer(ext=("int_001",)), "level"),
        (_answer(internal=("ext_001",)), "level"),
    ],
    ids=["unknown-id", "internal-cited-as-law", "law-cited-as-lna"],
)
def test_invalid_citations_reported(answer, problem):
    problems = check_citations(answer, EVIDENCE)
    assert problems and problem in problems[0]


@pytest.mark.unit
def test_basis_without_evidence_ids_is_invalid():
    answer = ModelAnswer(
        answer="x", external_basis=[Basis(statement="закон", evidence_ids=[])]
    )
    assert check_citations(answer, EVIDENCE)


@pytest.mark.unit
def test_both_levels_answered():
    assert decide(_answer(), EVIDENCE) == ("answered", [])


@pytest.mark.unit
def test_invalid_citation_fails_without_retry():
    assert decide(_answer(ext=("ext_404",)), EVIDENCE) == (
        "failed",
        ["citation_invalid"],
    )


@pytest.mark.unit
def test_missing_internal_basis_needs_review():
    assert decide(_answer(internal=()), EVIDENCE) == (
        "needs_review",
        ["internal_evidence_missing"],
    )


@pytest.mark.unit
def test_missing_external_basis_needs_review():
    assert decide(_answer(ext=()), EVIDENCE) == (
        "needs_review",
        ["external_evidence_missing"],
    )


@pytest.mark.unit
def test_mismatch_with_both_sides_needs_review():
    assert decide(_answer(possible_mismatch=True), EVIDENCE) == (
        "needs_review",
        ["possible_mismatch"],
    )


@pytest.mark.unit
def test_mismatch_without_both_sides_is_not_reported():
    # Spec §7: possible_mismatch only with concrete grounds on both sides.
    status, reasons = decide(_answer(internal=(), possible_mismatch=True), EVIDENCE)
    assert "possible_mismatch" not in reasons
    assert status == "needs_review"


@pytest.mark.unit
def test_clarifying_questions_need_context():
    answer = _answer(ext=(), internal=(), clarifying_questions=["Какое подразделение?"])
    assert decide(answer, EVIDENCE) == ("needs_context", ["applicability_unclear"])


@pytest.mark.unit
def test_out_of_scope_wins():
    assert decide(_answer(ext=(), internal=(), out_of_scope=True), EVIDENCE) == (
        "out_of_scope",
        [],
    )
