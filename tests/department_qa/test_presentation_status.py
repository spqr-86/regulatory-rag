"""Presentation status: backend status + reason_codes -> one UI status (spec §13, §32.5)."""

from __future__ import annotations

import pytest

from src.department_qa.service import DepartmentResponse
from src.department_qa.view import presentation_status


def _response(status: str, reasons: list[str] | None = None) -> DepartmentResponse:
    return DepartmentResponse(
        status=status, reason_codes=list(reasons or []), trace_id="t"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "status, reasons, expected",
    [
        ("answered", [], "sufficient"),
        ("needs_context", ["applicability_unclear"], "clarification_required"),
        ("needs_context", ["applied_on_unknown_field"], "clarification_required"),
        ("needs_context", ["verification_missing_facts"], "clarification_required"),
        ("out_of_scope", [], "out_of_scope"),
        ("failed", ["citation_invalid"], "failed"),
        ("failed", ["retrieval_failed"], "failed"),
        ("failed", ["generation_failed"], "failed"),
        ("failed", ["profile_mismatch"], "failed"),
        ("needs_review", ["internal_evidence_missing"], "insufficient_evidence"),
        ("needs_review", ["external_evidence_missing"], "insufficient_evidence"),
        ("needs_review", ["applied_without_norm"], "insufficient_evidence"),
        ("needs_review", ["object_fact_normative"], "insufficient_evidence"),
        ("needs_review", ["object_profile_undated"], "insufficient_evidence"),
        ("needs_review", ["verification_failed"], "insufficient_evidence"),
        ("needs_review", ["possible_mismatch"], "conflict"),
        ("needs_review", ["verification_contradiction"], "conflict"),
    ],
)
def test_backend_status_maps_to_ui_status(status, reasons, expected):
    assert presentation_status(_response(status, reasons)).code == expected


@pytest.mark.unit
def test_conflict_wins_over_missing_evidence():
    result = presentation_status(
        _response("needs_review", ["internal_evidence_missing", "possible_mismatch"])
    )
    assert result.code == "conflict"


@pytest.mark.unit
def test_unknown_review_reason_is_insufficient_not_a_crash():
    result = presentation_status(_response("needs_review", ["a_new_reason_code"]))
    assert result.code == "insufficient_evidence"


@pytest.mark.unit
def test_insufficient_names_the_missing_source():
    detail = presentation_status(
        _response("needs_review", ["internal_evidence_missing"])
    ).detail
    assert "локальный акт" in detail.lower()


@pytest.mark.unit
def test_titles_and_details_never_expose_reason_codes():
    for status, reasons in [
        ("answered", []),
        ("needs_context", ["applicability_unclear"]),
        ("needs_review", ["possible_mismatch"]),
        ("out_of_scope", []),
        ("failed", ["retrieval_failed"]),
    ]:
        result = presentation_status(_response(status, reasons))
        text = f"{result.title} {result.detail}"
        for code in (
            "needs_review",
            "needs_context",
            "verification_failed",
            "internal_evidence_missing",
            "applicability_unclear",
            "retrieval_failed",
        ):
            assert code not in text
