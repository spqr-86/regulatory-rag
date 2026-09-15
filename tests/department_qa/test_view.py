"""Tests for the department answer view model: what the screen shows per status (spec §10)."""

from __future__ import annotations

import pytest

from src.department_qa.contract import Basis, Evidence
from src.department_qa.service import DepartmentResponse
from src.department_qa.view import basis_lines, status_banner

EVIDENCE = [
    Evidence(
        id="ext_001",
        level="external",
        text="п. 60",
        source="ppr.pdf",
        title="ППР № 1479",
        locator="п. 60",
    ),
    Evidence(
        id="int_001",
        level="internal",
        text="раз в квартал",
        source="pril3.md",
        title="Инструкция по содержанию ТС ППЗ",
    ),
]


def _response(**kw) -> DepartmentResponse:
    base = dict(status="answered", trace_id="t", evidence=EVIDENCE)
    return DepartmentResponse(**{**base, **kw})


@pytest.mark.unit
@pytest.mark.parametrize(
    "status, reasons, kind, phrase",
    [
        ("answered", [], "success", "Ответ"),
        ("needs_review", ["possible_mismatch"], "warning", "расхожд"),
        ("needs_review", ["internal_evidence_missing"], "warning", "ЛНА"),
        ("needs_context", ["applicability_unclear"], "info", "Уточн"),
        ("out_of_scope", [], "info", "вне"),
        ("failed", ["citation_invalid"], "error", "ссылк"),
        ("failed", ["retrieval_failed"], "error", "поиск"),
    ],
)
def test_status_banner(status, reasons, kind, phrase):
    got_kind, text = status_banner(_response(status=status, reason_codes=reasons))
    assert got_kind == kind
    assert phrase.lower() in text.lower()


@pytest.mark.unit
def test_basis_lines_resolve_titles_from_stored_evidence():
    response = _response(
        external_basis=[Basis(statement="Осмотр по паспорту", evidence_ids=["ext_001"])]
    )
    lines = basis_lines(response.external_basis, response.evidence)
    assert lines == ["Осмотр по паспорту — [ext_001] ППР № 1479, п. 60"]


@pytest.mark.unit
def test_basis_lines_empty_level_says_not_found():
    assert basis_lines([], EVIDENCE) == []
