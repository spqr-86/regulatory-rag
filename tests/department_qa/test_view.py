"""Tests for the department answer view model: what the screen shows per status (spec §10)."""

from __future__ import annotations

from datetime import date

import pytest

from src.department_qa.contract import Basis, Evidence, ObjectFact
from src.department_qa.service import DepartmentResponse
from src.department_qa.view import (
    ANSWERED_BANNER,
    basis_lines,
    profile_caption,
    status_banner,
)

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
        ("answered", [], "success", "Ссылки сверены"),
        ("needs_review", ["possible_mismatch"], "warning", "расхожд"),
        ("needs_review", ["internal_evidence_missing"], "warning", "ЛНА"),
        ("needs_review", ["applied_without_norm"], "warning", "без нормы"),
        ("needs_review", ["object_fact_normative"], "warning", "вывод о норме"),
        ("needs_review", ["object_profile_undated"], "warning", "дата заполнения"),
        ("needs_context", ["applicability_unclear"], "info", "Уточн"),
        ("out_of_scope", [], "info", "вне"),
        ("failed", ["citation_invalid"], "error", "ссылк"),
        ("failed", ["retrieval_failed"], "error", "поиск"),
        ("failed", ["profile_mismatch"], "error", "другому подразделению"),
    ],
)
def test_status_banner(status, reasons, kind, phrase):
    got_kind, text = status_banner(_response(status=status, reason_codes=reasons))
    assert got_kind == kind
    assert phrase.lower() in text.lower()


@pytest.mark.unit
def test_answered_banner_states_guarantee_boundary_verbatim():
    kind, text = status_banner(_response(status="answered"))
    assert kind == "success"
    assert (
        text
        == ANSWERED_BANNER
        == (
            "Ссылки сверены: законодательство и ЛНА. Смысл ответа не проверен специалистом."
        )
    )
    assert "подтвержд" not in text.lower()


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


@pytest.mark.unit
def test_object_fact_lines_resolve_section_locator():
    obj = Evidence(
        id="obj_s5",
        level="object",
        text="нет",
        source="list.md",
        title="Лист особенностей объекта защиты: офис",
        locator="5 Дежурный персонал",
    )
    lines = basis_lines(
        [
            ObjectFact(
                statement="Круглосуточного дежурства нет", evidence_ids=["obj_s5"]
            )
        ],
        [obj],
    )
    assert lines == [
        "Круглосуточного дежурства нет — [obj_s5] Лист особенностей объекта защиты: офис, 5 Дежурный персонал"
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "as_of, sha, expected",
    [
        (
            date(2026, 9, 1),
            "abcdef1234",
            "Сведения объекта на 01.09.2026 · лист abcdef12",
        ),
        (
            None,
            "abcdef1234",
            "Сведения объекта: дата заполнения не указана · лист abcdef12",
        ),
        (None, None, ""),
    ],
)
def test_profile_caption(as_of, sha, expected):
    assert (
        profile_caption(_response(profile_as_of_date=as_of, profile_sha256=sha))
        == expected
    )
