"""Tests for the department answer view model: what the screen shows per status (spec §10)."""

from __future__ import annotations

from datetime import date

import pytest

from src.department_qa.contract import Basis, Evidence, ObjectFact
from src.department_qa.service import DepartmentResponse
from src.department_qa.view import (
    ANSWERED_BANNER,
    basis_cards,
    basis_lines,
    clarification_fallback,
    evidence_cards,
    profile_caption,
    status_banner,
    technical_details,
    unit_name,
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
        ("needs_context", ["applied_on_unknown_field"], "info", "неизвестное значение"),
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
def test_basis_cards_resolve_statement_with_title_and_locator():
    response = _response(
        external_basis=[Basis(statement="Осмотр по паспорту", evidence_ids=["ext_001"])]
    )
    cards = basis_cards(response.external_basis, response.evidence)
    assert [(c.statement, c.title, c.locator) for c in cards] == [
        ("Осмотр по паспорту", "ППР № 1479", "п. 60")
    ]


@pytest.mark.unit
def test_basis_cards_keep_statement_when_source_is_not_in_evidence():
    response = _response(
        internal_basis=[Basis(statement="Внутреннее правило", evidence_ids=["int_x"])]
    )
    cards = basis_cards(response.internal_basis, response.evidence)
    assert cards == [basis_cards(response.internal_basis, [])[0]]
    assert cards[0].statement == "Внутреннее правило"
    assert cards[0].title == ""
    assert cards[0].locator is None


@pytest.mark.unit
def test_basis_cards_one_card_per_object_fact_preferring_typed_field():
    evidence = [
        Evidence(
            id="obj_s2",
            level="object",
            text="Персонал: 24",
            source="sheet.md",
            title="Лист объекта",
            locator="раздел 2",
        ),
        Evidence(
            id="obj_f_headcount",
            level="object",
            text="24",
            source="sheet.md",
            title="Лист объекта",
            locator="Численность персонала",
        ),
    ]
    facts = [
        ObjectFact(
            statement="24 сотрудника", evidence_ids=["obj_s2", "obj_f_headcount"]
        ),
        ObjectFact(statement="Раздел без поля", evidence_ids=["obj_s2"]),
    ]
    cards = basis_cards(facts, evidence)
    assert [(c.statement, c.locator) for c in cards] == [
        ("24 сотрудника", "Численность персонала"),
        ("Раздел без поля", "раздел 2"),
    ]


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


@pytest.mark.unit
@pytest.mark.parametrize(
    "title, unit_id, expected",
    [
        (
            "Лист особенностей объекта защиты — офис (демонстрационные данные)",
            "unit_office",
            "Офис",
        ),
        (
            "Лист особенностей объекта защиты — диспетчерский центр (демонстрационные данные)",
            "unit_dispatch",
            "Диспетчерский центр",
        ),
        ("", "unit_x", "unit_x"),
    ],
)
def test_unit_name_is_short_human_name_from_sheet_title(title, unit_id, expected):
    assert unit_name(title, unit_id) == expected


@pytest.mark.unit
def test_evidence_cards_keep_type_locator_and_retrieval_score():
    external = Evidence(
        id="ext_001",
        level="external",
        text="raw chunk",
        source="ppr.pdf",
        title="ППР № 1479",
        locator="п. 60",
        retrieval_score=0.42,
    )
    obj = Evidence(
        id="obj_s7",
        level="object",
        text="raw object",
        source="list.md",
        title="Лист объекта",
    )
    cards = evidence_cards(_response(evidence=[external, obj]))

    assert [c.id for c in cards] == ["ext_001", "obj_s7"]
    assert cards[0].source_type == "external"
    assert cards[0].locator == "п. 60"
    assert cards[0].excerpt == "raw chunk"
    assert cards[0].retrieval_score == 0.42
    assert cards[1].source_type == "object"
    assert cards[1].retrieval_score is None


@pytest.mark.unit
def test_evidence_card_falls_back_to_source_as_title():
    ev = Evidence(id="int_001", level="internal", text="x", source="inst.md")
    assert evidence_cards(_response(evidence=[ev]))[0].title == "inst.md"


@pytest.mark.unit
def test_clarification_fallback_splits_norms_facts_and_questions():
    response = _response(
        status="needs_context",
        reason_codes=["applicability_unclear"],
        external_basis=[Basis(statement="Порог 50 человек", evidence_ids=["ext_001"])],
        internal_basis=[
            Basis(statement="Осмотр по инструкции", evidence_ids=["int_001"])
        ],
        object_facts=[
            ObjectFact(statement="24 сотрудника", evidence_ids=["obj_f_missing"])
        ],
        clarifying_questions=["Какова численность людей в здании?"],
    )
    fallback = clarification_fallback(response)

    assert [(c.statement, c.title, c.locator) for c in fallback.requirements] == [
        ("Порог 50 человек", "ППР № 1479", "п. 60"),
        ("Осмотр по инструкции", "Инструкция по содержанию ТС ППЗ", None),
    ]
    assert [c.statement for c in fallback.facts] == ["24 сотрудника"]
    assert fallback.questions == ["Какова численность людей в здании?"]


@pytest.mark.unit
def test_clarification_fallback_empty_when_nothing_to_show():
    fallback = clarification_fallback(_response(status="needs_context"))
    assert fallback.requirements == []
    assert fallback.facts == []
    assert fallback.questions == []


@pytest.mark.unit
def test_technical_details_list_known_values_and_omit_unknown():
    rows = dict(
        technical_details(_response(), model_name="deepseek/v4.1-flash", latency_s=3.84)
    )
    assert rows["LLM"] == "deepseek/v4.1-flash"
    assert rows["Retrieval"] == "dense + BM25 → RRF"
    assert rows["Used as evidence"] == "2 фрагмента"
    assert rows["Latency"] == "3.8 с"
    assert rows["Trace ID"] == "t"
    assert "Route" not in rows and "Cost" not in rows


@pytest.mark.unit
def test_technical_details_omit_model_and_latency_when_unknown():
    rows = dict(technical_details(_response()))
    assert "LLM" not in rows and "Latency" not in rows
    assert rows["Trace ID"] == "t"
