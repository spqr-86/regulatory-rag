"""Reasoning chains: AppliedConclusion -> (fact, requirement, conclusion) (spec §15)."""

from __future__ import annotations

import pytest

from src.department_qa.contract import AppliedConclusion, Basis, Evidence, ObjectFact
from src.department_qa.service import DepartmentResponse
from src.department_qa.view import build_reasoning_chains

EVIDENCE = [
    Evidence(
        id="obj_s7",
        level="object",
        text="Людей в зоне: 24",
        source="list.md",
        title="Лист объекта",
        locator="7 Планировка и эвакуация",
    ),
    Evidence(
        id="obj_f_people_in_object_zone",
        level="object",
        text="Людей в зоне объекта: 24",
        source="list.md",
        title="Лист объекта",
        locator="Типизированное поле: Людей в зоне объекта",
        field_state="known",
    ),
    Evidence(
        id="ext_001",
        level="external",
        text="п. 5 Правил",
        source="ppr.pdf",
        title="ПП № 1479",
        locator="п. 5",
    ),
]


def _response(**kw) -> DepartmentResponse:
    base = dict(status="answered", trace_id="t", evidence=EVIDENCE)
    return DepartmentResponse(**{**base, **kw})


@pytest.mark.unit
def test_chain_from_object_fact_and_norm_basis():
    response = _response(
        object_facts=[
            ObjectFact(
                statement="24 сотрудника",
                evidence_ids=["obj_f_people_in_object_zone"],
            )
        ],
        external_basis=[
            Basis(statement="Проверка не реже раза в год", evidence_ids=["ext_001"])
        ],
        applied_conclusions=[
            AppliedConclusion(
                statement="Требуется ежегодная проверка",
                evidence_ids=["obj_f_people_in_object_zone", "ext_001"],
            )
        ],
    )
    chains = build_reasoning_chains(response)
    assert len(chains) == 1
    assert chains[0].fact == "24 сотрудника"
    assert chains[0].requirement == "Проверка не реже раза в год"
    assert chains[0].conclusion == "Требуется ежегодная проверка"


@pytest.mark.unit
def test_fact_falls_back_to_evidence_text_when_absent_from_object_facts():
    response = _response(
        applied_conclusions=[
            AppliedConclusion(statement="Вывод", evidence_ids=["obj_s7", "ext_001"])
        ],
    )
    assert build_reasoning_chains(response)[0].fact == "Людей в зоне: 24"


@pytest.mark.unit
def test_requirement_falls_back_to_evidence_text_without_norm_basis():
    response = _response(
        applied_conclusions=[
            AppliedConclusion(
                statement="Вывод",
                evidence_ids=["obj_f_people_in_object_zone", "ext_001"],
            )
        ],
    )
    assert build_reasoning_chains(response)[0].requirement == "п. 5 Правил"


@pytest.mark.unit
def test_no_applied_conclusions_means_no_chains():
    assert build_reasoning_chains(_response()) == []


@pytest.mark.unit
def test_multiple_conclusions_keep_order():
    response = _response(
        applied_conclusions=[
            AppliedConclusion(statement="Первый", evidence_ids=["obj_s7", "ext_001"]),
            AppliedConclusion(
                statement="Второй",
                evidence_ids=["obj_f_people_in_object_zone", "ext_001"],
            ),
        ],
    )
    chains = build_reasoning_chains(response)
    assert [c.conclusion for c in chains] == ["Первый", "Второй"]
    assert chains[0].fact == "Людей в зоне: 24"
    assert chains[1].fact == "Людей в зоне объекта: 24"
