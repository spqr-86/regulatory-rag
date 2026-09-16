"""The department prompts render only through their typed contract (rule: one model per prompt)."""

from __future__ import annotations

import pytest
from jinja2 import Environment, FileSystemLoader, meta

from src.department_qa.contract import Evidence, ObjectSection, PromptVars
from src.infra.prompt_manager import PromptManager

OBJECT_FIELDS = {"object_label", "object_sections"}


def _template_source(version: str) -> str:
    pm = PromptManager()
    path = pm.registry["department_answer"]["versions"][version]
    env = Environment(loader=FileSystemLoader(pm.prompts_dir))
    return env.loader.get_source(env, path)[0]


def _template_vars(version: str) -> set[str]:
    env = Environment(loader=FileSystemLoader(PromptManager().prompts_dir))
    return meta.find_undeclared_variables(env.parse(_template_source(version)))


@pytest.mark.unit
def test_v1_variables_are_contract_without_object_fields():
    assert _template_vars("v1") == set(PromptVars.model_fields) - OBJECT_FIELDS


@pytest.mark.unit
@pytest.mark.parametrize("version", ["v2", "v3"])
def test_object_versions_variables_match_contract_fields(version):
    assert _template_vars(version) == set(PromptVars.model_fields)


@pytest.mark.unit
def test_active_version_stays_v1():
    assert PromptManager().registry["department_answer"]["active_version"] == "v1"


@pytest.mark.unit
def test_render_puts_evidence_ids_and_question_into_prompt():
    vars_ = PromptVars(
        question="Как часто осматривать огнетушители?",
        unit_label="Подразделение: unit_1",
        external_evidence=[
            Evidence(id="ext_001", level="external", text="п. 60", source="ppr.pdf")
        ],
        internal_evidence=[],
    )
    text = PromptManager().render("department_answer", **vars_.model_dump())
    assert "[ext_001]" in text
    assert "(не найдено)" in text
    assert "Как часто осматривать огнетушители?" in text
    assert "СВЕДЕНИЯ ОБ ОБЪЕКТЕ" not in text


@pytest.mark.unit
@pytest.mark.parametrize("version", ["v2", "v3"])
def test_object_versions_render_object_block_with_empty_and_missing_markers(version):
    vars_ = PromptVars(
        question="Кто принимает сигнал?",
        unit_label="Подразделение: unit_1",
        external_evidence=[],
        internal_evidence=[],
        object_label="Лист: тест. Дата заполнения: 01.09.2026.",
        object_sections=[
            ObjectSection(
                id="obj_s3",
                number=3,
                title="Системы противопожарной защиты объекта",
                text="АПС есть.",
                presence="present",
            ),
            ObjectSection(
                id="obj_s4",
                number=4,
                title="Первичные средства пожаротушения",
                presence="empty",
            ),
            ObjectSection(
                id="obj_s5", number=5, title="Дежурный персонал", presence="missing"
            ),
        ],
    )
    text = PromptManager().render(
        "department_answer", version=version, **vars_.model_dump()
    )
    assert "СВЕДЕНИЯ ОБ ОБЪЕКТЕ" in text
    assert "Дата заполнения: 01.09.2026" in text
    assert "[obj_s3] 3 Системы противопожарной защиты объекта\nАПС есть." in text
    assert (
        "4 Первичные средства пожаротушения (не заполнено) — ссылаться нельзя" in text
    )
    assert "5 Дежурный персонал (раздела нет в листе) — ссылаться нельзя" in text
    assert "[obj_s4]" not in text and "[obj_s5]" not in text


@pytest.mark.unit
def test_v3_requires_matching_the_fact_quantity_against_the_threshold():
    """q1 of the 2026-09-15 run: a plan-required conclusion drawn from 8 people in the
    office against a 50-per-building / 10-workplaces-per-floor threshold (design
    decision 10). v3 states the rule; v2 is kept as recorded for that run."""
    source = _template_source("v3")
    v2_source = _template_source("v2")
    for fragment in (
        "порог",
        "та же величина",
        "условно",
    ):
        assert fragment in source, fragment
    assert "та же величина" not in v2_source


@pytest.mark.unit
def test_explicit_version_unknown_raises():
    with pytest.raises(KeyError):
        PromptManager().render("department_answer", version="v9", question="q")
