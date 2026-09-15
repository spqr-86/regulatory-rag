"""The department prompts render only through their typed contract (rule: one model per prompt)."""

from __future__ import annotations

import pytest
from jinja2 import Environment, FileSystemLoader, meta

from src.department_qa.contract import Evidence, ObjectSection, PromptVars
from src.infra.prompt_manager import PromptManager

OBJECT_FIELDS = {"object_label", "object_sections"}


def _template_vars(version: str) -> set[str]:
    pm = PromptManager()
    path = pm.registry["department_answer"]["versions"][version]
    env = Environment(loader=FileSystemLoader(pm.prompts_dir))
    source = env.loader.get_source(env, path)[0]
    return meta.find_undeclared_variables(env.parse(source))


@pytest.mark.unit
def test_v1_variables_are_contract_without_object_fields():
    assert _template_vars("v1") == set(PromptVars.model_fields) - OBJECT_FIELDS


@pytest.mark.unit
def test_v2_variables_match_contract_fields():
    assert _template_vars("v2") == set(PromptVars.model_fields)


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
def test_v2_renders_object_block_with_empty_and_missing_markers():
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
        "department_answer", version="v2", **vars_.model_dump()
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
def test_explicit_version_unknown_raises():
    with pytest.raises(KeyError):
        PromptManager().render("department_answer", version="v9", question="q")
