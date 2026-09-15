"""The department prompt renders only through its typed contract (rule: one model per prompt)."""

from __future__ import annotations

import pytest
from jinja2 import Environment, FileSystemLoader, meta

from src.department_qa.contract import Evidence, PromptVars
from src.infra.prompt_manager import PromptManager


@pytest.mark.unit
def test_template_variables_match_contract_fields():
    pm = PromptManager()
    path = pm.registry["department_answer"]["versions"]["v1"]
    env = Environment(loader=FileSystemLoader(pm.prompts_dir))
    source = env.loader.get_source(env, path)[0]
    assert meta.find_undeclared_variables(env.parse(source)) == set(
        PromptVars.model_fields
    )


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
