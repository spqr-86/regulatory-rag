"""Root screen (app.py): portfolio shell and stack errors shown, not tracebacks.

Spec streamlit-portfolio-demo §5–§10: initial screen only — header, unit selector,
compact profile, example buttons, one input. Backend is stubbed; no API calls.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config.settings import settings
from src.department_qa import wiring
from src.department_qa.object_profile import (
    ObjectProfile,
    ObjectProfileError,
    TypedObjectFields,
)
from src.department_qa.wiring import DepartmentStack, ModeConfig
from src.indexing.manifest import Manifest

PAGE = str(Path(__file__).resolve().parents[2] / "app.py")


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    """AppTest shares the process-wide resource cache across tests."""
    st.cache_resource.clear()
    st.cache_data.clear()
    yield


OFFICE_TITLE = "Лист особенностей объекта защиты — офис (демонстрационные данные)"
DISPATCH_TITLE = (
    "Лист особенностей объекта защиты — диспетчерский центр (демонстрационные данные)"
)


def _profile(unit_id: str) -> ObjectProfile:
    return ObjectProfile(
        unit_id=unit_id,
        document_id=f"{unit_id}_list",
        source=f"{unit_id}_list.md",
        content_sha256="0" * 64,
        title=OFFICE_TITLE if unit_id == "unit_office" else DISPATCH_TITLE,
        as_of_date=None,
        sections={},
        typed_fields=TypedObjectFields(
            people_in_object_zone=24,
            people_on_floor_total="unknown",
            people_in_building_total="unknown",
            permanent_workplaces_on_floor="unknown",
            evacuation_plan_present=True,
            room_categories="unknown",
            aupt_present=False,
            extinguishers_total="unknown",
            outside_ladder_last_test_date="unknown",
        ),
    )


def _fake_stack() -> DepartmentStack:
    manifest = Manifest(
        snapshot_id="pb_demo_test",
        organization_id="org_test",
        checked_at="2026-09-15",
        documents={
            "office.md": {"unit_id": "unit_office", "title": OFFICE_TITLE},
            "dispatch.md": {"unit_id": "unit_dispatch", "title": DISPATCH_TITLE},
        },
        object_profiles={},
    )
    config = ModeConfig(
        mode="v2",
        chroma_db_path="./chroma_db_dept_v2",
        collection="department_demo_v2",
        prompt_version="v4",
        schema=object,
        profiles={"unit_office": _profile("unit_office")},
    )
    return DepartmentStack(
        config=config,
        manifest=manifest,
        search_fn=lambda *a, **k: [],
        model_fn=lambda prompt: None,
        verifier_fn=lambda prompt: None,
        store=object(),
    )


def _manifest_path(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("documents: {}\n", encoding="utf-8")
    return manifest


def test_initial_screen_shows_shell_and_no_debug(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", str(_manifest_path(tmp_path)))
    monkeypatch.setattr(wiring, "build_department_stack", _fake_stack)

    at = AppTest.from_file(PAGE, default_timeout=30).run()

    assert not at.exception
    assert at.title[0].value == "Regulatory Compliance Assistant"

    assert len(at.selectbox) == 1
    assert at.selectbox[0].options == [
        "Без подразделения — общие требования",
        "Диспетчерский центр",
        "Офис",
    ]
    assert at.selectbox[0].value == "unit_dispatch"

    body = " ".join(
        [m.value for m in at.markdown]
        + [h.value for h in at.subheader]
        + [c.value for c in at.caption]
    )
    assert "Контекст запроса" in body
    assert "Что хотите проверить?" in body
    labels = [b.label for b in at.button]
    for example in (
        "Какие требования применимы?",
        "Есть ли расхождения?",
        "Как часто требуется проверка?",
        "Спросить",
    ):
        assert example in labels
    assert len(at.text_input) == 1

    for banned in ("Simple:", "Complex:", "CHROMA_DB_PATH", "Prompt version"):
        assert banned not in body

    at.selectbox[0].select_index(2).run()
    assert not at.exception
    body = " ".join(
        [m.value for m in at.markdown]
        + [h.value for h in at.subheader]
        + [c.value for c in at.caption]
    )
    assert "24 сотрудника" in body
    assert "АУПТ отсутствует" in body
    assert "Данные: дата заполнения не указана" in body


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("DEPARTMENT_QA_MODE=v2 needs CHROMA_DB_PATH=./chroma_db_dept_v2"),
        ObjectProfileError("unit_office: sheet has no sections"),
    ],
)
def test_page_shows_stack_error_instead_of_traceback(monkeypatch, tmp_path, exc):
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", str(_manifest_path(tmp_path)))

    def _raise():
        raise exc

    monkeypatch.setattr(wiring, "build_department_stack", _raise)

    at = AppTest.from_file(PAGE, default_timeout=30).run()

    assert not at.exception
    assert len(at.error) == 1
    assert str(exc) in at.error[0].value
