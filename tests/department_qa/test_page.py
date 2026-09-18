"""Root screen (app.py): portfolio shell and stack errors shown, not tracebacks.

Spec streamlit-portfolio-demo §5–§18: initial screen and the structured result
screen (answer, reasoning, bases, sources, technical details). Backend is
stubbed; no API calls.
"""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config.settings import settings
from src.department_qa import wiring
from src.department_qa.contract import AppliedConclusion, Basis, Evidence, ObjectFact
from src.department_qa.object_profile import (
    ObjectProfile,
    ObjectProfileError,
    TypedObjectFields,
)
from src.department_qa.service import DepartmentResponse
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


EVIDENCE = [
    Evidence(
        id="ext_001",
        level="external",
        text="п. 60 ППР",
        source="ppr.pdf",
        title="ППР № 1479",
        locator="п. 60",
        retrieval_score=0.42,
    ),
    Evidence(
        id="int_001",
        level="internal",
        text="раз в год",
        source="inst.md",
        title="Инструкция по содержанию СПЗ",
        locator="п. 3.1",
    ),
    Evidence(
        id="obj_f_people_in_object_zone",
        level="object",
        text="Людей в зоне объекта: 24",
        source="office_list.md",
        title="Лист объекта",
        locator="Типизированное поле: Людей в зоне объекта",
        field_state="known",
    ),
]

ANSWER = "Плановая проверка проводится не реже одного раза в год."


def _answered_response() -> DepartmentResponse:
    return DepartmentResponse(
        status="answered",
        trace_id="t",
        answer=ANSWER,
        evidence=EVIDENCE,
        object_facts=[
            ObjectFact(
                statement="24 сотрудника",
                evidence_ids=["obj_f_people_in_object_zone"],
            )
        ],
        external_basis=[
            Basis(statement="Проверка не реже раза в год", evidence_ids=["ext_001"])
        ],
        internal_basis=[
            Basis(statement="Осмотр по инструкции", evidence_ids=["int_001"])
        ],
        applied_conclusions=[
            AppliedConclusion(
                statement="Требуется ежегодная проверка",
                evidence_ids=["obj_f_people_in_object_zone", "ext_001"],
            )
        ],
    )


def _low_confidence_response(status: str, reasons: list[str]) -> DepartmentResponse:
    return DepartmentResponse(
        status=status,
        reason_codes=reasons,
        trace_id="t",
        answer=ANSWER,
        evidence=EVIDENCE,
    )


def _submit(monkeypatch, tmp_path, response: DepartmentResponse) -> AppTest:
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", str(_manifest_path(tmp_path)))
    monkeypatch.setattr(wiring, "build_department_stack", _fake_stack)
    monkeypatch.setattr(
        "src.department_qa.service.answer_question", lambda *a, **k: response
    )

    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.text_input[0].set_value("Как часто требуется проверка?").run()
    [b for b in at.button if b.label == "Спросить"][0].click().run()
    return at


def test_answered_shows_result_screen(monkeypatch, tmp_path):
    at = _submit(monkeypatch, tmp_path, _answered_response())

    assert not at.exception
    subheaders = [h.value for h in at.subheader]
    for expected in (
        "Ваш вопрос",
        "Ответ",
        "Почему такой вывод",
        "На чём основан ответ",
    ):
        assert expected in subheaders

    body = " ".join(m.value for m in at.markdown)
    assert ANSWER in body
    assert "24 сотрудника" in body
    assert "Проверка не реже раза в год" in body
    assert "Требуется ежегодная проверка" in body
    assert "ППР № 1479" in body

    labels = [e.label for e in at.expander]
    assert "Источники, на которые опирается ответ (3)" in labels
    assert "Технические детали" in labels

    assert "Доказательств достаточно" in at.success[0].value


@pytest.mark.parametrize(
    "response",
    [
        _low_confidence_response("needs_review", ["internal_evidence_missing"]),
        _low_confidence_response("needs_context", ["applicability_unclear"]),
        _low_confidence_response("out_of_scope", []),
        _low_confidence_response("failed", ["retrieval_failed"]),
    ],
)
def test_low_confidence_never_shows_confident_answer(monkeypatch, tmp_path, response):
    at = _submit(monkeypatch, tmp_path, response)

    assert not at.exception
    subheaders = [h.value for h in at.subheader]
    for forbidden in ("Ответ", "Почему такой вывод", "На чём основан ответ"):
        assert forbidden not in subheaders

    body = " ".join(m.value for m in at.markdown)
    assert ANSWER not in body


def test_submit_shows_real_progress_stages(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", str(_manifest_path(tmp_path)))
    monkeypatch.setattr(wiring, "build_department_stack", _fake_stack)

    def fake_answer(*args, progress_fn=None, **kwargs):
        for stage in (
            "retrieval_started",
            "retrieval_completed",
            "generation_started",
            "generation_completed",
            "verification_started",
            "verification_completed",
        ):
            if progress_fn is not None:
                progress_fn(stage)
        return _answered_response()

    monkeypatch.setattr("src.department_qa.service.answer_question", fake_answer)

    at = AppTest.from_file(PAGE, default_timeout=30).run()
    at.text_input[0].set_value("Как часто требуется проверка?").run()
    [b for b in at.button if b.label == "Спросить"][0].click().run()

    assert not at.exception
    assert len(at.status) == 1
    body = " ".join(m.value for m in at.markdown)
    for label in (
        "Поиск требований",
        "Формирование ответа по найденным основаниям",
        "Проверка доказательств",
    ):
        assert label in body


def test_clarification_shows_question_and_answer_buttons(monkeypatch, tmp_path):
    response = _low_confidence_response("needs_context", ["applicability_unclear"])
    response.clarifying_questions = [
        "Есть ли на объекте круглосуточное пребывание людей?"
    ]
    at = _submit(monkeypatch, tmp_path, response)

    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "Есть ли на объекте круглосуточное пребывание людей?" in body

    labels = [b.label for b in at.button]
    for label in ("Да", "Нет", "Не знаю"):
        assert label in labels


def _conflict_response() -> DepartmentResponse:
    return DepartmentResponse(
        status="needs_review",
        reason_codes=["possible_mismatch"],
        trace_id="t",
        answer=ANSWER,
        evidence=EVIDENCE,
        external_basis=[
            Basis(statement="Проверка не реже раза в год", evidence_ids=["ext_001"])
        ],
        internal_basis=[
            Basis(statement="Осмотр по инструкции", evidence_ids=["int_001"])
        ],
    )


def test_conflict_shows_clashing_bases(monkeypatch, tmp_path):
    at = _submit(monkeypatch, tmp_path, _conflict_response())

    assert not at.exception
    subheaders = [h.value for h in at.subheader]
    assert "Обнаружено расхождение" in subheaders

    body = " ".join(m.value for m in at.markdown)
    assert "Проверка не реже раза в год" in body
    assert "Осмотр по инструкции" in body
    assert "Требуется проверка специалистом." in body
    assert ANSWER not in body


def test_insufficient_evidence_has_no_conflict_block(monkeypatch, tmp_path):
    at = _submit(
        monkeypatch,
        tmp_path,
        _low_confidence_response("needs_review", ["internal_evidence_missing"]),
    )

    assert not at.exception
    assert "Обнаружено расхождение" not in [h.value for h in at.subheader]


class _FakeFeedbackWriter:
    def __init__(self):
        self.votes = []

    def record(self, query_id, verdict, comment=None):
        self.votes.append((query_id, verdict, comment))


def test_feedback_buttons_show_and_record_when_writer_available(monkeypatch, tmp_path):
    fake = _FakeFeedbackWriter()
    monkeypatch.setattr("src.ui_feedback.get_feedback_writer", lambda: fake)

    at = _submit(monkeypatch, tmp_path, _answered_response())

    labels = [b.label for b in at.button]
    assert "👍" in labels and "👎" in labels

    [b for b in at.button if b.label == "👍"][0].click().run()
    assert fake.votes == [("t", 1, None)]
