"""Department page: a misconfigured stack is shown as an error, not a traceback (#47 п. 2)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from config.settings import settings
from src.department_qa import wiring
from src.department_qa.object_profile import ObjectProfileError

PAGE = str(Path(__file__).resolve().parents[2] / "pages" / "1_Подразделения.py")


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("DEPARTMENT_QA_MODE=v2 needs CHROMA_DB_PATH=./chroma_db_dept_v2"),
        ObjectProfileError("unit_office: sheet has no sections"),
    ],
)
def test_page_shows_stack_error_instead_of_traceback(monkeypatch, tmp_path, exc):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("documents: {}\n", encoding="utf-8")
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", str(manifest))

    def _raise():
        raise exc

    monkeypatch.setattr(wiring, "build_department_stack", _raise)

    at = AppTest.from_file(PAGE, default_timeout=30).run()

    assert not at.exception
    assert len(at.error) == 1
    assert str(exc) in at.error[0].value
