"""Isolation guards for the unified UI's Generic normative store."""

import pytest

from config.settings import settings
from src import ui_generic


def test_generic_store_must_not_alias_department_v2(monkeypatch, tmp_path):
    shared = str(tmp_path / "shared")
    monkeypatch.setattr(settings, "GENERIC_CHROMA_DB_PATH", shared)
    monkeypatch.setattr(settings, "GENERIC_CHROMA_COLLECTION", "department_demo_v2")
    monkeypatch.setattr(settings, "DEPARTMENT_V2_CHROMA_DB_PATH", shared)
    monkeypatch.setattr(settings, "DEPARTMENT_V2_COLLECTION", "department_demo_v2")

    with pytest.raises(ui_generic.GenericSearchError, match="корпус подразделений"):
        ui_generic._validate_store_isolation()
