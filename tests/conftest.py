"""Shared pytest isolation from developer-machine runtime configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def disable_runtime_domain_gate(monkeypatch):
    """Keep unrelated tests from inheriting the production gate from ``.env``."""
    from src.v7.config import v7_config

    monkeypatch.setattr(v7_config, "DOMAIN_GATE_THRESHOLD", 0.0)
