"""Unit tests for src/v7/context_format.py — the generator's context text."""

from __future__ import annotations

import pytest

from src.v7.context_format import format_generation_context

pytestmark = pytest.mark.unit


def _p(text, score, source, section=""):
    meta = {"source": source}
    if section:
        meta["parent_section"] = section
    return {"text": text, "score": score, "metadata": meta}


def test_headers_number_score_source_and_section():
    passages = [
        _p("Текст А", 0.7, "2464.pdf", "Раздел II"),
        _p("Текст Б", 0.5, "Трудовой кодекс РФ - Система Охрана труда.pdf"),
        _p("Текст В", 0.1, ""),
    ]
    assert format_generation_context(passages) == (
        "[1] (HIGH) [Источник: 2464; Раздел: Раздел II]\nТекст А\n\n"
        "[2] (MED) [Источник: Трудовой кодекс РФ]\nТекст Б\n\n"
        "[3] (LOW) [Источник: Unknown source]\nТекст В"
    )


def test_long_section_is_truncated_and_heading_path_is_fallback():
    p = {
        "text": "т",
        "score": 0.0,
        "metadata": {"source": "a.pdf", "heading_path": "x" * 130},
    }
    assert format_generation_context([p]).startswith(
        "[1] (LOW) [Источник: a; Раздел: " + "x" * 120 + "…]"
    )


def test_empty():
    assert format_generation_context([]) == ""
