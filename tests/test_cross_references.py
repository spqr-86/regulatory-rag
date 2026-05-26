"""Tests for cross-reference expansion in bridge.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


def _passage(text: str, source: str = "2464.pdf", score: float = 0.5) -> dict:
    return {"text": text, "score": score, "metadata": {"source": source}}


@pytest.mark.unit
def test_extract_refs_finds_punkt_number():
    from src.v7.bridge import _extract_refs

    refs = _extract_refs("работников в соответствии с пунктом 46 настоящих Правил")
    assert "46" in refs


@pytest.mark.unit
def test_extract_refs_finds_podpunkt_letter():
    from src.v7.bridge import _extract_refs

    refs = _extract_refs("обучение по подпункту «в» пункта 46 Правил")
    assert any("в" in r for r in refs)


@pytest.mark.unit
def test_extract_refs_finds_statya_number():
    from src.v7.bridge import _extract_refs

    refs = _extract_refs("согласно статье 229 Трудового кодекса")
    assert "229" in refs


@pytest.mark.unit
def test_extract_refs_no_refs_returns_empty():
    from src.v7.bridge import _extract_refs

    refs = _extract_refs("обычный текст без ссылок на пункты")
    assert refs == []


@pytest.mark.unit
def test_expand_adds_referenced_chunk():
    """If passage references п. 60, that chunk must be fetched and appended."""
    from src.v7.bridge import expand_cross_references

    original = _passage(
        "работники обучаются согласно подпункту «в» пункта 46 настоящих Правил",
        source="2464.pdf",
    )
    fetched_doc = Document(
        page_content="60. Периодичность по программе В — не реже одного раза в год.",
        metadata={"source": "2464.pdf"},
    )

    backend = MagicMock()
    backend.get_by_filter.return_value = [fetched_doc]

    result = expand_cross_references([original], backend)

    texts = [p["text"] for p in result]
    assert any("раза в год" in t for t in texts), "fetched chunk must be in result"


@pytest.mark.unit
def test_expand_does_not_add_duplicate():
    """Chunk already present in passages must not be added again."""
    from src.v7.bridge import expand_cross_references

    existing_text = "60. Периодичность по программе В — не реже одного раза в год."
    original = _passage(
        "согласно пункту 46 настоящих Правил",
        source="2464.pdf",
    )
    already_in = _passage(existing_text, source="2464.pdf")

    fetched_doc = Document(
        page_content=existing_text,
        metadata={"source": "2464.pdf"},
    )
    backend = MagicMock()
    backend.get_by_filter.return_value = [fetched_doc]

    result = expand_cross_references([original, already_in], backend)

    assert len([p for p in result if p["text"] == existing_text]) == 1


@pytest.mark.unit
def test_expand_only_searches_same_source():
    """Cross-ref search must filter by the same source document."""
    from src.v7.bridge import expand_cross_references

    passage = _passage("согласно пункту 46 настоящих Правил", source="2464.pdf")
    backend = MagicMock()
    backend.get_by_filter.return_value = []

    expand_cross_references([passage], backend)

    calls = backend.get_by_filter.call_args_list
    assert len(calls) > 0
    for call in calls:
        where = call[0][0] if call[0] else call[1].get("where", {})
        assert "source" in str(where), "must filter by source"
        assert "2464.pdf" in str(where)


@pytest.mark.unit
def test_expand_empty_passages_returns_empty():
    from src.v7.bridge import expand_cross_references

    backend = MagicMock()
    result = expand_cross_references([], backend)
    assert result == []
    backend.get_by_filter.assert_not_called()
