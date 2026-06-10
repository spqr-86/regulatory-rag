"""Tests for cross-reference expansion in bridge.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document


def _passage(text: str, source: str = "2464.pdf", score: float = 0.5) -> dict:
    return {"text": text, "score": score, "metadata": {"source": source}}


@pytest.mark.unit
def test_extract_refs_finds_punkt_number():
    from src.v7.cross_ref import _extract_refs

    refs = _extract_refs("работников в соответствии с пунктом 46 настоящих Правил")
    assert ("clause", "46") in refs


@pytest.mark.unit
def test_extract_refs_finds_podpunkt_letter():
    from src.v7.cross_ref import _extract_refs

    refs = _extract_refs("обучение по подпункту «в» пункта 46 Правил")
    assert ("subpara", "в") in refs


@pytest.mark.unit
def test_extract_refs_finds_statya_number():
    from src.v7.cross_ref import _extract_refs

    refs = _extract_refs("согласно статье 229 Трудового кодекса")
    assert ("article", "229") in refs


@pytest.mark.unit
def test_extract_refs_no_refs_returns_empty():
    from src.v7.cross_ref import _extract_refs

    refs = _extract_refs("обычный текст без ссылок на пункты")
    assert refs == []


@pytest.mark.unit
def test_expand_adds_referenced_chunk():
    """If passage references п. 46, the chunk starting with '46.' must be added."""
    from src.v7.cross_ref import expand_cross_references

    original = _passage(
        "работники обучаются согласно подпункту «в» пункта 46 настоящих Правил",
        source="2464.pdf",
    )
    fetched_doc = Document(
        page_content="46. Периодичность инструктажа устанавливается работодателем.",
        metadata={"source": "2464.pdf"},
    )

    backend = MagicMock()
    backend.get_by_filter.return_value = [fetched_doc]

    result = expand_cross_references([original], backend)

    texts = [p["text"] for p in result]
    assert any("Периодичность инструктажа" in t for t in texts), (
        "п.46 chunk must be in result"
    )


@pytest.mark.unit
def test_expand_does_not_add_duplicate():
    """Chunk already present in passages must not be added again."""
    from src.v7.cross_ref import expand_cross_references

    existing_text = "46. Периодичность инструктажа устанавливается работодателем."
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
    from src.v7.cross_ref import expand_cross_references

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
    from src.v7.cross_ref import expand_cross_references

    backend = MagicMock()
    result = expand_cross_references([], backend)
    assert result == []
    backend.get_by_filter.assert_not_called()


@pytest.mark.unit
def test_each_source_fetched_at_most_once(monkeypatch):
    """H2: a source's docs must be fetched once per query, reused across mechanisms."""
    from src.v7 import cross_ref

    monkeypatch.setattr(cross_ref, "bm25_search", lambda *a, **k: [])

    passages = [
        _passage("согласно пункту 46 настоящих Правил", source="2464.pdf"),
        _passage("в соответствии с пунктом 53 настоящих Правил", source="2464.pdf"),
        _passage("46. а) первый подпункт списка", source="2464.pdf"),
        _passage("текст со ссылкой статья 229", source="tk.pdf"),
    ]

    fetch_counter: dict[str, int] = {}

    def counting_get_by_filter(where=None, limit=None, **kwargs):
        src = where.get("source") if isinstance(where, dict) else None
        fetch_counter[src] = fetch_counter.get(src, 0) + 1
        return []

    backend = MagicMock()
    backend.get_by_filter.side_effect = counting_get_by_filter

    cross_ref.expand_cross_references(passages, backend, query="инструктаж")

    assert fetch_counter.get("2464.pdf", 0) == 1, fetch_counter
    assert fetch_counter.get("tk.pdf", 0) == 1, fetch_counter


@pytest.mark.unit
def test_added_cross_ref_lifts_chunk_id():
    """Cross-ref passages must carry top-level chunk_id from fetched doc metadata."""
    from src.v7.cross_ref import expand_cross_references

    original = _passage("согласно пункту 60 настоящих Правил", source="2464.pdf")
    fetched_doc = Document(
        page_content="60. Периодичность по программе В — не реже одного раза в год.",
        metadata={"source": "2464.pdf", "chunk_id": 99},
    )
    backend = MagicMock()
    backend.get_by_filter.return_value = [fetched_doc]

    result = expand_cross_references([original], backend)
    added = [p for p in result if p.get("cross_ref")]
    assert added and added[0]["chunk_id"] == 99


# ── Phase 2: structural matching & caps tests ────────────────────────────────


@pytest.mark.unit
def test_clause_ref_no_false_positive():
    """Chunks containing '2464' or '146' must NOT match a ref to 'пункт 46'."""
    from src.v7.cross_ref import expand_cross_references

    passage = _passage(
        "работники проходят инструктаж согласно пункту 46 настоящих Правил",
        source="2464.pdf",
    )
    false_positive_1 = Document(
        page_content="Приказ Минтруда России от 28.10.2020 № 2464 об обучении.",
        metadata={"source": "2464.pdf"},
    )
    false_positive_2 = Document(
        page_content="Таблица 146: допустимые нормы вибрации на рабочих местах.",
        metadata={"source": "2464.pdf"},
    )
    true_positive = Document(
        page_content="46. Периодичность инструктажа устанавливается работодателем.",
        metadata={"source": "2464.pdf"},
    )
    backend = MagicMock()
    backend.get_by_filter.return_value = [
        false_positive_1,
        false_positive_2,
        true_positive,
    ]

    result = expand_cross_references([passage], backend)
    added_texts = [p["text"] for p in result if p.get("cross_ref")]

    assert false_positive_1.page_content not in added_texts, (
        "2464 should not match п.46"
    )
    assert false_positive_2.page_content not in added_texts, "146 should not match п.46"
    assert true_positive.page_content in added_texts, "real п.46 chunk must be added"


@pytest.mark.unit
def test_subpara_no_bare_letter_match():
    """Passage with 'подпункта «а»' must NOT add chunks that only contain 'а' as a letter."""
    from src.v7.cross_ref import expand_cross_references

    # Passage references both a clause AND subpara «а»
    passage = _passage(
        "обучение по подпункту «а» пункта 46 Правил охраны труда",
        source="2464.pdf",
    )
    # Chunk that just happens to contain letter «а» in Russian text — everywhere
    bare_letter = Document(
        page_content="Работодатель обязан обеспечить безопасные условия труда.",
        metadata={"source": "2464.pdf"},
    )
    # Chunk that has the actual list marker "а)" at line start
    real_subpara = Document(
        page_content="46. Программы обучения:\nа) программа А — ежегодно.",
        metadata={"source": "2464.pdf"},
    )
    backend = MagicMock()
    backend.get_by_filter.return_value = [bare_letter, real_subpara]

    result = expand_cross_references([passage], backend)
    added_texts = [p["text"] for p in result if p.get("cross_ref")]

    assert bare_letter.page_content not in added_texts, (
        "bare letter chunk must not match"
    )
    assert real_subpara.page_content in added_texts, "real subpara marker must match"


@pytest.mark.unit
def test_expansion_caps(monkeypatch):
    """Synthetic source of 100 chunks — mech1 must add ≤ 15 chunks total."""
    from src.v7 import cross_ref

    monkeypatch.setattr(cross_ref, "bm25_search", lambda *a, **k: [])

    # Single passage referencing пункт 1
    passage = _passage("согласно пункту 1 настоящих Правил", source="big.pdf")

    # 100 chunks all matching "1. текст"
    docs = [
        Document(
            page_content=f"1. Пункт первый, подпункт {i}.",
            metadata={"source": "big.pdf"},
        )
        for i in range(100)
    ]
    backend = MagicMock()
    backend.get_by_filter.return_value = docs

    result = cross_ref.expand_cross_references([passage], backend)
    added = [p for p in result if p.get("cross_ref")]

    assert len(added) <= 15, f"expected ≤15 added chunks, got {len(added)}"


@pytest.mark.unit
def test_article_ref_phrase_match():
    """'статьи 5' must match chunk with 'статьёй 5', but NOT a chunk with bare '5'."""
    from src.v7.cross_ref import expand_cross_references

    passage = _passage(
        "в соответствии со статьёй 5 Федерального закона",
        source="fz.pdf",
    )
    bare_number = Document(
        page_content="Глава 5. Общие положения об охране труда.",
        metadata={"source": "fz.pdf"},
    )
    phrase_match = Document(
        page_content="Статьёй 5 установлено право работника на безопасный труд.",
        metadata={"source": "fz.pdf"},
    )
    backend = MagicMock()
    backend.get_by_filter.return_value = [bare_number, phrase_match]

    result = expand_cross_references([passage], backend)
    added_texts = [p["text"] for p in result if p.get("cross_ref")]

    assert bare_number.page_content not in added_texts, (
        "bare '5' must not match статьи 5"
    )
    assert phrase_match.page_content in added_texts, "'статьёй 5' phrase must match"
