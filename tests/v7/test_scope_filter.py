"""Tests for department scope filters used by dense, BM25 and cross_ref paths (spec §6)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.v7.scope_filter import build_scope_filters, matches_filter

EXT = {"source": "ppr.pdf", "source_type": "external"}
COMPANY = {"source": "prikaz.md", "source_type": "internal", "audience": "company"}
UNIT1 = {"source": "u1.md", "source_type": "internal", "audience": "unit_1"}
UNIT2 = {"source": "u2.md", "source_type": "internal", "audience": "unit_2"}


@pytest.mark.unit
def test_unit_selected_sees_company_and_own_unit_only():
    ext, internal = build_scope_filters("unit_1")
    assert [matches_filter(m, ext) for m in (EXT, COMPANY, UNIT1)] == [
        True,
        False,
        False,
    ]
    assert [matches_filter(m, internal) for m in (EXT, COMPANY, UNIT1, UNIT2)] == [
        False,
        True,
        True,
        False,
    ]


@pytest.mark.unit
def test_no_unit_excludes_all_unit_documents():
    _, internal = build_scope_filters(None)
    assert matches_filter(COMPANY, internal)
    assert not matches_filter(UNIT1, internal)


@pytest.mark.unit
def test_empty_filter_matches_everything():
    assert matches_filter(UNIT2, None)
    assert matches_filter(UNIT2, {})


@pytest.mark.unit
def test_bm25_respects_in_operator():
    from src.v7.nlp_core import BM25Index

    passages = [
        {"text": "огнетушитель осмотр", "metadata": COMPANY},
        {"text": "огнетушитель осмотр", "metadata": UNIT2},
    ]
    _, internal = build_scope_filters("unit_1")
    hits = BM25Index(passages).search("огнетушитель", top_k=5, filters=internal)
    assert [h["metadata"]["source"] for h in hits] == ["prikaz.md"]


@pytest.mark.unit
def test_cross_ref_applies_scope_to_backend_and_bm25():
    from src.v7.cross_ref import expand_cross_references

    _, internal = build_scope_filters("unit_1")
    backend = MagicMock()
    backend.get_by_filter.return_value = [
        Document(page_content="46. Порядок осмотра.", metadata=dict(COMPANY))
    ]
    passage = {
        "text": "согласно пункту 46 порядка",
        "score": 0.5,
        "metadata": dict(COMPANY),
    }
    leaked = {"text": "чужое подразделение", "metadata": dict(UNIT2)}

    with patch("src.v7.cross_ref.bm25_search", return_value=[leaked]) as bm25:
        out = expand_cross_references(
            [passage], backend, query="осмотр", filters=internal
        )

    where = backend.get_by_filter.call_args.kwargs["where"]
    assert where["source"] == "prikaz.md"
    assert where["audience"] == {"$in": ["company", "unit_1"]}
    assert bm25.call_args.kwargs["filters"] == internal
    assert all(p["metadata"].get("audience") != "unit_2" for p in out)
