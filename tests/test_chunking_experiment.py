"""Experimental serialization preserves list stems and merged table categories."""

import pytest
import tiktoken

from scripts.compare_ingest_chunking import bounded_context, table_fixture, table_rows

pytestmark = pytest.mark.unit


def test_table_rows_repeat_merged_category_and_headers():
    rows = table_rows(table_fixture())
    assert len(rows) == 10
    assert all(
        "Категория" in row and "Управление транспортными средствами" in row
        for row in rows
    )
    assert "Водители погрузчиков" in rows[-1]


def test_oversize_norm_is_reported_instead_of_truncated():
    with pytest.raises(ValueError, match="budget"):
        bounded_context(
            "Вводная", "слово " * 300, tiktoken.get_encoding("cl100k_base"), 20
        )
