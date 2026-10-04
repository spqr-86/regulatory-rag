"""Row-level serialization of numbered regulatory tables (29н pattern)."""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.indexing.table_rows import (
    explicit_table_chunks,
    merge_page_continuations,
)

IR_29N = Path("eval/runs/table_rows_29n_hybrid_2026-10-04/29n_ir.json")


def _table(ref, page, rows):
    cells = [[SimpleNamespace(text=t) for t in row] for row in rows]
    return SimpleNamespace(
        self_ref=ref,
        prov=[SimpleNamespace(page_no=page)],
        data=SimpleNamespace(grid=cells, num_cols=len(rows[0])),
    )


def _chunk(text, refs, element_type="table", section="Приложение"):
    return Document(
        page_content=text,
        metadata={
            "source": "x.pdf",
            "type": "hybrid_chunk",
            "parent_section": section,
            "heading_path": section,
            "doc_item_refs": json.dumps(refs),
            "element_type": element_type,
            "page_no": 1,
            "bbox": "[0, 0, 1, 1]",
        },
    )


@pytest.mark.unit
def test_page_continuation_with_same_width_is_merged():
    a = _table("#/tables/0", 1, [["1", "a"]])
    b = _table("#/tables/1", 2, [["1.1", "b"]])
    c = _table("#/tables/2", 3, [["x", "y", "z"]])
    assert [[t.self_ref for t in g] for g in merge_page_continuations([a, b, c])] == [
        ["#/tables/0", "#/tables/1"],
        ["#/tables/2"],
    ]


@pytest.mark.unit
def test_sub_row_carries_parent_across_page_break():
    tables = [
        _table("#/tables/0", 1, [["18", "Управление транспортом", ""]]),
        _table("#/tables/1", 2, [["18.1", "Водители категории B", "1 раз в 2 года"]]),
    ]
    chunks = [
        _chunk("таблица стр 1", ["#/tables/0"]),
        _chunk("таблица стр 2", ["#/tables/1"]),
    ]
    out = explicit_table_chunks(tables, chunks)
    row = next(c for c in out if c.metadata.get("table_row_id") == "18.1")
    assert row.page_content.splitlines()[0] == "Приложение"
    assert "Пункт 18: Управление транспортом" in row.page_content
    assert "Водители категории B" in row.page_content
    assert "1 раз в 2 года" in row.page_content
    assert row.metadata["element_type"] == "table"
    assert "bbox" not in row.metadata


@pytest.mark.unit
def test_unnumbered_rows_of_numbered_table_are_kept():
    tables = [
        _table(
            "#/tables/0",
            1,
            [
                ["1", "Химические факторы", "1 раз в год"],
                ["Класс V. Психические расстройства", "", ""],
                ["б) глаукома III стадии", "4.2.1", "10, 16"],
            ],
        )
    ]
    out = explicit_table_chunks(tables, [_chunk("исходный", ["#/tables/0"])])
    text = "\n".join(c.page_content for c in out)
    assert "Класс V. Психические расстройства" in text
    assert "б) глаукома III стадии" in text
    assert "исходный" not in text


@pytest.mark.unit
def test_unnumbered_table_keeps_original_chunks():
    tables = [_table("#/tables/0", 1, [["Должность", "Срок"], ["Инженер", "1 год"]])]
    chunks = [_chunk("исходная таблица", ["#/tables/0"])]
    assert explicit_table_chunks(tables, chunks) == chunks


@pytest.mark.unit
def test_rows_replace_table_chunks_in_place():
    tables = [_table("#/tables/0", 1, [["1", "a"], ["2", "b"]])]
    chunks = [
        _chunk("до", [], element_type="text"),
        _chunk("таблица", ["#/tables/0"]),
        _chunk("после", [], element_type="text"),
    ]
    out = [c.page_content for c in explicit_table_chunks(tables, chunks)]
    assert out[0] == "до" and out[-1] == "после"
    assert len(out) == 4


@pytest.mark.unit
def test_document_start_is_not_used_as_heading():
    tables = [_table("#/tables/0", 1, [["1", "a"]])]
    chunks = [_chunk("t", ["#/tables/0"], section="Document start")]
    (row,) = explicit_table_chunks(tables, chunks)
    assert "Document start" not in row.page_content


@pytest.mark.unit
def test_oversized_row_is_split_without_truncation():
    long_cell = " ".join(f"слово{i}" for i in range(600))
    tables = [_table("#/tables/0", 1, [["1", "Пункт", long_cell]])]
    out = explicit_table_chunks(tables, [_chunk("t", ["#/tables/0"])], budget=100)
    assert len(out) > 1
    assert all(c.page_content.count("Колонка 1: 1") == 1 for c in out)
    joined = " ".join(c.page_content for c in out)
    assert all(f"слово{i}" in joined for i in range(600))


@pytest.mark.unit
def test_short_group_title_joins_next_row():
    tables = [
        _table(
            "#/tables/0",
            1,
            [
                ["3", "Шум", "1 раз в 2 года"],
                ["IV. Физические факторы", "", ""],
                ["4", "Вибрация", "1 раз в год"],
            ],
        )
    ]
    out = explicit_table_chunks(tables, [_chunk("t", ["#/tables/0"])])
    assert len(out) == 2
    row = next(c for c in out if c.metadata.get("table_row_id") == "4")
    assert "IV. Физические факторы" in row.page_content


@pytest.mark.unit
def test_short_lowercase_tail_joins_previous_row():
    tables = [
        _table(
            "#/tables/0",
            1,
            [
                ["5", "Осмотр врачами", "Врач-оториноларинг"],
                ["олог Врач- офтальмолог", "", ""],
                ["6", "Работы на высоте", "1 раз в 2 года"],
            ],
        )
    ]
    out = explicit_table_chunks(tables, [_chunk("t", ["#/tables/0"])])
    assert len(out) == 2
    row = next(c for c in out if c.metadata.get("table_row_id") == "5")
    assert "олог Врач- офтальмолог" in row.page_content


@pytest.mark.unit
def test_footnote_marker_joins_previous_row():
    tables = [
        _table(
            "#/tables/0",
            1,
            [
                ["18", "Управление транспортом", ""],
                ["<2>:", "", ""],
                ["18.1", "Водители категории B", "1 раз в 2 года"],
            ],
        )
    ]
    out = explicit_table_chunks(tables, [_chunk("t", ["#/tables/0"])])
    by_row = {c.metadata.get("table_row_id"): c.page_content for c in out}
    assert "<2>:" in by_row["18"]
    assert "<2>:" not in by_row["18.1"]


@pytest.mark.unit
def test_processor_without_tables_is_unchanged():
    # Mocked Docling documents in other tests have no real table list.
    assert explicit_table_chunks(MagicMock(), [_chunk("a", [])])[0].page_content == "a"


def _words(text):
    return set(re.findall(r"\w{4,}", text.lower()))


@pytest.mark.integration
@pytest.mark.skipif(not IR_29N.exists(), reason="saved 29н IR not available")
def test_29n_has_parent_rows_and_loses_no_table_text():
    from docling_core.types.doc import DoclingDocument

    from src.indexing.file_handler import DocumentProcessor

    doc = DoclingDocument.model_validate_json(IR_29N.read_text(encoding="utf-8"))
    processor = DocumentProcessor()
    chunks = processor._process_docling_document(doc, "29н.pdf")
    by_row = {c.metadata.get("table_row_id"): c.page_content for c in chunks}
    assert "Пункт 18: " in by_row["18.1"]
    indexed = _words("\n".join(c.page_content for c in chunks))
    table_words = _words(
        "\n".join(cell.text for t in doc.tables for row in t.data.grid for cell in row)
    )
    assert table_words - indexed == set()
    # Short unnumbered rows (group titles, page-break tails) are not chunks of their own.
    heads = {c.metadata.get("parent_section", "") for c in chunks}
    tiny = [
        c.page_content
        for c in chunks
        if "table_row_id" not in c.metadata
        and c.metadata.get("element_type") == "table"
        and len(c.page_content.split("\n", 1)[-1].split()) < 4
        and c.page_content.split("\n", 1)[0] in heads
    ]
    assert tiny == []
