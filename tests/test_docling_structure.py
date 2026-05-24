from unittest.mock import MagicMock
import json
from src.indexing.file_handler import DocumentProcessor

# Correct imports based on inspection
from docling.datamodel.document import SectionHeaderItem, TextItem
from docling.datamodel.base_models import BoundingBox
from docling_core.types.doc.document import ProvenanceItem


def test_extract_chunks_real_types():
    processor = DocumentProcessor()

    # Create real objects
    bbox = BoundingBox(l=10, b=30, r=200, t=50)
    prov = ProvenanceItem(page_no=1, bbox=bbox, charspan=[0, 10])

    # Create items with provenance
    header = SectionHeaderItem(
        text="Chapter 1",
        orig="Chapter 1",
        label="section_header",
        self_ref="#/header1",
        parent=None,
        children=[],
        prov=[prov],
    )

    text = TextItem(
        text="Content",
        orig="Content",
        label="text",
        self_ref="#/text1",
        parent=None,
        children=[],
        prov=[prov],
    )

    doc = MagicMock()
    # Mock texts() iterator to return our items
    doc.texts.return_value = iter([header, text])

    chunks = processor._process_docling_document(doc, "test.pdf")

    assert len(chunks) == 2

    # Check Header
    assert chunks[0].metadata["type"] == "header"
    assert "Chapter 1" in chunks[0].page_content
    # Check context injection (now in metadata)
    assert chunks[0].metadata["parent_section"] == "Chapter 1"
    # assert "[Chapter 1]" in chunks[0].page_content # Old behavior removed

    # Check BBox
    bbox_json = chunks[0].metadata["bbox"]
    # Docling returns [l, t, r, b] where t=50, b=30 (PDF coords)
    assert json.loads(bbox_json) == [10.0, 50.0, 200.0, 30.0]

    # Check Text
    assert chunks[1].metadata["type"] == "grouped_text"  # Now it's grouped_text
    assert "Content" in chunks[1].page_content
    # Context should NOT be in content anymore, but in metadata
    assert "[Chapter 1]" not in chunks[1].page_content
    assert chunks[1].metadata["parent_section"] == "Chapter 1"


def test_short_bbox_keeps_text_drops_bbox():
    """Регрессия: короткий bbox (< MIN_BBOX_HEIGHT) не должен ронять текст из индекса.

    Раньше items с bbox_h < 7 выбрасывались целиком — это роняло однострочные
    пункты норм (напр. «Повторный инструктаж ... 6 месяцев» из ППРФ 2464).
    Теперь текст остаётся, bbox просто отсутствует.
    """
    processor = DocumentProcessor()

    short_bbox = BoundingBox(l=10, b=44, r=200, t=50)  # height=6, < 7
    prov_short = ProvenanceItem(page_no=4, bbox=short_bbox, charspan=[0, 10])

    item = TextItem(
        text="Повторный инструктаж проводится не реже 1 раза в 6 месяцев.",
        orig="Повторный инструктаж проводится не реже 1 раза в 6 месяцев.",
        label="text",
        self_ref="#/text-short",
        parent=None,
        children=[],
        prov=[prov_short],
    )

    doc = MagicMock()
    doc.texts.return_value = iter([item])

    chunks = processor._process_docling_document(doc, "2464.pdf")

    assert len(chunks) == 1
    assert "Повторный инструктаж" in chunks[0].page_content
    assert chunks[0].metadata.get("bbox") is None
