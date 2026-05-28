from unittest.mock import MagicMock
import pytest
from src.indexing.file_handler import DocumentProcessor


def _make_chunk(text, headings=None, page_no=None, bbox=None):
    """Creates a mock HybridChunker chunk."""
    chunk = MagicMock()
    chunk.text = text
    chunk.meta.headings = headings or []
    if page_no is not None or bbox is not None:
        item = MagicMock()
        prov = MagicMock()
        prov.page_no = page_no
        if bbox is not None:
            prov.bbox = MagicMock()
            prov.bbox.as_tuple.return_value = bbox
        else:
            prov.bbox = None
        item.prov = [prov]
        chunk.meta.doc_items = [item]
    else:
        chunk.meta.doc_items = []
    return chunk


def _make_processor_with_chunks(chunks_data):
    """Creates a DocumentProcessor with a mocked _chunker."""
    processor = DocumentProcessor()
    processor._chunker = MagicMock()
    processor._chunker.chunk.return_value = iter(chunks_data)
    return processor


@pytest.mark.unit
def test_extract_chunks_hybrid():
    chunks_data = [
        _make_chunk(
            "Chapter 1",
            headings=["Chapter 1"],
            page_no=1,
            bbox=(10.0, 50.0, 200.0, 30.0),
        ),
        _make_chunk(
            "Content", headings=["Chapter 1"], page_no=1, bbox=(10.0, 50.0, 200.0, 30.0)
        ),
    ]
    processor = _make_processor_with_chunks(chunks_data)
    chunks = processor._process_docling_document(MagicMock(), "test.pdf")

    assert len(chunks) == 2
    assert chunks[0].metadata["type"] == "hybrid_chunk"
    assert chunks[0].page_content == "Chapter 1"
    assert chunks[0].metadata["parent_section"] == "Chapter 1"
    assert chunks[1].metadata["type"] == "hybrid_chunk"
    assert "Content" in chunks[1].page_content
    assert chunks[1].metadata["parent_section"] == "Chapter 1"


@pytest.mark.unit
def test_short_bbox_keeps_text_drops_bbox():
    """Regression: short bbox (< MIN_BBOX_HEIGHT) must not drop text from the index."""
    chunk = _make_chunk(
        "Повторный инструктаж проводится не реже 1 раза в 6 месяцев.",
        headings=[],
        page_no=4,
        bbox=(10.0, 50.0, 200.0, 44.0),  # height = abs(44-50) = 6 < 7
    )
    processor = _make_processor_with_chunks([chunk])
    chunks = processor._process_docling_document(MagicMock(), "2464.pdf")

    assert len(chunks) == 1
    assert "Повторный инструктаж" in chunks[0].page_content
    assert chunks[0].metadata.get("bbox") is None


@pytest.mark.unit
def test_blacklist_phrase_filtered():
    """Chunks with blacklist phrases must be filtered out."""
    chunk = _make_chunk("Скачано с https://example.com", headings=[])
    processor = _make_processor_with_chunks([chunk])
    chunks = processor._process_docling_document(MagicMock(), "test.pdf")

    assert len(chunks) == 0


@pytest.mark.unit
def test_heading_path_populated():
    """heading_path must contain the full heading path joined with ' > '."""
    chunk = _make_chunk("Body text", headings=["Chapter 1", "Section 1.1"])
    processor = _make_processor_with_chunks([chunk])
    chunks = processor._process_docling_document(MagicMock(), "test.pdf")

    assert len(chunks) == 1
    assert chunks[0].metadata["parent_section"] == "Section 1.1"
    assert chunks[0].metadata["heading_path"] == "Chapter 1 > Section 1.1"
