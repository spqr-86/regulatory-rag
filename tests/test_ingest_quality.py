"""Reject degraded conversion and preserve legal text and occurrence identity."""

import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.indexing.file_handler import DocumentProcessor, ProcessingError, _clean_noise

pytestmark = pytest.mark.unit


def test_cleaning_preserves_identifiers_urls_and_legitimate_page_word():
    text = "приказ № 14/34\nСтраница журнала должна быть подписана.\nСм. https://example.org/act\nкод 14-\n34"
    assert _clean_noise(text) == text


def test_cleaning_removes_only_boilerplate_lines():
    assert (
        _clean_noise("Скачано с 1otruda.ru\nСтраница 2 из 10\nПолезная норма")
        == "Полезная норма"
    )


def test_partial_conversion_rejected(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    processor._docling = SimpleNamespace(
        convert=lambda _: SimpleNamespace(
            status="partial_success", errors=[], document=None
        )
    )
    with patch.object(
        processor,
        "_process_docling_document",
        return_value=[Document(page_content="partial")],
    ):
        with pytest.raises(ProcessingError, match="partial_success"):
            processor._convert_and_extract(io.BytesIO(b"x"), "norm.pdf", "hash")


def test_strict_rejects_fallback_chunks(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    source = tmp_path / "norm.docx"
    source.write_bytes(b"x")
    with patch.object(
        processor,
        "_convert_and_extract",
        return_value=[
            Document(page_content="flattened table", metadata={"type": "fallback_lxml"})
        ],
    ):
        with pytest.raises(ProcessingError, match="fallback"):
            processor.process([source], strict=True)


@pytest.mark.parametrize("cached_fallback", [False, True])
def test_strict_retries_after_degraded_conversion(tmp_path, cached_fallback):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    source = tmp_path / "norm.docx"
    source.write_bytes(b"x")
    degraded = Document(page_content="flattened", metadata={"type": "fallback_lxml"})
    complete = Document(page_content="complete", metadata={"type": "text"})
    with patch.object(
        processor, "_convert_and_extract", side_effect=[[degraded], [complete]]
    ) as convert:
        if cached_fallback:
            processor.process([source])
            # Simulate a degraded cache written by an earlier implementation.
            key = f"{processor._hash_bytes_stream(io.BytesIO(b'x'))}:.docx"
            processor._save_to_cache([degraded], processor._cache_path_for(key))
        else:
            with pytest.raises(ProcessingError, match="fallback"):
                processor.process([source], strict=True)
        result = processor.process([source], strict=True)
    assert convert.call_count == 2
    assert [chunk.page_content for chunk in result] == ["complete"]


def test_dedup_keeps_distinct_locations_and_reports_retained_count(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    source = tmp_path / "norm.pdf"
    source.write_bytes(b"x")
    chunks = [
        Document(page_content="same", metadata={"page_no": 1}),
        Document(page_content="same", metadata={"page_no": 2}),
        Document(page_content="same", metadata={"page_no": 2}),
    ]
    with patch.object(processor, "_convert_and_extract", return_value=chunks):
        result = processor.process([source], strict=True)
    assert [c.metadata["page_no"] for c in result] == [1, 2]
    assert processor.processing_report[0].chunk_count == 2


def test_structured_ir_reused_after_chunk_cache_invalidation(tmp_path):
    from docling_core.types.doc import DoclingDocument

    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    document = DoclingDocument(name="fixture")
    converter = SimpleNamespace(
        convert=lambda _: SimpleNamespace(
            status="success", errors=[], document=document
        )
    )
    processor._docling = converter
    with patch.object(
        processor,
        "_process_docling_document",
        return_value=[Document(page_content="norm")],
    ):
        processor._convert_and_extract(io.BytesIO(b"x"), "a.pdf", "hash")
        with patch.object(
            converter, "convert", side_effect=AssertionError("must reuse IR")
        ):
            processor._convert_and_extract(io.BytesIO(b"x"), "b.pdf", "hash")
    assert list(tmp_path.glob("ir/*.json"))
