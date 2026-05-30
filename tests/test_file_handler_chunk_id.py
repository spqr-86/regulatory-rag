"""Tests for per-source sequential chunk_id assignment in file_handler.process()."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.indexing.file_handler import DocumentProcessor


def _stream():
    return io.BytesIO(b"x")


def _doc(text: str, source: str) -> Document:
    return Document(page_content=text, metadata={"source": source})


@pytest.fixture
def processor():
    # Avoid constructing Docling/HybridChunker (heavy, network).
    with patch.object(DocumentProcessor, "__init__", lambda self: None):
        p = DocumentProcessor()
    return p


@pytest.mark.unit
def test_chunk_id_is_per_source_zero_based_contiguous(processor):
    docs_a = [_doc(f"text a {i}", "a.pdf") for i in range(3)]
    docs_b = [_doc(f"text b {i}", "b.pdf") for i in range(2)]

    def fake_extract(stream, name, file_hash):
        return docs_a if name == "a.pdf" else docs_b

    with (
        patch.object(processor, "validate_files", lambda files: None),
        patch.object(
            processor,
            "_get_stream_and_name",
            side_effect=[(_stream(), "a.pdf"), (_stream(), "b.pdf")],
        ),
        patch.object(processor, "_hash_bytes_stream", side_effect=["ha", "hb"]),
        patch.object(processor, "_cache_path_for", side_effect=lambda h: None),
        patch.object(processor, "_is_cache_valid", return_value=False),
        patch.object(processor, "_convert_and_extract", side_effect=fake_extract),
        patch.object(processor, "_save_to_cache", lambda chunks, path: None),
    ):
        result = processor.process(["a.pdf", "b.pdf"])

    by_source: dict[str, list[int]] = {}
    for ch in result:
        by_source.setdefault(ch.metadata["source"], []).append(ch.metadata["chunk_id"])

    assert by_source["a.pdf"] == [0, 1, 2]
    assert by_source["b.pdf"] == [0, 1]
    # chunk_id must be int
    assert all(isinstance(ch.metadata["chunk_id"], int) for ch in result)


@pytest.mark.unit
def test_chunk_id_contiguous_after_dedup(processor):
    # Two identical chunks within the same source: dedup drops one, ids must
    # stay contiguous on the survivors (no gap from the dropped duplicate).
    docs_a = [
        _doc("уникальный 1", "a.pdf"),
        _doc("дубликат", "a.pdf"),
        _doc("дубликат", "a.pdf"),  # dropped by global content dedup
        _doc("уникальный 2", "a.pdf"),
    ]

    with (
        patch.object(processor, "validate_files", lambda files: None),
        patch.object(
            processor, "_get_stream_and_name", side_effect=[(_stream(), "a.pdf")]
        ),
        patch.object(processor, "_hash_bytes_stream", side_effect=["ha"]),
        patch.object(processor, "_cache_path_for", side_effect=lambda h: None),
        patch.object(processor, "_is_cache_valid", return_value=False),
        patch.object(processor, "_convert_and_extract", return_value=docs_a),
        patch.object(processor, "_save_to_cache", lambda chunks, path: None),
    ):
        result = processor.process(["a.pdf"])

    ids = [ch.metadata["chunk_id"] for ch in result]
    assert ids == list(range(len(result)))
    assert len(result) == 3  # one duplicate removed
