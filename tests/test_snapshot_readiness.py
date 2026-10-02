"""Managed candidates must pass readiness and embedding identity checks."""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.indexing.vector_store import load_bound_vector_store

pytestmark = pytest.mark.unit


def test_partial_candidate_rejected_before_chroma_is_opened(tmp_path):
    target = tmp_path / "active.snapshots" / "build-partial"
    target.mkdir(parents=True)
    with patch("src.indexing.vector_store.Chroma") as chroma:
        with pytest.raises(ValueError, match="ready"):
            load_bound_vector_store(
                path=str(target), collection="norms", embeddings=MagicMock()
            )
    chroma.assert_not_called()


def test_moved_unfinished_candidate_rejected(tmp_path):
    (tmp_path / "snapshot-build.json").write_text('{"status":"building"}')
    with patch("src.indexing.vector_store.Chroma") as chroma:
        with pytest.raises(ValueError, match="ready"):
            load_bound_vector_store(
                path=str(tmp_path), collection="norms", embeddings=MagicMock()
            )
    chroma.assert_not_called()


def test_corrupted_report_rejected(tmp_path):
    (tmp_path / "snapshot-report.json").write_text('{"status":"validated"}')
    with patch("src.indexing.vector_store.Chroma") as chroma:
        with pytest.raises(ValueError):
            load_bound_vector_store(
                path=str(tmp_path), collection="norms", embeddings=MagicMock()
            )
    chroma.assert_not_called()


def _report(path, **changes):
    report = {
        "schema_version": 2,
        "status": "validated",
        "snapshot_path": str(path),
        "collection": "norms",
        "chunk_count": 1,
        "content_digest": "a" * 64,
        "embedding": {"provider": "openai", "model": "text-embedding-3-small"},
    }
    report.update(changes)
    (path / "snapshot-report.json").write_text(json.dumps(report))


def test_embedding_mismatch_rejected_before_open(tmp_path):
    _report(tmp_path)
    embeddings = MagicMock(model="another-model")
    with patch("src.indexing.vector_store.Chroma") as chroma:
        with pytest.raises(ValueError, match="model"):
            load_bound_vector_store(
                path=str(tmp_path), collection="norms", embeddings=embeddings
            )
    chroma.assert_not_called()


def test_loaded_count_mismatch_rejected(tmp_path):
    _report(tmp_path)
    with patch("src.indexing.vector_store.Chroma") as chroma:
        chroma.return_value._collection.count.return_value = 2
        with pytest.raises(ValueError, match="count"):
            load_bound_vector_store(
                path=str(tmp_path),
                collection="norms",
                embeddings=MagicMock(model="openai/text-embedding-3-small"),
            )


def test_legacy_store_remains_loadable(tmp_path):
    with patch("src.indexing.vector_store.Chroma") as chroma:
        chroma.return_value._collection.count.return_value = 1
        assert (
            load_bound_vector_store(
                path=str(tmp_path), collection="legacy", embeddings=MagicMock()
            )
            is chroma.return_value
        )


def test_ready_snapshot_checks_content_even_when_count_matches(tmp_path):
    _report(tmp_path)
    with patch("src.indexing.vector_store.Chroma") as chroma:
        chroma.return_value._collection.count.return_value = 1
        chroma.return_value.get.return_value = {
            "documents": ["corrupted"],
            "metadatas": [{"source": "a.md"}],
        }
        with pytest.raises(ValueError, match="content"):
            load_bound_vector_store(
                path=str(tmp_path),
                collection="norms",
                embeddings=MagicMock(model="text-embedding-3-small"),
            )


def test_ready_snapshot_with_matching_checksum_loads(tmp_path):
    from collections import Counter

    from src.indexing.snapshot import counter_digest, record_digest

    metadata = {"source": "a.md", "chunk_id": 0}
    digest = counter_digest(Counter([record_digest("norm", metadata)]))
    _report(tmp_path, content_digest=digest)
    with patch("src.indexing.vector_store.Chroma") as chroma:
        chroma.return_value._collection.count.return_value = 1
        chroma.return_value.get.return_value = {
            "documents": ["norm"],
            "metadatas": [metadata],
        }
        assert (
            load_bound_vector_store(
                path=str(tmp_path),
                collection="norms",
                embeddings=MagicMock(model="openai/text-embedding-3-small"),
            )
            is chroma.return_value
        )
