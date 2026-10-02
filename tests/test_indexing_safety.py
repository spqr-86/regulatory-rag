"""Regression checks for isolated builds, completeness and cache provenance."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.indexing.file_handler import DocumentProcessor

pytestmark = pytest.mark.unit


@pytest.fixture
def env(tmp_path, monkeypatch):
    from config.settings import settings

    source = tmp_path / "source"
    source.mkdir()
    (source / "a.md").write_text("norm")
    active = tmp_path / "active"
    active.mkdir()
    (active / "sentinel").write_text("old index")
    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(source))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(active))
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", "")
    monkeypatch.chdir(tmp_path)
    return source, active


def test_embedding_failure_keeps_active_index(env):
    import index

    with (
        patch.object(index, "DocumentProcessor") as processor,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        processor.return_value.process.return_value = [
            Document(page_content="norm", metadata={"source": "a.md", "chunk_id": 0})
        ]
        backend.return_value.create.side_effect = RuntimeError("embedding failed")
        with pytest.raises(RuntimeError, match="embedding failed"):
            index.main()
    assert (env[1] / "sentinel").read_text() == "old index"


def test_success_builds_separate_validated_snapshot(env):
    import index

    chunk = Document(page_content="norm", metadata={"source": "a.md", "chunk_id": 0})
    store = MagicMock()
    store.count.return_value = 1
    store.iter_all_documents.return_value = iter(
        [{"text": chunk.page_content, "metadata": chunk.metadata}]
    )
    with (
        patch.object(index, "DocumentProcessor") as processor,
        patch.object(index, "get_vector_store_backend", return_value=store),
    ):
        processor.return_value.process.return_value = [chunk]
        snapshot = index.main()
    assert Path(snapshot) != env[1]
    assert (Path(snapshot) / "snapshot-report.json").is_file()
    assert (env[1] / "sentinel").read_text() == "old index"
    assert store.create.call_args.kwargs["path"] == str(snapshot)


def test_missing_document_blocks_build(env):
    import index

    (env[0] / "b.md").write_text("other norm")
    with (
        patch.object(index, "DocumentProcessor") as processor,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        processor.return_value.process.return_value = [
            Document(page_content="norm", metadata={"source": "a.md", "chunk_id": 0})
        ]
        with pytest.raises(index.IndexingError, match="b.md"):
            index.main()
    backend.assert_not_called()
    assert (env[1] / "sentinel").exists()


def test_persisted_content_mismatch_blocks_readiness(env):
    import index

    store = MagicMock()
    store.count.return_value = 1
    store.iter_all_documents.return_value = iter(
        [{"text": "wrong norm", "metadata": {"source": "a.md", "chunk_id": 0}}]
    )
    with (
        patch.object(index, "DocumentProcessor") as processor,
        patch.object(index, "get_vector_store_backend", return_value=store),
    ):
        processor.return_value.process.return_value = [
            Document(page_content="norm", metadata={"source": "a.md", "chunk_id": 0})
        ]
        with pytest.raises(index.IndexingError):
            index.main()
    assert not list(env[1].parent.glob("active.snapshots/*/snapshot-report.json"))
    assert (env[1] / "sentinel").exists()


def test_real_cache_rebinds_identical_bytes_to_each_source(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path / "cache"
    processor.cache_dir.mkdir()
    files = [tmp_path / "a.pdf", tmp_path / "b.pdf"]
    for file in files:
        file.write_bytes(b"same document")
    with patch.object(processor, "_convert_and_extract") as convert:
        convert.return_value = [
            Document(page_content="shared norm", metadata={"source": "a.pdf"})
        ]
        chunks = processor.process(files)
    assert [(c.metadata["source"], c.metadata["chunk_id"]) for c in chunks] == [
        ("a.pdf", 0),
        ("b.pdf", 0),
    ]
    assert convert.call_count == 1


def test_strict_processing_rejects_empty_document(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    source = tmp_path / "empty.pdf"
    source.write_bytes(b"empty")
    with patch.object(processor, "_convert_and_extract", return_value=[]):
        with pytest.raises(ValueError, match="empty.pdf"):
            processor.process([source], strict=True)


def test_manifest_missing_required_file_fails_before_parsing(env):
    import index
    from src.indexing.manifest import Manifest

    manifest = Manifest("snapshot", "org", None, {"a.md": {}, "missing.pdf": {}})
    with (
        patch.object(index.settings, "CORPUS_MANIFEST_PATH", "manifest.yaml"),
        patch.object(index, "load_manifest", return_value=manifest),
        patch.object(index, "DocumentProcessor") as processor,
    ):
        with pytest.raises(index.IndexingError, match="missing.pdf"):
            index.main()
    processor.assert_not_called()


def test_manifest_excludes_profiles_and_unlisted_files(env):
    import index
    from src.indexing.manifest import Manifest

    for name in ("profile.md", "unlisted.md"):
        (env[0] / name).write_text("not indexed")
    manifest = Manifest(
        "snapshot", "org", None, {"a.md": {}, "profile.md": {}}, {"unit": "profile.md"}
    )
    store = MagicMock()
    store.count.return_value = 1
    chunk = Document(page_content="norm", metadata={"source": "a.md", "chunk_id": 0})
    store.iter_all_documents.return_value = iter(
        [{"text": chunk.page_content, "metadata": chunk.metadata}]
    )
    with (
        patch.object(index.settings, "CORPUS_MANIFEST_PATH", "manifest.yaml"),
        patch.object(index, "load_manifest", return_value=manifest),
        patch.object(index, "DocumentProcessor") as processor,
        patch.object(index, "get_vector_store_backend", return_value=store),
    ):
        processor.return_value.process.return_value = [chunk]
        index.main()
    processor.return_value.process.assert_called_once_with(
        [str(env[0] / "a.md")], strict=True
    )


def test_duplicate_basenames_rejected(env):
    import index

    nested = env[0] / "nested"
    nested.mkdir()
    (nested / "a.md").write_text("different edition")
    with patch.object(index, "DocumentProcessor") as processor:
        with pytest.raises(index.IndexingError, match="basenames"):
            index.main()
    processor.assert_not_called()


def test_cache_has_no_provenance_and_corruption_is_reparsed(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    source = tmp_path / "a.pdf"
    source.write_bytes(b"norm")
    with patch.object(processor, "_convert_and_extract") as convert:
        convert.return_value = [
            Document(
                page_content="norm",
                metadata={"source": "a.pdf", "chunk_id": 99, "type": "hybrid_chunk"},
            )
        ]
        processor.process([source], strict=True)
        cache = next(tmp_path.glob("*.json"))
        payload = json.loads(cache.read_text())
        assert payload["chunks"][0]["metadata"] == {"type": "hybrid_chunk"}
        cache.write_text("broken JSON")
        processor.process([source], strict=True)
    assert convert.call_count == 2


def test_cache_fingerprint_changes_with_dependency_version(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    with patch("src.indexing.file_handler.version", return_value="1"):
        first = processor._cache_path_for("same")
    with patch("src.indexing.file_handler.version", return_value="2"):
        assert processor._cache_path_for("same") != first


def test_processing_generator_and_failure_report(tmp_path):
    processor = object.__new__(DocumentProcessor)
    processor.cache_dir = tmp_path
    good = tmp_path / "good.pdf"
    missing = tmp_path / "missing.pdf"
    good.write_bytes(b"norm")
    with patch.object(
        processor, "_convert_and_extract", return_value=[Document(page_content="norm")]
    ):
        chunks = processor.process(iter([good, missing]))
    assert len(chunks) == 1
    assert len(processor.processing_report) == 2
    assert processor.processing_report[0].chunk_count == 1
    assert processor.processing_report[1].error is not None


def test_source_changed_during_parsing_rejected(env):
    import index

    def process(*args, **kwargs):
        (env[0] / "a.md").write_text("new edition")
        return [Document(page_content="old edition", metadata={"source": "a.md"})]

    with (
        patch.object(index, "DocumentProcessor") as processor,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        processor.return_value.process.side_effect = process
        with pytest.raises(index.IndexingError, match="changed"):
            index.main()
    backend.assert_not_called()
