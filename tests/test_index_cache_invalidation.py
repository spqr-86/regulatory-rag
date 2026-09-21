"""Regression: index.main() must invalidate BM25-cache and Docling-cache,
and must never drop an existing index before new chunks are ready (issue #32).

Without cache invalidation, after a destructive reindex search operates on
deleted chunks. Without input validation, an empty or missing source dir
deletes the index and leaves nothing in its place.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Replaces CACHE_DIR/CHROMA_DB_PATH/SOURCE_DOCS_PATH/cwd, creates fake caches and index."""
    cache_dir = tmp_path / "document_cache"
    chroma_dir = tmp_path / "chroma_db"
    src_dir = tmp_path / "source_docs"
    cache_dir.mkdir()
    chroma_dir.mkdir()
    src_dir.mkdir()
    (cache_dir / "fake.pkl").write_bytes(b"stale")
    (chroma_dir / "chroma.sqlite3").write_bytes(b"existing index")
    bm25 = tmp_path / ".bm25_cache.pkl"
    bm25.write_bytes(b"stale")

    from config.settings import settings

    monkeypatch.setattr(settings, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(chroma_dir))
    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(src_dir))
    monkeypatch.chdir(tmp_path)
    return {"cache_dir": cache_dir, "chroma": chroma_dir, "src": src_dir, "bm25": bm25}


def _add_doc(src_dir):
    (src_dir / "doc.md").write_text("# Норма\nтекст", encoding="utf-8")


def test_main_clears_bm25_and_docling_cache(fake_env):
    import index

    _add_doc(fake_env["src"])
    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend"),
    ):
        proc.return_value.process.return_value = ["chunk"]
        index.main()

    assert not fake_env["cache_dir"].exists(), "Docling cache must be removed"
    assert not fake_env["bm25"].exists(), "BM25 cache must be removed"


def test_main_handles_missing_caches_gracefully(tmp_path, monkeypatch):
    """If caches and index are absent — main() must not raise on valid input."""
    from config.settings import settings

    src_dir = tmp_path / "source_docs"
    src_dir.mkdir()
    _add_doc(src_dir)
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "nope_cache"))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path / "nope_chroma"))
    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(src_dir))
    monkeypatch.chdir(tmp_path)

    import index

    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend"),
    ):
        proc.return_value.process.return_value = ["chunk"]
        index.main()  # should not raise


def test_empty_source_dir_keeps_existing_index(fake_env):
    import index

    with patch.object(index, "get_vector_store_backend") as backend:
        with pytest.raises(index.IndexingError):
            index.main()

    assert (fake_env["chroma"] / "chroma.sqlite3").exists(), "index must survive"
    backend.assert_not_called()


def test_missing_source_dir_keeps_existing_index(fake_env, monkeypatch):
    import index
    from config.settings import settings

    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(fake_env["src"] / "absent"))
    with patch.object(index, "get_vector_store_backend") as backend:
        with pytest.raises(index.IndexingError):
            index.main()

    assert (fake_env["chroma"] / "chroma.sqlite3").exists(), "index must survive"
    backend.assert_not_called()


def test_no_chunks_keeps_existing_index(fake_env):
    import index

    _add_doc(fake_env["src"])
    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        proc.return_value.process.return_value = []
        with pytest.raises(index.IndexingError):
            index.main()

    assert (fake_env["chroma"] / "chroma.sqlite3").exists(), "index must survive"
    backend.assert_not_called()


def test_index_dropped_only_after_chunks_ready(fake_env):
    import index

    _add_doc(fake_env["src"])
    seen = {}

    def _process(paths):
        seen["index_alive_during_processing"] = (
            fake_env["chroma"] / "chroma.sqlite3"
        ).exists()
        return ["chunk"]

    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        proc.return_value.process.side_effect = _process
        index.main()

    assert seen["index_alive_during_processing"] is True
    backend.return_value.create.assert_called_once_with(["chunk"])
