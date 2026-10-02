"""Regression: isolated builds preserve parser caches and the active index.

BM25 is rebuilt from the loaded snapshot; parser cache keys fingerprint the
pipeline and cached structural chunks are rebound to each input source.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

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
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", "")
    monkeypatch.chdir(tmp_path)
    return {"cache_dir": cache_dir, "chroma": chroma_dir, "src": src_dir, "bm25": bm25}


def _chunks():
    return [Document(page_content="norm", metadata={"source": "doc.md", "chunk_id": 0})]


def _rows():
    return [{"text": c.page_content, "metadata": c.metadata} for c in _chunks()]


def _add_doc(src_dir):
    (src_dir / "doc.md").write_text("# Норма\nтекст", encoding="utf-8")


def test_main_preserves_bm25_and_docling_cache(fake_env):
    import index

    _add_doc(fake_env["src"])
    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        proc.return_value.process.return_value = _chunks()
        backend.return_value.count.return_value = 1
        backend.return_value.iter_all_documents.return_value = iter(_rows())
        index.main()

    assert (fake_env["cache_dir"] / "fake.pkl").read_bytes() == b"stale"
    assert fake_env["bm25"].read_bytes() == b"stale"


def test_main_handles_missing_caches_gracefully(tmp_path, monkeypatch):
    """If caches and index are absent — main() must not raise on valid input."""
    from config.settings import settings

    src_dir = tmp_path / "source_docs"
    src_dir.mkdir()
    _add_doc(src_dir)
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "nope_cache"))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path / "nope_chroma"))
    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(src_dir))
    monkeypatch.setattr(settings, "CORPUS_MANIFEST_PATH", "")
    monkeypatch.chdir(tmp_path)

    import index

    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        proc.return_value.process.return_value = _chunks()
        backend.return_value.count.return_value = 1
        backend.return_value.iter_all_documents.return_value = iter(_rows())
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


def test_index_preserved_during_and_after_build(fake_env):
    import index

    _add_doc(fake_env["src"])
    seen = {}

    def _process(paths, *, strict):
        assert strict is True
        seen["index_alive_during_processing"] = (
            fake_env["chroma"] / "chroma.sqlite3"
        ).exists()
        return _chunks()

    with (
        patch.object(index, "DocumentProcessor") as proc,
        patch.object(index, "get_vector_store_backend") as backend,
    ):
        proc.return_value.process.side_effect = _process
        backend.return_value.count.return_value = 1
        backend.return_value.iter_all_documents.return_value = iter(_rows())
        snapshot = index.main()

    assert seen["index_alive_during_processing"] is True
    backend.return_value.create.assert_called_once_with(_chunks(), path=str(snapshot))
    assert (fake_env["chroma"] / "chroma.sqlite3").read_bytes() == b"existing index"
