"""Регрессия: index.main() должен инвалидировать BM25-cache и Docling-cache.

Без этого после destructive reindex поиск идёт по удалённым чанкам.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def fake_caches(tmp_path, monkeypatch):
    """Подменяет CACHE_DIR/CHROMA_DB_PATH/cwd и создаёт фейковые кэши."""
    cache_dir = tmp_path / "document_cache"
    chroma_dir = tmp_path / "chroma_db"
    src_dir = tmp_path / "source_docs"
    cache_dir.mkdir()
    chroma_dir.mkdir()
    src_dir.mkdir()
    (cache_dir / "fake.pkl").write_bytes(b"stale")
    bm25 = tmp_path / ".bm25_cache.pkl"
    bm25.write_bytes(b"stale")

    from config.settings import settings

    monkeypatch.setattr(settings, "CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(chroma_dir))
    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(src_dir))
    monkeypatch.chdir(tmp_path)
    return {"cache_dir": cache_dir, "chroma": chroma_dir, "bm25": bm25}


def test_main_clears_bm25_and_docling_cache(fake_caches):
    import index

    with patch.object(index, "get_vector_store_backend"):
        index.main()

    assert not fake_caches["cache_dir"].exists(), "Docling cache must be removed"
    assert not fake_caches["bm25"].exists(), "BM25 cache must be removed"


def test_main_handles_missing_caches_gracefully(tmp_path, monkeypatch):
    """Если кэшей нет — main() не должен падать."""
    from config.settings import settings

    src_dir = tmp_path / "source_docs"
    src_dir.mkdir()
    monkeypatch.setattr(settings, "CACHE_DIR", str(tmp_path / "nope_cache"))
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(tmp_path / "nope_chroma"))
    monkeypatch.setattr(settings, "SOURCE_DOCS_PATH", str(src_dir))
    monkeypatch.chdir(tmp_path)

    import index

    with patch.object(index, "get_vector_store_backend"):
        index.main()  # should not raise
