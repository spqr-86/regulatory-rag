# index.py
import os
import shutil
import sys

from dotenv import load_dotenv

from config.settings import settings
from src.backends.vector_store import get_vector_store_backend
from src.indexing.file_handler import DocumentProcessor
from utils.logging import logger

load_dotenv()


def _collect_paths(root_dir: str, allowed_exts: list[str]) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed_exts:
                paths.append(os.path.join(dirpath, name))
    return paths


class IndexingError(RuntimeError):
    """Reindex aborted before touching the existing index."""


def main():
    logger.info("Starting indexing...")

    # Validate inputs first: an empty or missing source dir must not cost the
    # existing index (issue #32).
    file_paths = _collect_paths(settings.SOURCE_DOCS_PATH, settings.ALLOWED_TYPES)
    if not file_paths:
        raise IndexingError(
            f"No suitable files found in {settings.SOURCE_DOCS_PATH}; "
            "existing index left untouched."
        )

    # Invalidate caches tied to index contents.
    # Without this, BM25/Docling caches survive a destructive reindex and
    # search operates on ghost chunks from the deleted collection.
    if os.path.exists(settings.CACHE_DIR):
        logger.info(f"Clearing Docling cache: {settings.CACHE_DIR}")
        shutil.rmtree(settings.CACHE_DIR, ignore_errors=True)
    # .bm25_cache.pkl no longer produced (CARD-2.1b: BM25 rebuilt on startup).
    # Kept here as a no-op cleanup for any stale files from older deployments.
    bm25_cache = ".bm25_cache.pkl"
    if os.path.exists(bm25_cache):
        logger.info(f"Removing stale BM25 cache: {bm25_cache}")
        os.remove(bm25_cache)

    processor = DocumentProcessor()
    chunks = processor.process(file_paths)
    if not chunks:
        raise IndexingError(
            "No chunks produced. Check documents/conversion; "
            "existing index left untouched."
        )

    # Drop the old index only once replacement chunks are ready.
    if os.path.exists(settings.CHROMA_DB_PATH):
        logger.info(f"Dropping existing DB: {settings.CHROMA_DB_PATH}...")
        shutil.rmtree(settings.CHROMA_DB_PATH, ignore_errors=True)

    logger.info(f"Indexing {len(chunks)} chunks...")
    # Embed chunks and persist to ChromaDB (destructive: drops old collection).
    get_vector_store_backend(load_existing=False).create(chunks)
    logger.info("Indexing complete.")


if __name__ == "__main__":
    try:
        main()
    except IndexingError as e:
        logger.error(str(e))
        sys.exit(1)
