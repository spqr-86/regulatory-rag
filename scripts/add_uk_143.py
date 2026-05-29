"""Incrementally add Статья 143 УК РФ to existing ChromaDB without full reindex.

Adds one new markdown file to the existing index using DocumentProcessor,
without dropping the database.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.backends.vector_store import get_vector_store_backend
from src.indexing.file_handler import DocumentProcessor


def main() -> None:
    file_path = str(
        Path(__file__).parent.parent / "source_docs" / "Статья 143 УК РФ.md"
    )

    print(f"Processing {file_path}…")
    processor = DocumentProcessor()
    chunks = processor.process([file_path])

    if not chunks:
        print("ERROR: no chunks produced")
        return

    print(f"Produced {len(chunks)} chunks")

    backend = get_vector_store_backend(load_existing=True)
    print(
        f"Existing documents: {backend.count() if hasattr(backend, 'count') else '?'}"
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    if hasattr(backend, "add_texts"):
        backend.add_texts(texts=texts, metadatas=metadatas)
        print(f"Added {len(chunks)} chunks via backend.add_texts()")
    else:
        print("ERROR: backend has no add_texts method")
        return

    final_count = backend.count() if hasattr(backend, "count") else "?"
    print(f"Final documents: {final_count}")


if __name__ == "__main__":
    main()
