"""Build and validate an isolated corpus snapshot; activation is a separate step."""

# ANCHOR: source files -> strict parsing -> isolated Chroma -> verified snapshot.
# CHROMA_DB_PATH identifies the active store; it is never deleted or overwritten.
# Successful main() returns a candidate path with snapshot-report.json.
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from config.settings import settings
from src.backends.vector_store import get_vector_store_backend
from src.indexing.file_handler import (
    DocumentProcessor,
    ProcessingError,
    pipeline_fingerprint,
)
from src.indexing.manifest import ManifestError, apply_manifest, load_manifest
from src.indexing.snapshot import counter_digest
from src.indexing.vector_store import _sanitize_metadata
from utils.logging import logger

load_dotenv()


def _collect_paths(root_dir: str, allowed_exts: list[str]) -> list[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if os.path.splitext(name)[1].lower() in allowed_exts:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


class IndexingError(RuntimeError):
    """Snapshot build rejected; the active index remains untouched."""


def _record_digest(text: str, metadata: dict) -> str:
    record = json.dumps([text, metadata], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(record.encode()).hexdigest()


def _source_hash(path: str) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> Path:
    logger.info("Starting isolated snapshot build...")
    if not settings.EMBEDDING_MODEL_NAME.strip():
        raise IndexingError(
            "An explicit embedding model is required for reproducible snapshots"
        )
    file_paths = _collect_paths(settings.SOURCE_DOCS_PATH, settings.ALLOWED_TYPES)
    manifest = (
        load_manifest(settings.CORPUS_MANIFEST_PATH)
        if settings.CORPUS_MANIFEST_PATH
        else None
    )
    names = [Path(p).name for p in file_paths]
    if len(names) != len(set(names)):
        raise IndexingError("Duplicate source basenames; provenance would be ambiguous")
    if manifest is not None:
        required = set(manifest.documents) - set(manifest.object_profiles.values())
        missing = required - set(names)
        if missing:
            raise IndexingError(f"Missing required documents: {sorted(missing)}")
        file_paths = [p for p in file_paths if Path(p).name in required]
    else:
        required = set(names)
    if not file_paths:
        raise IndexingError(
            f"No suitable required files found in {settings.SOURCE_DOCS_PATH}"
        )

    hashes = {Path(p).name: _source_hash(p) for p in file_paths}
    processor = DocumentProcessor()
    try:
        chunks = processor.process(file_paths, strict=True)
    except ProcessingError as e:
        raise IndexingError(str(e)) from e
    if manifest is not None:
        chunks = apply_manifest(chunks, manifest)
    produced = Counter(ch.metadata.get("source") for ch in chunks)
    if not chunks or required != set(produced):
        raise IndexingError(
            f"Incomplete corpus; missing={sorted(required - set(produced))}"
        )
    for p in file_paths:
        if _source_hash(p) != hashes[Path(p).name]:
            raise IndexingError(f"Source changed during parsing: {p}")

    # Never mutate process-global settings or move directories with open Chroma clients.
    active = Path(settings.CHROMA_DB_PATH).absolute()
    snapshots = active.parent / (active.name + ".snapshots")
    snapshots.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="build-", dir=snapshots))
    (target / "snapshot-build.json").write_text(
        '{"status":"building","schema_version":2}', encoding="utf-8"
    )
    backend = get_vector_store_backend(load_existing=False)
    backend.create(chunks, path=str(target))
    expected = Counter(
        _record_digest(ch.page_content, _sanitize_metadata(ch.metadata))
        for ch in chunks
    )
    stored = Counter(
        _record_digest(row["text"], row["metadata"])
        for row in backend.iter_all_documents()
    )
    if backend.count() != len(chunks) or stored != expected:
        raise IndexingError(f"Stored snapshot differs from parsed corpus: {target}")

    # The readiness report appears ONLY after content and provenance validation.
    report = {
        "schema_version": 2,
        "status": "validated",
        "active_path": str(active),
        "snapshot_path": str(target),
        "collection": settings.CHROMA_COLLECTION_NAME,
        "snapshot_id": manifest.snapshot_id if manifest else None,
        "chunk_count": len(chunks),
        "content_digest": counter_digest(expected),
        "pipeline": pipeline_fingerprint(),
        "embedding": {
            "provider": settings.EMBEDDING_PROVIDER,
            "model": settings.EMBEDDING_MODEL_NAME,
        },
        "documents": [
            {"source": name, "sha256": hashes[name], "chunk_count": produced[name]}
            for name in sorted(required)
        ],
    }
    pending_report = target / ".snapshot-report.tmp"
    pending_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pending_report.replace(target / "snapshot-report.json")
    logger.info(
        f"Snapshot validated: {target}. Active index unchanged. Activate via the target consumer's path/collection settings (docs/getting-started.md)."
    )
    return target


if __name__ == "__main__":
    try:
        main()
    except (IndexingError, ManifestError) as e:
        logger.error(str(e))
        sys.exit(1)
