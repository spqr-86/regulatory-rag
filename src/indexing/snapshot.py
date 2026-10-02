"""Readiness contract for managed corpus snapshots."""

# ANCHOR: validate report, collection/model identity and persisted corpus checksum.
# Legacy stores have no report; managed candidates require a validated schema-2 report.
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingIdentity(BaseModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)


class SnapshotReport(BaseModel):
    schema_version: Literal[2]
    status: Literal["validated"]
    snapshot_path: str = Field(min_length=1)
    collection: str = Field(min_length=1)
    chunk_count: int = Field(gt=0)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding: EmbeddingIdentity


def record_digest(text: str, metadata: dict) -> str:
    record = json.dumps([text, metadata], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(record.encode()).hexdigest()


def counter_digest(records: Counter) -> str:
    return hashlib.sha256(json.dumps(sorted(records.items())).encode()).hexdigest()


def _canonical_model(model: str) -> str:
    return model.removeprefix("openai/")


def read_snapshot_report(
    path: str, collection: str, embeddings
) -> SnapshotReport | None:
    root = Path(path)
    report_path = root / "snapshot-report.json"
    managed = (
        root.parent.name.endswith(".snapshots")
        or (root / "snapshot-build.json").exists()
    )
    if not report_path.exists():
        if managed:
            raise ValueError(f"Snapshot is not ready: {path}")
        return None
    report = SnapshotReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if (
        Path(report.snapshot_path).resolve() != root.resolve()
        or report.collection != collection
    ):
        raise ValueError("Snapshot path or collection differs from readiness report")
    model = getattr(embeddings, "model", None)
    if not isinstance(model, str):
        model = getattr(embeddings, "model_name", None)
    if not isinstance(model, str) or _canonical_model(model) != _canonical_model(
        report.embedding.model
    ):
        raise ValueError("Embedding model differs from readiness report")
    return report


def validate_stored_snapshot(store, report: SnapshotReport) -> None:
    if store._collection.count() != report.chunk_count:
        raise ValueError("Snapshot count differs from readiness report")
    records = Counter()
    offset = 0
    while offset < report.chunk_count:
        page = store.get(include=["documents", "metadatas"], limit=1000, offset=offset)
        texts = page.get("documents") or []
        metadata = page.get("metadatas") or []
        if not texts or len(texts) != len(metadata):
            raise ValueError("Snapshot contains missing text or metadata")
        records.update(
            record_digest(text, meta or {}) for text, meta in zip(texts, metadata)
        )
        offset += len(texts)
    if offset != report.chunk_count or counter_digest(records) != report.content_digest:
        raise ValueError("Snapshot content differs from readiness report")
