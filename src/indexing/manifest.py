"""Corpus manifest: per-document metadata for the department Q&A snapshot (issue #36).

Spec §5: every chunk carries source_type (external|internal); internal documents
also carry organization_id, scope (company|unit) and unit_id for scope=unit.
A file without a manifest entry has unknown applicability and is excluded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml
from langchain_core.documents import Document

SOURCE_TYPES = {"external", "internal"}
SCOPES = {"company", "unit"}


class ManifestError(ValueError):
    """Manifest is malformed; the snapshot must not be built from it."""


@dataclass(frozen=True)
class Manifest:
    snapshot_id: str
    organization_id: str
    checked_at: str | None
    documents: dict[str, dict]  # basename -> chunk metadata to merge


def _entry_metadata(entry: dict, snapshot_id: str, organization_id: str) -> dict:
    for key in ("file", "document_id", "source_type", "title"):
        if not entry.get(key):
            raise ManifestError(f"entry {entry!r}: missing {key}")
    source_type = entry["source_type"]
    if source_type not in SOURCE_TYPES:
        raise ManifestError(f"{entry['file']}: source_type={source_type!r}")

    meta = {
        "document_id": entry["document_id"],
        "source_type": source_type,
        "title": entry["title"],
        # One edition per snapshot (spec П5).
        "version_id": snapshot_id,
        "snapshot_id": snapshot_id,
    }
    for optional in ("locator", "source_ref"):
        if entry.get(optional):
            meta[optional] = entry[optional]

    if source_type == "internal":
        scope = entry.get("scope")
        if scope not in SCOPES:
            raise ManifestError(f"{entry['file']}: internal needs scope, got {scope!r}")
        meta["organization_id"] = organization_id
        meta["scope"] = scope
        if scope == "unit":
            if not entry.get("unit_id"):
                raise ManifestError(f"{entry['file']}: scope=unit needs unit_id")
            meta["unit_id"] = entry["unit_id"]
    return meta


def load_manifest(path: str | Path) -> Manifest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    snapshot_id = data.get("snapshot_id")
    organization_id = data.get("organization_id")
    if not snapshot_id or not organization_id:
        raise ManifestError("snapshot_id and organization_id are required")

    documents: dict[str, dict] = {}
    seen_ids: set[str] = set()
    for entry in data.get("documents") or []:
        meta = _entry_metadata(entry, snapshot_id, organization_id)
        name = os.path.basename(entry["file"])
        if name in documents or meta["document_id"] in seen_ids:
            raise ManifestError(f"duplicate file or document_id: {entry['file']}")
        documents[name] = meta
        seen_ids.add(meta["document_id"])

    checked_at = data.get("checked_at")
    return Manifest(
        snapshot_id=snapshot_id,
        organization_id=organization_id,
        checked_at=str(checked_at) if checked_at else None,
        documents=documents,
    )


def apply_manifest(chunks: List[Document], manifest: Manifest) -> List[Document]:
    """Merge manifest metadata into chunks; drop chunks of unlisted files."""
    kept: List[Document] = []
    for ch in chunks:
        meta = manifest.documents.get(os.path.basename(ch.metadata.get("source", "")))
        if meta is None:
            continue
        ch.metadata.update(meta)
        kept.append(ch)
    return kept
