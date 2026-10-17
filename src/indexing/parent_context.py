"""Snapshot-bound parent windows, loaded only for opt-in context packing."""

from __future__ import annotations

# ANCHOR: snapshot sidecar + child passages → stable, source-local parent dedup.
# Invalid/missing sidecars preserve children. Cache identity includes file revisions.
# Alignment is offline; request handling never reads source documents or calls models.
import copy
import hashlib
import json
import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.indexing.parents.norms import Norm, words

logger = logging.getLogger(__name__)
SIDECAR_NAME = "parent_context.json"


class ParentEntry(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    parent_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=3000)
    snapshot_sha256: str
    # Old v2 sidecars lack this bit: their precomputed window loss is unknown.
    truncated: bool = False


class ParentSidecar(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    version: int
    snapshot_sha256: str
    entries: dict[str, dict[str, ParentEntry]]


def active_snapshot() -> Path:
    from config.settings import settings

    return Path(settings.CHROMA_DB_PATH).resolve()


def snapshot_hash(snapshot: Path) -> str:
    """Hash logical chunks, including text; ignore SQLite layout and bookkeeping.

    One read-only SELECT observes a consistent SQLite snapshot, including WAL.
    Sorted records retain duplicates and the count, independent of internal row IDs.
    """
    with closing(
        sqlite3.connect(
            f"{(snapshot / 'chroma.sqlite3').resolve().as_uri()}?mode=ro", uri=True
        )
    ) as con:
        rows = con.execute(
            "select max(case when key='source' then string_value end), "
            "max(case when key='chunk_id' then coalesce(string_value,cast(int_value as text)) end), "
            "max(case when key='chroma:document' then string_value end) "
            "from embedding_metadata group by id"
        ).fetchall()
    records = sorted(json.dumps(row, ensure_ascii=False) for row in rows)
    return hashlib.sha256(
        json.dumps([len(records), records], ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _revision(path: Path) -> tuple:
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return ()


def _identity() -> tuple:
    snapshot = active_snapshot()
    try:
        digest = snapshot_hash(snapshot)
    except (OSError, sqlite3.Error) as exc:
        digest = f"unavailable: {exc}"
    return str(snapshot), digest, _revision(snapshot / SIDECAR_NAME)


@dataclass(frozen=True)
class SidecarLoad:
    sidecar: ParentSidecar | None
    reason: str | None = None


@dataclass
class ParentStats:
    # Replacement counts precede pack limits; slots_lost is set after packing.
    # truncated counts emitted parent windows, once per source/parent, including
    # known build-time clipping. Legacy sidecars expose only runtime clipping.
    replaced: int = 0
    children: int = 0
    deduplicated: int = 0
    truncated: int = 0
    slots_lost: int = 0
    unavailable_reason: str | None = None


@lru_cache(maxsize=8)
def _load_sidecar(identity: tuple) -> SidecarLoad:
    snapshot = Path(identity[0])
    try:
        sidecar = ParentSidecar.model_validate_json(
            (snapshot / SIDECAR_NAME).read_bytes()
        )
        if sidecar.version != 2:
            raise ValueError(
                f"sidecar version {sidecar.version}: пересобрать sidecar (requires version 2)"
            )
        if identity[1].startswith("unavailable:"):
            raise ValueError(f"collection {identity[1]}")
        if sidecar.snapshot_sha256 != identity[1]:
            raise ValueError("collection fingerprint mismatch: пересобрать sidecar")
        if any(
            entry.snapshot_sha256 != sidecar.snapshot_sha256
            for entries in sidecar.entries.values()
            for entry in entries.values()
        ):
            raise ValueError("entry snapshot hash mismatch: пересобрать sidecar")
        return SidecarLoad(sidecar)
    except (OSError, ValueError, ValidationError) as exc:
        reason = str(exc)
        logger.warning(
            "parent context sidecar unavailable for %s: %s", snapshot, reason
        )
        return SidecarLoad(None, reason)


def parent_cache_version(max_chars: int) -> str:
    return repr(("parent-context-v2", max_chars, _identity()))


def replace_parents(
    passages: list[dict], max_chars: int = 3000, *, stats: ParentStats | None = None
) -> list[dict]:
    """Replace known children; optionally report stage counts without mutating input."""
    stats = stats if stats is not None else ParentStats()
    loaded = _load_sidecar(_identity())
    sidecar = loaded.sidecar
    if sidecar is None:
        stats.children += len(passages)
        stats.unavailable_reason = loaded.reason
        return passages
    result, seen = [], {}
    for passage in passages:
        meta = passage.get("metadata") or {}
        source = meta.get("source") or passage.get("doc_id", "")
        cid = passage.get("chunk_id", meta.get("chunk_id"))
        entry = sidecar.entries.get(source, {}).get(str(cid))
        if entry is None:
            stats.children += 1
            result.append(passage)
            continue
        key = (source, entry.parent_id)
        if key in seen:
            stats.deduplicated += 1
            children = seen[key]["metadata"]["child_chunk_ids"]
            if cid not in children:
                children.append(cid)
            continue
        stats.replaced += 1
        parent = copy.deepcopy(passage)
        cap = min(max_chars, 3000)
        stats.truncated += int(entry.truncated or len(entry.text) > cap)
        parent["text"] = entry.text[:cap]
        parent["metadata"] = {
            **meta,
            "child_chunk_id": cid,
            "child_chunk_ids": [cid],
            "parent_id": entry.parent_id,
            "child_score": passage.get("score"),
        }
        seen[key] = parent
        result.append(parent)
    return result


def parent_window(norm: Norm, body: str, cap: int = 3000) -> str:
    """Step4 window policy: strongest paragraph, then alternate left/right growth."""
    if len(norm.text) <= cap:
        return norm.text
    target = words(body)
    scores = [len(words(p) & target) for p in norm.paragraphs]
    first = max(range(len(scores)), key=scores.__getitem__)
    lo, hi = first, first + 1
    size = len(norm.paragraphs[first])
    if size > cap:
        # One paragraph can exceed the cap: retain the strongest bounded fragment.
        paragraph = norm.paragraphs[first]
        pieces = [paragraph[i : i + cap] for i in range(0, len(paragraph), cap)]
        return max(pieces, key=lambda text: len(words(text) & target))
    while True:
        grew = False
        if lo > 0 and size + len(norm.paragraphs[lo - 1]) + 1 <= cap:
            lo -= 1
            size += len(norm.paragraphs[lo]) + 1
            grew = True
        if hi < len(norm.paragraphs) and size + len(norm.paragraphs[hi]) + 1 <= cap:
            size += len(norm.paragraphs[hi]) + 1
            hi += 1
            grew = True
        if not grew:
            return "\n".join(norm.paragraphs[lo:hi])
