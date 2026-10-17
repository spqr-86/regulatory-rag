"""Snapshot-bound parent windows, loaded only for opt-in context packing."""

from __future__ import annotations

# ANCHOR: snapshot sidecar + child passages → stable, source-local parent dedup.
# Invalid/missing sidecars preserve children. Cache identity includes file revisions.
# Alignment is offline; request handling never reads source documents or calls models.
import copy
import hashlib
import logging
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


class ParentSidecar(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    version: int
    snapshot_sha256: str
    entries: dict[str, dict[str, ParentEntry]]


def active_snapshot() -> Path:
    from config.settings import settings

    return Path(settings.CHROMA_DB_PATH).resolve()


def snapshot_hash(snapshot: Path) -> str:
    """Bind to the stored SQLite snapshot, including an outstanding WAL if any."""
    digest = hashlib.sha256()
    for path in (snapshot / "chroma.sqlite3", snapshot / "chroma.sqlite3-wal"):
        if path.name.endswith("-wal") and not path.exists():
            continue
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def _revision(path: Path) -> tuple:
    try:
        stat = path.stat()
        return stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
    except OSError:
        return ()


def _identity() -> tuple:
    snapshot = active_snapshot()
    return (
        str(snapshot),
        *(
            _revision(snapshot / name)
            for name in (SIDECAR_NAME, "chroma.sqlite3", "chroma.sqlite3-wal")
        ),
    )


@lru_cache(maxsize=8)
def _load_sidecar(identity: tuple) -> ParentSidecar | None:
    snapshot = Path(identity[0])
    try:
        sidecar = ParentSidecar.model_validate_json(
            (snapshot / SIDECAR_NAME).read_bytes()
        )
        if sidecar.version != 1 or sidecar.snapshot_sha256 != snapshot_hash(snapshot):
            raise ValueError("snapshot hash/version mismatch")
        if any(
            entry.snapshot_sha256 != sidecar.snapshot_sha256
            for entries in sidecar.entries.values()
            for entry in entries.values()
        ):
            raise ValueError("entry snapshot hash mismatch")
        return sidecar
    except (OSError, ValueError, ValidationError) as exc:
        logger.warning("parent context sidecar unavailable for %s: %s", snapshot, exc)
        return None


def parent_cache_version(max_chars: int) -> str:
    return repr(("parent-context-v1", max_chars, _identity()))


def replace_parents(passages: list[dict], max_chars: int = 3000) -> list[dict]:
    """Replace known children without mutating input; retain the first parent window."""
    sidecar = _load_sidecar(_identity())
    if sidecar is None:
        return passages
    result, seen = [], set()
    for passage in passages:
        meta = passage.get("metadata") or {}
        source = meta.get("source") or passage.get("doc_id", "")
        cid = passage.get("chunk_id", meta.get("chunk_id"))
        entry = sidecar.entries.get(source, {}).get(str(cid))
        if entry is None:
            result.append(passage)
            continue
        key = (source, entry.parent_id)
        if key in seen:
            continue
        seen.add(key)
        parent = copy.deepcopy(passage)
        parent["text"] = entry.text[: min(max_chars, 3000)]
        parent["metadata"] = {
            **meta,
            "child_chunk_id": cid,
            "parent_id": entry.parent_id,
            "child_score": passage.get("score"),
        }
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
