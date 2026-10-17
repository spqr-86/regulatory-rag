"""Build snapshot-local parent windows offline, without opening Chroma clients."""

from __future__ import annotations

# ANCHOR: source_docs + read-only SQLite metadata → parent_context.json.
# Monotonic alignment: rollback=0, overlap threshold=0.60. Missing parents are omitted.
import argparse
import json
import socket
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.indexing.parent_context import (  # noqa: E402
    SIDECAR_NAME,
    ParentEntry,
    ParentSidecar,
    parent_window,
    snapshot_hash,
)
from src.indexing.parents.align import AlignInput, align_document  # noqa: E402
from src.indexing.parents.chunkers import (
    paragraphs_from_text,
    parse_units,
)  # noqa: E402
from src.indexing.parents.norms import Norm  # noqa: E402


def build_sidecar(snapshot: Path, source_docs: Path) -> ParentSidecar:
    digest = snapshot_hash(snapshot)
    grouped = defaultdict(list)
    rows = defaultdict(dict)
    with sqlite3.connect(
        f"{(snapshot / 'chroma.sqlite3').resolve().as_uri()}?mode=ro", uri=True
    ) as con:
        for i, key, value in con.execute(
            "select id,key,coalesce(string_value,cast(int_value as text)) from embedding_metadata "
            "where key in ('source','chunk_id','chroma:document','parent_section','article','part','heading_path') order by id"
        ):
            rows[i][key] = value
    for meta in rows.values():
        if all(k in meta for k in ("source", "chunk_id", "chroma:document")):
            grouped[meta["source"]].append(meta)
    entries = {}
    for source, chunks in sorted(grouped.items()):
        # Resolve only files directly inside the supplied source directory.
        if Path(source).name != source:
            continue
        path = source_docs / source
        if not path.is_file() or path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        if path.suffix.lower() == ".pdf":
            import pymupdf

            with pymupdf.open(path) as pdf:
                text = "\n".join(page.get_text() for page in pdf)
        else:
            text = path.read_text(encoding="utf-8")
        paras = paragraphs_from_text(text, wrapped=path.suffix.lower() == ".pdf")
        norms = [
            Norm(source, unit.kind, unit.number, tuple(unit.paragraphs), i)
            for i, unit in enumerate(parse_units(paras))
        ]
        chunks.sort(key=lambda meta: int(meta["chunk_id"]))
        inputs = []
        for meta in chunks:
            body = meta["chroma:document"]
            heading = meta.get("parent_section", "")
            if heading and body.startswith(heading):
                body = body[len(heading) :].lstrip("\n ")
            inputs.append(
                AlignInput(
                    body,
                    meta.get("article", ""),
                    meta.get("part", ""),
                    meta.get("heading_path", ""),
                )
            )
        aligned = align_document(inputs, norms, rollback=0, threshold=0.60)
        entries[source] = {
            str(meta["chunk_id"]): ParentEntry(
                parent_id=alignment.parent.id,
                text=parent_window(alignment.parent, meta["chroma:document"]),
                snapshot_sha256=digest,
            )
            for meta, alignment in zip(chunks, aligned)
            if alignment.parent is not None
        }
    if snapshot_hash(snapshot) != digest:
        raise RuntimeError("Snapshot changed while building parent sidecar")
    return ParentSidecar(version=2, snapshot_sha256=digest, entries=entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source-docs", type=Path, default=Path("source_docs"))
    args = parser.parse_args()

    def offline(*args, **kwargs):
        raise RuntimeError("Network is disabled during parent sidecar build")

    socket.socket = offline
    socket.create_connection = offline
    result = build_sidecar(args.snapshot, args.source_docs)
    destination = args.snapshot / SIDECAR_NAME
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(
        json.dumps(
            {
                "path": str(destination),
                "parents": sum(len(v) for v in result.entries.values()),
                "sources": len(result.entries),
                "snapshot_sha256": result.snapshot_sha256,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
