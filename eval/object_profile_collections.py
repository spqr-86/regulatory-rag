"""Check that object sheets are in the v1 collection and absent from the v2 one.

Two channels, because a leak can hide in either (spec object-profile §5.2 step 4):
stored chunks in Chroma (by source) and hybrid search results (vector + BM25) for
queries taken from the sheets, with each sheet's unit scope filter.

Usage::

    CHROMA_DB_PATH=./chroma_db_dept CHROMA_COLLECTION_NAME=department_demo \\
    CORPUS_MANIFEST_PATH=corpus/manifest.yaml \\
        .venv/bin/python eval/object_profile_collections.py --expect present
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

QUERIES = (
    "Лист особенностей объекта защиты",
    "Дежурный персонал объекта",
    "Первичные средства пожаротушения на объекте",
)


def sheet_sources(passages: Iterable[dict], sheet_files: set[str]) -> set[str]:
    found = set()
    for p in passages:
        name = os.path.basename((p.get("metadata") or {}).get("source", ""))
        if name in sheet_files:
            found.add(name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("present", "absent"), required=True)
    args = parser.parse_args()

    from config.settings import settings
    from src.backends.vector_store import get_vector_store_backend
    from src.department_qa.wiring import make_hybrid_search_fn
    from src.indexing.manifest import load_manifest
    from src.v7.bridge import init_v7_pipeline
    from src.v7.scope_filter import build_scope_filters

    manifest = load_manifest(settings.CORPUS_MANIFEST_PATH)
    sheets = {
        name: meta["unit_id"]
        for name, meta in manifest.documents.items()
        if name.startswith("int_unit_") and name.endswith("_list.md")
    }
    store = get_vector_store_backend(load_existing=True)
    stored = list(store.iter_all_documents())
    in_chroma = sheet_sources(stored, set(sheets))

    init_v7_pipeline(store, llm_provider=None)
    search = make_hybrid_search_fn(store)
    in_search: set[str] = set()
    for name, unit_id in sheets.items():
        _, internal = build_scope_filters(unit_id)
        for query in QUERIES:
            in_search |= sheet_sources(search(query, filters=internal, top_k=8), {name})

    report = {
        "chroma_db_path": settings.CHROMA_DB_PATH,
        "collection": settings.CHROMA_COLLECTION_NAME,
        "chunks": len(stored),
        "sheets": sorted(sheets),
        "in_chroma": sorted(in_chroma),
        "in_hybrid_search": sorted(in_search),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.expect == "present":
        ok = in_chroma == set(sheets) and in_search == set(sheets)
    else:
        ok = not in_chroma and not in_search
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
