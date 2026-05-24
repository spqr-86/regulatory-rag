"""One-time migration: convert document_cache/*.pkl -> *.json.

Run MANUALLY after CARD-2.1 lands and verified working:
    python scripts/migrate_cache_to_json.py
"""

from __future__ import annotations

import json
import pickle  # noqa: S403 — intentional: migrating trusted local cache
from pathlib import Path

from src.indexing.file_handler import _document_to_dict

CACHE_DIR = Path("document_cache")


def main() -> int:
    converted = 0
    failed = 0
    for pkl_path in CACHE_DIR.glob("*.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)  # UNSAFE — only run on trusted cache dir
            chunks = data.chunks
            payload = {
                "schema_version": 1,
                "timestamp": data.timestamp,
                "chunks": [_document_to_dict(c) for c in chunks],
            }
            json_path = pkl_path.with_suffix(".json")
            json_path.write_text(json.dumps(payload, ensure_ascii=False))
            pkl_path.unlink()
            converted += 1
        except Exception as exc:
            print(f"FAIL {pkl_path.name}: {exc}")
            failed += 1
    print(f"Converted {converted}, failed {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
