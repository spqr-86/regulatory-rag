"""Build trimmed evaluate_complex fixtures from a `langsmith run get --full` dump.

Usage: python scripts/make_merge_fixtures.py <run.out> <fixture.json>

Keeps only what `_candidates` reads: identities and scores. Passage text and
other internal fields are dropped — the repository is public.
"""

import json
import sys

PASSAGE_KEYS = (
    "chunk_id",
    "doc_id",
    "vector_score",
    "rerank_score",
    "bm25_score",
    "rrf_score",
    "score",
)
META_KEYS = ("source", "chunk_id", "page_no")


def trim_passage(p: dict) -> dict:
    out = {k: p[k] for k in PASSAGE_KEYS if k in p}
    meta = p.get("metadata") or {}
    out["metadata"] = {k: meta[k] for k in META_KEYS if k in meta}
    return out


def trim(inputs: dict) -> dict:
    attempts = [
        {
            "stage": a.get("stage"),
            "attempt_plan": a.get("attempt_plan") or {},
            "retrieval_error": a.get("retrieval_error", False),
            "passages": [trim_passage(p) for p in a.get("passages") or []],
        }
        for a in inputs.get("retrieval_attempts") or []
    ]
    out = {
        "query": inputs.get("query", ""),
        "active_query": inputs.get("active_query", inputs.get("query", "")),
        "plan": inputs.get("plan") or {},
        "filters": inputs.get("filters"),
        "retrieval_attempts": attempts,
    }
    snap = inputs.get("fallback_snapshot")
    if snap:
        out["fallback_snapshot"] = {
            **{k: v for k, v in snap.items() if k != "passages"},
            "passages": [trim_passage(p) for p in snap.get("passages") or []],
        }
    return out


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src, encoding="utf-8") as f:
        run = json.load(f)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(trim(run["inputs"]), f, ensure_ascii=False, indent=1)
        f.write("\n")
