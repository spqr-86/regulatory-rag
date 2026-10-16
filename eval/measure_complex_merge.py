"""Measure the complex merge (evaluate_complex) — HR@k of the merged candidate.

Runs router → rag_simple → evaluate_triage → rag_complex over the GT and
builds the merged candidate exactly as evaluate_complex does (simple anchors,
then MMR over the complex pool by rerank score). No generation and no LLM
calls: the only cost is query embeddings.

Every question is pushed through rag_complex, not only the escalated ones, so
the merge is measured on the whole set; the escalated subset is reported apart.
MMR selection is greedy, so the merged list for FINAL_MERGE_TOP_K=K is the
prefix of the list for a larger K: one run at the largest K answers 24/16/12.

Usage:
    CHROMA_DB_PATH=... CHROMA_COLLECTION_NAME=documents \
    .venv/bin/python eval/measure_complex_merge.py --gt <gt.jsonl> --out <run.json>
"""

# ANCHOR: offline HR@k of the evaluate_complex merged candidate for several
# FINAL_MERGE_TOP_K values. Input: labeled GT; output: per-question ranks of the
# first relevant chunk in simple / complex pool / merged lists + aggregates.

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import List, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.run_retrieval_eval import extract_chunk_ids, load_gt  # noqa: E402

DEFAULT_KS = (12, 16, 24)


def first_rank(passages: Sequence[dict], relevant: set) -> int | None:
    """1-based rank of the first relevant chunk, None if absent."""
    for i, cid in enumerate(extract_chunk_ids(list(passages)), 1):
        if cid in relevant:
            return i
    return None


def hit_rates(records: List[dict], key: str, ks: Sequence[int]) -> dict:
    n = len(records)
    return {
        f"{key}@{k}": round(
            sum(1 for r in records if r[key] is not None and r[key] <= k) / n, 3
        )
        if n
        else 0.0
        for k in ks
    }


def run(gt: Sequence[dict], ks: Sequence[int]) -> dict:
    from src.v7.config import v7_config
    from src.v7.nodes.evaluate_complex import _merged_passages
    from src.v7.nodes.evaluate_triage import evaluate_triage
    from src.v7.nodes.rag_complex import rag_complex
    from src.v7.nodes.rag_simple import rag_simple
    from src.v7.nodes.router import router

    v7_config.FINAL_MERGE_TOP_K = max(ks)
    records, t0 = [], time.perf_counter()
    for rec in gt:
        q = rec["question"]
        relevant = set(rec.get("relevant_chunk_ids") or [rec["chunk_id"]])
        try:
            state: dict = {"query": q, "filters": None}
            state.update(router(state))
            if state.get("clarify_message"):
                records.append({"question": q, "skipped": "clarify"})
                continue
            state.update(rag_simple(state))
            update = evaluate_triage(state)
            escalated = update.get("route_decision") == "complex"
            state.update(update)
            if not escalated:
                state["fallback_snapshot"] = {"passages": update["final_context"]}
            state.update(rag_complex(state))
            complex_attempts = [
                a for a in state["retrieval_attempts"] if a.get("stage") == "complex"
            ]
            pool = complex_attempts[-1]["passages"] if complex_attempts else []
            plan = (
                complex_attempts[-1].get("attempt_plan") if complex_attempts else None
            ) or {}
            merged = _merged_passages(state, plan)
            records.append(
                {
                    "question": q,
                    "source": rec.get("source"),
                    "escalated": escalated,
                    "n_anchors": len(state["fallback_snapshot"].get("passages") or []),
                    "n_merged": len(merged),
                    "simple": first_rank(update["final_context"], relevant),
                    "complex": first_rank(pool, relevant),
                    "merged": first_rank(merged, relevant),
                }
            )
        except Exception as exc:  # a dead query must not kill the run
            print(f"  ! failed: {q[:60]}… — {exc}")
            records.append({"question": q, "error": str(exc)})

    ok = [r for r in records if "simple" in r]
    esc = [r for r in ok if r["escalated"]]
    summary = {"n": len(gt), "measured": len(ok), "n_escalated": len(esc)}
    for name, subset in (("all", ok), ("escalated", esc)):
        summary[name] = {
            **hit_rates(subset, "simple", ks),
            **hit_rates(subset, "complex", ks),
            **hit_rates(subset, "merged", ks),
        }
    summary.update(
        {
            "ks": list(ks),
            "records": records,
            "elapsed_s": round(time.perf_counter() - t0, 1),
            "date": date.today().isoformat(),
        }
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    from config.settings import settings
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import init_v7_pipeline

    gt = load_gt(a.gt, limit=a.limit)
    print(f"GT: {len(gt)} из {a.gt}; DB {settings.CHROMA_DB_PATH}")
    init_v7_pipeline(get_vector_store_backend(), llm_provider=None)
    result = run(gt, a.ks)
    result.update({"gt_path": str(a.gt), "db": str(settings.CHROMA_DB_PATH)})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("n", "measured", "n_escalated", "all", "escalated")
            },
            indent=1,
        )
    )
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
