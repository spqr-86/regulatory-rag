"""Выгрузка пулов кандидатов для ручной (агентной) разметки held-out.

Тот же пул, что видит платный судья в ``label_golden_retrieval.py``: те же два
пути поиска, тот же round-robin и тот же лимит. Отличие одно — LLM не зовётся,
пул пишется на диск шардами, и судит его агент-разметчик.

Usage::

    .venv/bin/python eval/dump_candidate_pools.py \
        --dataset eval/data/dataset_heldout_ext_2026-09-06.csv \
        --out-dir eval/data/pools_ext --shard-size 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.label_golden_retrieval import (  # noqa: E402
    CANDIDATE_LIMIT,
    PATHS,
    TOP_K,
    build_candidate_pools,
    load_golden_questions,
    merge_candidates,
)

CAND_FIELDS = ("chunk_id", "source", "text")


def candidate_view(cand: dict, index: int) -> dict:
    """Кандидат в том виде, в каком его читает разметчик: номер как в промпте."""
    source = cand.get("source") or (cand.get("chunk_id") or "").split("#")[0]
    return {
        "index": index,
        "chunk_id": cand.get("chunk_id", ""),
        "source": source,
        "text": cand.get("text", ""),
    }


def shard(items: list, size: int) -> list[list]:
    """Нарезка на пачки по ``size`` с сохранением порядка."""
    if size < 1:
        raise ValueError("shard size must be >= 1")
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_LIMIT)
    parser.add_argument("--limit", type=int, default=0, help="первые N вопросов")
    args = parser.parse_args()

    records = load_golden_questions(args.dataset)
    records = [r for r in records if r["in_scope"]]
    if args.limit:
        records = records[: args.limit]

    from eval.run_retrieval_eval import init_engine  # noqa: PLC0415

    init_engine()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    empty = 0
    for rec in records:
        pools = build_candidate_pools(rec["question"], top_k=args.top_k)
        merged = merge_candidates(pools, limit=args.candidates, pool_names=PATHS)
        if not merged:
            empty += 1
        items.append(
            {
                "n": rec["n"],
                "question": rec["question"],
                "candidates": [
                    candidate_view(c, i) for i, c in enumerate(merged, start=1)
                ],
            }
        )
        print(f"[{rec['n']:>3}] {len(merged):>2} кандидатов · {rec['question'][:60]}")

    for i, batch in enumerate(shard(items, args.shard_size), start=1):
        path = args.out_dir / f"shard_{i:02d}.json"
        path.write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{path} — вопросов {len(batch)}")
    print(f"\nитого: {len(items)} вопросов, пустых пулов {empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
