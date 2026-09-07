"""Задания арбитру: что именно осталось нерешённым после двух проходов.

Арбитр судит только открытую часть вердикта — спорных кандидатов и согласия,
подпёртые цитатой, которой в чанке нет. Весь пул он пересматривает лишь там, где
дешёвая пара не нашла ничего: правило то же, что у платного пути
(``apply_arbitration``), иначе третье мнение молча переписало бы эталон целиком.

Usage::

    .venv/bin/python eval/dump_arbitration_tasks.py \
        --pools eval/data/pools_ext --verdicts eval/data/verdicts_ext \
        --out-dir eval/data/arbitration_ext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.ingest_agent_labels import (  # noqa: E402
    build_verdicts,
    load_passes,
    load_pools,
)
from eval.label_golden_retrieval import needs_arbitration  # noqa: E402


def build_tasks(pools: dict[int, dict], verdicts: dict[int, dict]) -> list[dict]:
    """Открытые вопросы с указанием, какие кандидаты остались предметом спора."""
    tasks = []
    for n in sorted(verdicts):
        verdict = verdicts[n]
        if not needs_arbitration(verdict):
            continue
        pool = pools[n]
        open_idx = list(verdict.get("disputed_indices") or [])
        if not open_idx:
            open_idx = [c["index"] for c in pool["candidates"]]
        tasks.append(
            {
                "n": n,
                "question": pool["question"],
                "status": verdict.get("status", ""),
                "open_indices": open_idx,
                "candidates": pool["candidates"],
            }
        )
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=8)
    args = parser.parse_args()

    pools = load_pools(args.pools)
    verdicts = build_verdicts(
        pools,
        load_passes(args.verdicts, "strict"),
        load_passes(args.verdicts, "lenient"),
    )
    tasks = build_tasks(pools, verdicts)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    batches = [
        tasks[i : i + args.shard_size] for i in range(0, len(tasks), args.shard_size)
    ]
    for i, batch in enumerate(batches, start=1):
        path = args.out_dir / f"shard_{i:02d}.json"
        path.write_text(
            json.dumps(batch, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"{path} — вопросов {len(batch)}")
    print(f"\nна арбитраж: {len(tasks)} вопросов из {len(verdicts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
