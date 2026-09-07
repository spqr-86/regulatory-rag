"""Сборка вердиктов агентов-разметчиков в тот же эталон, что даёт платный путь.

``label_golden_retrieval.py`` зовёт модель по API; здесь ровно те же два прохода
с противоположными установками выполняют субагенты, а этот скрипт делает всё
остальное существующим кодом: разбор вердикта, проверку цитаты по тексту чанка,
слияние проходов и очередь человеку. Логика согласия не дублируется — она
импортируется, иначе два пути разошлись бы молча.

Usage::

    .venv/bin/python eval/ingest_agent_labels.py \
        --pools eval/data/pools_ext --verdicts eval/data/verdicts_ext \
        --out eval/data/golden_retrieval_labeled_ext.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.label_golden_retrieval import (  # noqa: E402
    CONTROL_SAMPLE,
    CONTROL_SEED,
    STATUS_NONE_FOUND,
    apply_arbitration,
    build_labeled_gt,
    build_review_rows,
    merge_passes,
    parse_labels,
    summarize,
    validate_label_spans,
    write_tsv,
)


def load_pools(pools_dir: Path) -> dict[int, dict]:
    """Пулы кандидатов из шардов дампа, по номеру вопроса."""
    pools: dict[int, dict] = {}
    for path in sorted(Path(pools_dir).glob("shard_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            pools[int(item["n"])] = item
    return pools


def load_passes(verdicts_dir: Path, pass_name: str) -> dict[int, dict]:
    """Вердикты одного прохода: ``shard_XX.<pass>.json`` → ``n -> вердикт``."""
    out: dict[int, dict] = {}
    for path in sorted(Path(verdicts_dir).glob(f"shard_*.{pass_name}.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for verdict in data.get("verdicts") or []:
            out[int(verdict["n"])] = verdict
    return out


def _check_coverage(pools: dict[int, dict], passes: dict[str, dict[int, dict]]) -> None:
    """Вопрос без вердикта — не none_found, а незаконченная работа.

    Молча размеченный пустым, он уходит из метрики, и набор читается лучше, чем
    он есть, — ровно то, ради чего вся эта разметка и затевалась.
    """
    for name, verdicts in passes.items():
        missing = sorted(set(pools) - set(verdicts))
        if missing:
            raise ValueError(
                f"проход {name}: нет вердикта по вопросам {missing[:20]}"
                f" (всего {len(missing)})"
            )
        extra = sorted(set(verdicts) - set(pools))
        if extra:
            raise ValueError(f"проход {name}: вердикт по неизвестным вопросам {extra}")


def build_verdicts(
    pools: dict[int, dict],
    strict: dict[int, dict],
    lenient: dict[int, dict],
) -> dict[int, dict]:
    """Слить два прохода в вердикт по каждому вопросу."""
    _check_coverage(pools, {"strict": strict, "lenient": lenient})
    out: dict[int, dict] = {}
    for n in sorted(pools):
        candidates = pools[n]["candidates"]
        if not candidates:
            out[n] = {
                "status": STATUS_NONE_FOUND,
                "gold_chunk_ids": [],
                "disputed_chunk_ids": [],
                "disputed_indices": [],
                "quotes": [],
                "note": "поиск не вернул кандидатов",
            }
            continue
        s = validate_label_spans(parse_labels(strict[n], len(candidates)), candidates)
        l = validate_label_spans(parse_labels(lenient[n], len(candidates)), candidates)
        out[n] = merge_passes(s, l, candidates)
    return out


def apply_arbitrations(
    pools: dict[int, dict],
    verdicts: dict[int, dict],
    arbitrations: dict[int, dict],
) -> dict[int, dict]:
    """Наложить решения арбитра (тем же правилом доказательства, что у платного пути)."""
    out = dict(verdicts)
    for n, arb in arbitrations.items():
        if n not in out:
            raise ValueError(f"арбитраж по неизвестному вопросу {n}")
        candidates = pools[n]["candidates"]
        checked = validate_label_spans(parse_labels(arb, len(candidates)), candidates)
        out[n] = apply_arbitration(out[n], checked, candidates)
    return out


def _records(pools: dict[int, dict]) -> list[dict]:
    return [
        {"n": n, "question": pools[n]["question"]} for n in sorted(pools)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pools", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--arbitration", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=CONTROL_SAMPLE)
    parser.add_argument("--seed", type=int, default=CONTROL_SEED)
    args = parser.parse_args()

    pools = load_pools(args.pools)
    strict = load_passes(args.verdicts, "strict")
    lenient = load_passes(args.verdicts, "lenient")
    verdicts = build_verdicts(pools, strict, lenient)

    if args.arbitration and Path(args.arbitration).is_dir():
        verdicts = apply_arbitrations(
            pools, verdicts, load_passes(args.arbitration, "arbiter")
        )

    records = _records(pools)
    ordered = [{**verdicts[r["n"]], "n": r["n"]} for r in records]
    summary = summarize(ordered)

    gt = build_labeled_gt(records, verdicts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in gt:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_n = {r["n"]: r for r in records}
    rows = build_review_rows(by_n, ordered, controls=args.controls, seed=args.seed)
    tsv_path = Path(str(args.out).replace(".jsonl", "")).with_suffix(".review.tsv")
    write_tsv(rows, tsv_path)

    print(
        f"итог: {summary['agreed']} согласий · {summary['arbitrated']} решено арбитром · "
        f"{summary['disputed']} споров · {summary['none_found']} без ответа · "
        f"{summary['unverified_quote']} без цитаты"
    )
    print(f"эталон: {len(gt)} вопросов → {args.out}")
    blocking = sum(1 for r in rows if r["role"] == "blocking")
    print(f"человеку: {len(rows)} строк ({blocking} обязательных) → {tsv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
