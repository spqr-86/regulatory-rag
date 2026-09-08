"""Платный арбитр по спорным вопросам, которые уже размечены агентами.

Основной скрипт (``label_golden_retrieval.py``) строит пулы заново и делает оба
прохода сам. Здесь пулы и вердикты уже есть на диске: нужен только третий
проход по конкретному списку вопросов, той же функцией ``arbitrate_one`` и с той
же проверкой цитаты, чтобы платный и агентный пути не разошлись в критерии.

Usage::

    .venv/bin/python eval/run_paid_arbitration.py \
        --tasks eval/data/arbitration_ext --verdicts eval/data/verdicts_ext \
        --out eval/data/verdicts_paid_ext --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval.ingest_agent_labels import load_passes  # noqa: E402
from eval.label_golden_retrieval import (  # noqa: E402
    ARBITER_MODEL,
    CANDIDATE_LIMIT,
    arbitrate_one,
    parse_labels,
    validate_label_spans,
)
from src.pricing import calc_total_price, price_for  # noqa: E402


def load_tasks(tasks_dir: Path) -> dict[int, dict]:
    """Задания арбитра из шардов дампа, по номеру вопроса."""
    tasks: dict[int, dict] = {}
    for path in sorted(Path(tasks_dir).glob("shard_*.json")):
        for item in json.loads(path.read_text(encoding="utf-8")):
            tasks[int(item["n"])] = item
    return tasks


def load_cited(verdicts_dir: Path) -> dict[int, list[dict]]:
    """Номера, по которым хоть один проход дал цитату, с этими цитатами."""
    cited: dict[int, list[dict]] = {}
    for pass_name in ("strict", "lenient"):
        for n, verdict in load_passes(Path(verdicts_dir), pass_name).items():
            relevant = verdict.get("relevant") or []
            if relevant:
                cited.setdefault(n, []).extend(relevant)
    return cited


def select_rejected(
    arbiter: dict[int, dict], cited: dict[int, list[dict]]
) -> list[int]:
    """Вопросы, где арбитр не засчитал ничего, а разметчик цитату всё же дал.

    Именно на них контрольная выборка 07.09 показала пропуски: там, где обе
    стороны молчали, спорить не о чем."""
    return sorted(n for n, v in arbiter.items() if not (v.get("relevant") or []) and n in cited)


def arbitrate_tasks(
    tasks: Sequence[dict],
    llm,
    arbiter: Callable = arbitrate_one,
) -> tuple[list[dict], list[dict]]:
    """Третий проход по заданиям.

    Вердикт сохраняется сырым: цитату проверяет ``ingest_agent_labels`` тем же
    ``validate_label_spans``, что и агентный путь, а непроверенную отдаёт
    человеку. Фильтровать здесь значило бы завести второй критерий."""
    verdicts: list[dict] = []
    usages: list[dict] = []
    for task in tasks:
        candidates = task["candidates"]
        labels, usage = arbiter(
            task["question"], candidates, task.get("open_indices") or [], llm
        )
        parsed = parse_labels(labels, len(candidates))
        verdicts.append(
            {
                "n": int(task["n"]),
                "relevant": parsed.get("relevant") or [],
                "note": parsed.get("note", ""),
            }
        )
        usages.append(usage)
    return verdicts, usages


def count_unverified(
    verdicts: Sequence[dict], candidates_by_n: dict[int, Sequence[dict]]
) -> int:
    """Сколько цитат не нашлось в тексте своего чанка — сигнал для сводки."""
    total = 0
    for verdict in verdicts:
        candidates = candidates_by_n.get(int(verdict["n"])) or []
        checked = validate_label_spans(verdict, candidates)
        total += sum(1 for r in checked.get("relevant") or [] if not r.get("verified"))
    return total


def write_verdicts(verdicts: Sequence[dict], out_dir: Path, name: str = "paid") -> Path:
    """Вердикты в том же формате, что читает ``ingest_agent_labels``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.arbiter.json"
    path.write_text(
        json.dumps({"pass": "arbiter", "verdicts": list(verdicts)}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return path


def estimate_arbitration_cost(
    n_questions: int,
    model: str = ARBITER_MODEL,
    avg_candidate_tokens: int = 180,
    n_candidates: int = CANDIDATE_LIMIT,
) -> float:
    """Один проход арбитра по каждому вопросу, ~120 токенов на выход."""
    per_question = calc_total_price(
        [{"input": avg_candidate_tokens * n_candidates + 300, "output": 120}],
        model=model,
    )
    return per_question * n_questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=ARBITER_MODEL)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_LIMIT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="оценка, без вызовов")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    arbiter_verdicts = load_passes(args.verdicts, "arbiter")
    picked = select_rejected(arbiter_verdicts, load_cited(args.verdicts))
    picked = [n for n in picked if n in tasks]
    if args.limit:
        picked = picked[: args.limit]

    rate = price_for(args.model)
    estimate = estimate_arbitration_cost(
        len(picked), model=args.model, n_candidates=args.candidates
    )
    print(
        f"вопросов: {len(picked)} · арбитр: {args.model} "
        f"(${rate['input']}/${rate['output']} за 1M) · оценка: ${estimate:.2f}"
    )
    if args.dry_run:
        return 0

    from eval.label_golden_retrieval import _make_llm  # noqa: PLC0415

    llm = _make_llm(args.model)
    verdicts, usages = arbitrate_tasks([tasks[n] for n in picked], llm)
    path = write_verdicts(verdicts, args.out)

    found = sum(1 for v in verdicts if v["relevant"])
    unverified = count_unverified(verdicts, {n: tasks[n]["candidates"] for n in picked})
    spent = calc_total_price(usages, model=args.model)
    print(f"засчитано: {found} из {len(verdicts)} → {path}")
    print(f"цитат без подтверждения в тексте: {unverified}")
    print(f"потрачено: ${spent:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
