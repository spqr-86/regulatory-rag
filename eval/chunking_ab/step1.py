"""Шаг 1: эталон по нормам и C → родитель, строго offline/cache-only.

.venv/bin/python -m eval.chunking_ab.step1
"""

# ANCHOR: source_docs + read-only sqlite + существующий embed_cache.pkl → step1_* и norm_labels*.
# Поиск C: неизменённый rank_lists(run.py), dense/bm25/hybrid, B=8000/16000.
# Никаких embed_all, фабрики моделей, сетевых клиентов или записей в индекс.

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.chunking_ab import run
from eval.chunking_ab.chunkers import parse_units
from eval.chunking_ab.norms import (
    Chunk,
    Match,
    Norm,
    context_metrics,
    match_norm,
    select_context,
)

BUDGETS = (8000, 16000)


@dataclass(frozen=True)
class Question:
    text: str
    norms: tuple[Norm, ...]
    status: str
    labels: tuple[dict, ...]  # сериализуемые доказательства сопоставления


@dataclass(frozen=True)
class Outcome:
    found: tuple[bool, ...]
    complete: tuple[bool, ...]
    coverage: tuple[float, ...]
    chars: int
    chunks: tuple[int, ...]


def cached_vectors(texts: list[str], path: Path) -> np.ndarray:
    if not path.exists():
        raise RuntimeError(
            f"embedding cache missing: {path}; остановка, вызовы запрещены"
        )
    cache = pickle.loads(path.read_bytes())
    keys = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    missing = sorted({k for k in keys if k not in cache})
    if missing:
        raise RuntimeError(
            f"embedding vectors missing: {len(missing)}; остановка, вызовы запрещены; sha256={missing[:5]}"
        )
    matrix = np.stack([cache[k] for k in keys])
    if (
        matrix.ndim != 2
        or not np.isfinite(matrix).all()
        or (np.linalg.norm(matrix, axis=1) == 0).any()
    ):
        raise RuntimeError("invalid cached vectors; остановка")
    return matrix


def match_record(m: Match) -> dict:
    return {
        "norm_id": m.norm.id if m.norm else None,
        "score": m.score,
        "reason": m.reason,
        "candidates": [
            {"norm_id": c.norm.id, "score": c.score, "chars": len(c.norm.text)}
            for c in m.candidates
        ],
    }


def same_sample(questions: list[dict], variants: dict) -> tuple[list[dict], list[dict]]:
    """Тот же intersection-оракул первого A/B; здесь только фиксирует выборку, не метрику."""
    cw = {v: [run.words(b) for _, _, b in chunks] for v, chunks in variants.items()}
    kept, excluded = [], []
    for q in questions:
        missing = [
            v
            for v, w in cw.items()
            if not any(run.label_cover_sets(w, lab["words"]) for lab in q["labels"])
        ]
        if missing:
            excluded.append(
                {"question": q["question"], "original_oracle_missing": missing}
            )
        else:
            kept.append(q)
    return kept, excluded


def summarize(rows: list[Outcome]) -> dict:
    evaluable = [r for r in rows if r.found]
    norms = sum(len(r.found) for r in rows)
    qfound, qcomplete = (
        sum(any(r.found) for r in rows),
        sum(any(r.complete) for r in rows),
    )
    nfound, ncomplete = (
        sum(sum(r.found) for r in rows),
        sum(sum(r.complete) for r in rows),
    )
    return {
        "questions": len(rows),
        "evaluable_questions": len(evaluable),
        "norms": norms,
        "q_found": qfound,
        "q_complete": qcomplete,
        "q_found_pct": 100 * qfound / len(rows) if rows else 0.0,
        "q_complete_pct": 100 * qcomplete / len(rows) if rows else 0.0,
        "q_found_evaluable_pct": 100 * qfound / len(evaluable) if evaluable else 0.0,
        "q_complete_evaluable_pct": (
            100 * qcomplete / len(evaluable) if evaluable else 0.0
        ),
        "n_found": nfound,
        "n_complete": ncomplete,
        "n_found_pct": 100 * nfound / norms if norms else 0.0,
        "n_complete_pct": 100 * ncomplete / norms if norms else 0.0,
        "mean_chars": sum(r.chars for r in rows) / len(rows) if rows else 0.0,
    }


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def pct(count: int, total: int) -> str:
    return f"{count}/{total} ({100 * count / total:.1f}%)" if total else "0/0 (—)"


def metric_cells(m: dict) -> str:
    return " | ".join(
        (
            pct(m["q_found"], m["questions"]),
            pct(m["q_complete"], m["questions"]),
            pct(m["n_found"], m["norms"]),
            pct(m["n_complete"], m["norms"]),
        )
    )


def render_report(
    questions,
    norms,
    chunks,
    parent_records,
    excluded,
    outcomes,
    aggregates,
    special,
    disputed,
    label_count,
):
    lines = []
    if disputed / label_count > 0.15:
        lines += [
            f"**ВНИМАНИЕ: disputed = {pct(disputed, label_count)} меток, больше 15%; выводы предварительные.**",
            "",
        ]
    lines += [
        "# Шаг 1 — нормы и раскрытие родителей",
        "",
        "Снимок: `build-8cs1plb3`; поиск только по C, без реранкера/LLM. "
        "Эмбеддинги взяты только из `embed_cache.pkl`; новых векторов и сетевых вызовов нет.",
        "",
        f"`load_questions()` возвращает {len(questions) + len(excluded)} вопросов; "
        f"сохранена выборка первого A/B: {len(questions)}. Исключения прежнего оракула: "
        + "; ".join(
            e["question"] + " (" + ", ".join(e["original_oracle_missing"]) + ")"
            for e in excluded
        )
        + ".",
        "",
        f"Метки (вхождения вопрос×метка): {label_count}; disputed: {pct(disputed, label_count)}. "
        f"Уникальных исходных id меток: {len({lab['label_id'] for q in questions for lab in q.labels})}. "
        f"Норм вопрос×норма после дедупликации: {sum(len(q.norms) for q in questions)}. "
        f"Статусы вопросов: {dict(Counter(q.status for q in questions))}.",
        "",
        "Знаменатель вопросов в основных таблицах — вся фиксированная выборка, включая вопросы без "
        "подтверждённых норм (они не засчитываются). Знаменатель норм — подтверждённые уникальные "
        "пары вопрос×норма; disputed в него не входят. В JSON дополнительно есть проценты только "
        "среди вопросов с подтверждённой нормой. Это разные знаменатели, их нельзя смешивать.",
        "",
        "## Правила и ограничения",
        "",
        "- Слова: уникальные `[а-яёa-z0-9]{4,}` в нижнем регистре, без стемминга; как в первом A/B. "
        "Сопоставление строго внутри документа, доля относительно слов метки/чанка ≥0.6, "
        "разрыв лучших двух <0.1 → disputed; при равенстве первым идёт меньший текст.",
        "- Норма — статья/пункт `parse_units`, preamble исключён. Парсер и исходные PDF не исправлялись. "
        "Покрытие complete — объединение слов фактически вошедших тел того же документа / слова нормы ≥0.9. "
        "Повторы слов и порядок не учитываются: это лексическая полнота, а не доказательство наличия каждого подпункта.",
        "- found: чанк имеет эту норму родителем; дополнительно в A одиночный чанк может покрыть ≥0.6 слов нормы. "
        "found и complete считаются независимо.",
        "- A: исходные C; B: родитель целиком, дедуп по id родителя. Без родителя — исходный C. "
        "Для длинного родителя (>6000) сначала ищется точная последовательность слов чанка; якорь — покрывающие её абзацы. Если её нет, берётся абзац с максимальным пересечением слов чанка "
        "(при равенстве первый); слева/справа добавляются только целые абзацы до 3000 знаков. "
        "Первое окно родителя сохраняется, последующие попадания в него пропускаются. "
        "Сам якорный абзац не режется, поэтому окно может быть длиннее 6000.",
        "- Бюджет строгий, включая переводы строки между элементами; на первом переполнении остановка "
        "(после дедупликации), даже для первого элемента. В первом A/B первый чанк допускал переполнение; "
        "здесь его нет. Заголовок раздела входит в бюджет A, но не в проверку complete.",
        "",
        "## Основная таблица (заглавный поиск — hybrid)",
        "",
        "| поиск | бюджет | схема | found вопросов | complete вопросов | found норм | complete норм | средний контекст, знаков |",
        "|---|---:|---|---|---|---|---|---:|",
    ]
    for r in ("hybrid", "dense", "bm25"):
        for b in BUDGETS:
            for s in ("A", "B"):
                m = aggregates[f"{r}/{b}/{s}"]
                lines.append(
                    f"| {r} | {b} | {s} | {metric_cells(m)} | {m['mean_chars']:.0f} |"
                )
    lines += [
        "",
        "## По документам (все поиски и бюджеты)",
        "",
        "Вопрос относится к документу, если имеет в нём подтверждённую норму; "
        "в многодокументном вопросе здесь оцениваются только нормы этого документа. Сумма числа вопросов по документам может превышать общий n; сумма норм совпадает с общим числом норм.",
        "",
        "| документ | поиск | бюджет | схема | found вопросов | complete вопросов | found норм | complete норм |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for doc in run.DOCS:
        for r in ("hybrid", "dense", "bm25"):
            for b in BUDGETS:
                for s in ("A", "B"):
                    rows = []
                    for q, out in zip(questions, outcomes[f"{r}/{b}/{s}"]):
                        ix = [i for i, n in enumerate(q.norms) if n.doc == doc]
                        if ix:
                            rows.append(
                                Outcome(
                                    tuple(out.found[i] for i in ix),
                                    tuple(out.complete[i] for i in ix),
                                    (),
                                    out.chars,
                                    (),
                                )
                            )
                    lines.append(
                        f"| {doc} | {r} | {b} | {s} | {metric_cells(summarize(rows))} |"
                    )
    no_parent = sum(c.parent is None for c in chunks)
    lines += [
        "",
        "## Родители C",
        "",
        f"Всего C: {len(chunks)}; без родителя: {pct(no_parent, len(chunks))}.",
        "",
        "| документ | C | без родителя | ниже порога | неоднозначные | нет слов |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for doc in run.DOCS:
        records = [p for p in parent_records if p["doc"] == doc]
        reasons = Counter(p["reason"] for p in records)
        lines.append(
            f"| {doc} | {len(records)} | {pct(sum(p['norm_id'] is None for p in records), len(records))} | {reasons['below_threshold']} | {reasons['ambiguous']} | {reasons['empty_words']} |"
        )
    lines += [
        "",
        "## G005 и G047 — именно целевые нормы",
        "",
        "Здесь проверяется целевой п.16 / ст.229, даже если привязка метки disputed или ведёт в другую норму. "
        "Результат не подменяется успехом любой иной нормы вопроса. «Полон» означает complete нормы, не качество ответа LLM.",
        "",
        "| кейс | норма | поиск | бюджет | схема | found | complete | покрытие слов |",
        "|---|---|---|---:|---|---|---|---:|",
    ]
    for record in special:
        lines.append(
            f"| {record['case']} | {record['norm_id']} | {record['retriever']} | {record['budget']} | {record['scheme']} | {record['found']} | {record['complete']} | {record['coverage']:.1%} |"
        )
    lines += [
        "",
        "## Примеры перехода complete",
        "",
        "Сначала hybrid/8000, затем hybrid/16000 и остальные конфигурации. "
        "Пять разных вопросов в каждом направлении; параметры каждого примера указаны.",
        "",
    ]
    for title, wanted in (("B complete, A нет", True), ("A complete, B нет", False)):
        lines += [f"### {title}", ""]
        seen = set()
        examples = []
        for retriever in ("hybrid", "dense", "bm25"):
            for budget in BUDGETS:
                a = outcomes[f"{retriever}/{budget}/A"]
                b = outcomes[f"{retriever}/{budget}/B"]
                indices = [
                    i
                    for i in range(len(questions))
                    if any(a[i].complete) != any(b[i].complete)
                    and any(b[i].complete) == wanted
                ]
                lines.append(f"{retriever}/{budget}: {len(indices)} вопросов.")
                for i in indices:
                    if i in seen:
                        continue
                    seen.add(i)
                    q = questions[i]
                    coverage = "; ".join(
                        f"{n.id}: A={a[i].coverage[j]:.1%}, B={b[i].coverage[j]:.1%}"
                        for j, n in enumerate(q.norms)
                    )
                    examples.append(f"- **{retriever}/{budget}** {q.text} — {coverage}")
        lines += ["", *examples[:5], ""]
        if len(examples) < 5:
            lines += [
                f"Всего разных вопросов: {len(examples)}; пять примеров набрать невозможно.",
                "",
            ]
    lines += [
        "## Что требует внимания",
        "",
        "G005: основной успех по правилу «хотя бы одна норма» может происходить по п.17, "
        "который тоже получился из исходной разметки. Это не означает полноту целевого п.16. "
        "G047: ст.229 длиннее 6000 знаков, поэтому B передаёт окно, а не статью целиком; "
        "complete зависит от конкретного первого найденного чанка. Целевые строки выше позволяют "
        "не скрыть эти дефекты за общей долей успешных вопросов.",
        "",
    ]
    lines += [
        "## Воспроизведение и аудит",
        "",
        "`.venv/bin/python -m eval.chunking_ab.step1` (offline). "
        "`norm_labels.jsonl` хранит каждое сопоставление и двух кандидатов; "
        "`norm_labels_disputed.md` — спорные метки; `step1_norms.jsonl` — тексты норм; "
        "`step1_parents.jsonl` — привязки C; `step1_details.jsonl` — выбранные C и покрытие каждой нормы; "
        "`step1_results.json` — агрегаты, особые кейсы, исключённый вопрос и SHA256 входов.",
        "",
        "Ограничения: метки по-прежнему получены из C, хотя новая целевая единица от нарезки C не зависит. "
        "Ошибки прототипа parse_units, PDF/Docling, редакций источников здесь не исправлены; "
        "качество генерации и полнота правового ответа не измерены. Разница при равных бюджетах "
        "показывает лишь эффект раскрытия на этих рангах и этой выборке.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    paras = run.load_paragraphs()
    norms = [
        Norm(doc, u.kind, u.number, tuple(u.paragraphs), i)
        for doc, ps in paras.items()
        for i, u in enumerate(parse_units(ps))
        if u.kind != "preamble"
    ]
    print(f"norms: {len(norms)}", flush=True)
    variants = run.build_variants(paras)
    raw_questions, _ = run.load_questions()
    raw_questions, excluded = same_sample(raw_questions, variants)
    print(f"same sample: {len(raw_questions)}; excluded: {excluded}", flush=True)
    questions = []
    for raw in raw_questions:
        labels, matched = [], {}
        for lab in raw["labels"]:
            m = match_norm(lab["doc"], " ".join(lab["words"]), norms)
            labels.append({"label_id": lab["id"], "doc": lab["doc"], **match_record(m)})
            if m.norm:
                matched[m.norm.id] = m.norm
        bad = sum(lab["reason"] is not None for lab in labels)
        status = "disputed" if not matched else "partial_disputed" if bad else "ok"
        questions.append(
            Question(raw["question"], tuple(matched.values()), status, tuple(labels))
        )
    chunks, parents = [], []
    current_rows = run.read_chunks(run.CURRENT_DB, set(run.DOCS))
    for i, (doc, text, body) in enumerate(variants["C"]):
        m = match_norm(doc, body, norms)
        chunks.append(Chunk(doc, text, body, m.norm))
        parents.append(
            {"index": i, "doc": doc, "chunk_id": current_rows[i][1], **match_record(m)}
        )
    qtexts = [q.text for q in questions]
    texts = [c.text for c in chunks]
    # Проверяем все нужные векторы до любых выходных файлов. Нет пути к платному fallback.
    vecs = cached_vectors(texts + qtexts, run.OUT / "embed_cache.pkl")
    ranks = run.rank_lists(texts, vecs[: len(texts)], qtexts, vecs[len(texts) :])
    print(
        f"cache-only ranks ready: {len(chunks)} C, {len(questions)} questions",
        flush=True,
    )
    outcomes, details, special = {}, [], []
    cases = {
        "G005": (
            "В каких случаях проводится внеплановый инструктаж?",
            "2464.md",
            "item",
            "16",
        ),
        "G047": (
            "Кто входит в комиссию по расследованию несчастного случая на производстве?",
            "тк рф.pdf",
            "article",
            "229",
        ),
    }
    for r in run.RETRIEVERS:
        for b in BUDGETS:
            for s in ("A", "B"):
                key = f"{r}/{b}/{s}"
                rows = []
                for qi, q in enumerate(questions):
                    context = select_context(ranks[r][qi], chunks, s, b)
                    metrics = [context_metrics(context, n, s) for n in q.norms]
                    chars = len("\n".join(c.text for c in context))
                    out = Outcome(
                        tuple(m[0] for m in metrics),
                        tuple(m[1] for m in metrics),
                        tuple(m[2] for m in metrics),
                        chars,
                        tuple(c.chunk_index for c in context),
                    )
                    rows.append(out)
                    details.append(
                        {
                            "question_index": qi,
                            "question": q.text,
                            "retriever": r,
                            "budget": b,
                            "scheme": s,
                            "chars": chars,
                            "selected_C_indices": out.chunks,
                            "context": [
                                {
                                    "C_index": c.chunk_index,
                                    "parent": c.parent.id if c.parent else None,
                                    "expanded": c.expanded,
                                    "chars": len(c.text),
                                    "text_sha256": hashlib.sha256(
                                        c.text.encode()
                                    ).hexdigest(),
                                }
                                for c in context
                            ],
                            "norms": [
                                {
                                    "norm_id": n.id,
                                    "found": m[0],
                                    "complete": m[1],
                                    "coverage": m[2],
                                }
                                for n, m in zip(q.norms, metrics)
                            ],
                        }
                    )
                    for case, (question, doc, kind, number) in cases.items():
                        if q.text == question:
                            targets = [
                                n
                                for n in norms
                                if (n.doc, n.kind, n.number) == (doc, kind, number)
                            ]
                            if len(targets) != 1:
                                raise RuntimeError(
                                    f"{case}: target norm count {len(targets)}"
                                )
                            f, c, cov = context_metrics(context, targets[0], s)
                            special.append(
                                {
                                    "case": case,
                                    "norm_id": targets[0].id,
                                    "retriever": r,
                                    "budget": b,
                                    "scheme": s,
                                    "found": f,
                                    "complete": c,
                                    "coverage": cov,
                                }
                            )
                outcomes[key] = rows
    if len(special) != 24:
        raise RuntimeError(f"G005/G047: missing cases, rows={len(special)}")
    aggregates = {k: summarize(v) for k, v in outcomes.items()}
    disputed = sum(lab["reason"] is not None for q in questions for lab in q.labels)
    label_count = sum(len(q.labels) for q in questions)

    def jsonl(name, rows):
        (run.OUT / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    jsonl(
        "norm_labels.jsonl",
        (
            {
                "question_index": i,
                "question": q.text,
                "status": q.status,
                "norms": [
                    {"id": n.id, "doc": n.doc, "kind": n.kind, "number": n.number}
                    for n in q.norms
                ],
                "labels": q.labels,
            }
            for i, q in enumerate(questions)
        ),
    )
    jsonl(
        "step1_norms.jsonl",
        (
            {
                "id": n.id,
                "doc": n.doc,
                "kind": n.kind,
                "number": n.number,
                "paragraphs": n.paragraphs,
                "chars": len(n.text),
            }
            for n in norms
        ),
    )
    jsonl("step1_parents.jsonl", parents)
    jsonl("step1_details.jsonl", details)
    dl = [
        "# Спорные метки",
        "",
        f"Вхождений: {pct(disputed, label_count)}.",
        "",
        "| вопрос | метка | причина | лучший кандидат | доля | второй кандидат | доля |",
        "|---|---|---|---|---:|---|---:|",
    ]
    for q in questions:
        for lab in q.labels:
            if lab["reason"]:
                cs = lab["candidates"]
                cells = [f"{c['norm_id']} | {c['score']:.3f}" for c in cs]
                cells += ["— | —"] * (2 - len(cells))
                dl.append(
                    f"| {q.text.replace('|', '/')} | {lab['label_id']} | {lab['reason']} | "
                    + " | ".join(cells)
                    + " |"
                )
    (run.OUT / "norm_labels_disputed.md").write_text(
        "\n".join(dl) + "\n", encoding="utf-8"
    )
    inputs = [
        run.CURRENT_DB,
        run.LABEL_DB,
        *(run.ROOT / f for f in run.QUESTION_FILES),
        *(run.ROOT / "source_docs" / fname for fname, _ in run.DOCS.values()),
        run.OUT / "embed_cache.pkl",
        Path(__file__),
        Path(__file__).with_name("norms.py"),
        Path(run.__file__),
        Path(__file__).with_name("chunkers.py"),
    ]
    results = {
        "snapshot": "build-8cs1plb3",
        "questions": len(questions),
        "excluded": excluded,
        "label_count": label_count,
        "disputed": disputed,
        "disputed_fraction": disputed / label_count,
        "norm_instances": sum(len(q.norms) for q in questions),
        "question_statuses": dict(Counter(q.status for q in questions)),
        "chunks_C": len(chunks),
        "chunks_without_parent": sum(c.parent is None for c in chunks),
        "metrics": aggregates,
        "special": special,
        "input_sha256": {
            str(p.relative_to(run.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in inputs
        },
    }
    write_json(run.OUT / "step1_results.json", results)
    report = render_report(
        questions,
        norms,
        chunks,
        parents,
        excluded,
        outcomes,
        aggregates,
        special,
        disputed,
        label_count,
    )
    (run.OUT / "step1_report.md").write_text(report, encoding="utf-8")
    print(report[:4500], flush=True)
    print(f"saved: {run.OUT / 'step1_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
