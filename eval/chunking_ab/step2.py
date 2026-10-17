"""Шаг 2: позиционные родители, бюджетная диагностика и K dry-run, строго offline.

.venv/bin/python -m eval.chunking_ab.step2
"""

# ANCHOR: read-only sqlite + source_docs + неизменённые метки шага 1 + cache → step2_*.
# A фиксирован из шага 1; B1 полный parent, B2 ≤3000, Bdiag все top100 без бюджета.
# Пороги alignment выбираются до retrieval; K только оценка, фабрика LLM не импортируется.

from __future__ import annotations

import hashlib
import json
import pickle
import socket
import sqlite3
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from eval.chunking_ab import run
from eval.chunking_ab.align import (
    ROLLBACK_PENALTY,
    THRESHOLD,
    AlignInput,
    align_document,
    build_children,
    parent_window,
)
from eval.chunking_ab.chunkers import parse_units
from eval.chunking_ab.norms import Chunk, Context, Norm, context_metrics
from eval.chunking_ab.norms import select_context as old_select_context
from eval.chunking_ab.step1 import (
    Outcome,
    Question,
    cached_vectors,
    metric_cells,
    pct,
    summarize,
    write_json,
)


def select_context(
    order: list[int], chunks: list[Chunk], scheme: str, budget: int | None
) -> list[Context]:
    if scheme == "A":
        return old_select_context(order, chunks, "A", budget)
    if scheme not in ("B1", "B2", "Bdiag"):
        raise ValueError(scheme)
    chosen, seen, total = [], set(), 0
    for raw_i in order:
        i = int(raw_i)
        c = chunks[i]
        expanded = c.parent is not None
        if expanded:
            if c.parent.id in seen:
                continue
            text = body = (
                parent_window(c.parent, c.body) if scheme == "B2" else c.parent.text
            )
        else:
            text, body = c.text, c.body
        cost = len(text) + bool(chosen)
        if budget is not None and total + cost > budget:
            break
        chosen.append(Context(c.doc, text, body, c.parent, i, expanded))
        total += cost
        if expanded:
            seen.add(c.parent.id)
    return chosen


def loss_reason(
    a_found: bool,
    b_found: bool,
    diag_found: bool,
    norm: Norm,
    order: list[int],
    chunks: list[Chunk],
    original_chunks: list[Chunk] | None = None,
) -> str | None:
    if not a_found or b_found:
        return None
    if diag_found:
        return "budget"
    # Причинный кандидат: parentless чанк в top100, который дал бы found по правилу A.
    if any(
        chunks[i].doc == norm.doc
        and chunks[i].parent is None
        and (
            (original_chunks is not None and original_chunks[i].parent == norm)
            or (
                norm.words
                and len(run.words(chunks[i].body) & norm.words) / len(norm.words) >= 0.6
            )
        )
        for i in order
    ):
        return "missing_parent"
    return "other"


def k_stats(texts: list[str], cache_path: Path) -> dict:
    import tiktoken

    cache = pickle.loads(cache_path.read_bytes())
    unique = set(texts)
    missing = sorted(
        t for t in unique if hashlib.sha256(t.encode()).hexdigest() not in cache
    )
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(enc.encode(t, disallowed_special=())) for t in missing)
    return {
        "children": len(texts),
        "median_chars": statistics.median(map(len, texts)) if texts else 0,
        "max_chars": max(map(len, texts), default=0),
        "unique_texts": len(unique),
        "missing_unique_texts": len(missing),
        "estimated_tokens": tokens,
        "estimated_usd": tokens / 1e6 * run.PRICE_PER_M_TOKENS,
        "price_per_m_tokens_usd": run.PRICE_PER_M_TOKENS,
        "tokenizer": "cl100k_base",
        "embedding_performed": False,
        "permission": "нужно разрешение Петра" if missing else "все тексты в кеше",
    }


def read_jsonl(name):
    return [json.loads(line) for line in (run.OUT / name).read_text().splitlines()]


def jsonl(name, rows):
    (run.OUT / name).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )


def read_metadata() -> dict:
    with sqlite3.connect(f"file:{run.CURRENT_DB}?mode=ro", uri=True) as con:
        data = {}
        for i, key, value in con.execute(
            "select id,key,coalesce(string_value,cast(int_value as text)) from embedding_metadata "
            "where key in ('source','chunk_id','article','part','heading_path')"
        ):
            data.setdefault(i, {})[key] = value
    return {
        (m["source"], m["chunk_id"]): m
        for m in data.values()
        if m.get("source") in run.DOCS and "chunk_id" in m
    }


def alignment_run(paras, current, metadata, rollback):
    aligned = {}
    for doc in run.DOCS:
        units = [
            Norm(doc, u.kind, u.number, tuple(u.paragraphs), i)
            for i, u in enumerate(parse_units(paras[doc]))
        ]
        indices = [i for i, row in enumerate(current) if row[0] == doc]
        cs = []
        for i in indices:
            src, cid, text, heading = current[i]
            meta = metadata[(src, cid)]
            cs.append(
                AlignInput(
                    run.strip_heading(text, heading),
                    meta.get("article", ""),
                    meta.get("part", ""),
                    meta.get("heading_path", ""),
                )
            )
        for i, a in zip(indices, align_document(cs, units, rollback=rollback)):
            aligned[i] = a
    return aligned


def alignment_stats(aligned, current):
    return {
        "chunks": len(current),
        "without_parent": sum(a.parent is None for a in aligned.values()),
        "reasons": dict(Counter(a.reason for a in aligned.values() if a.reason)),
        "rollbacks": sum(a.rollback for a in aligned.values()),
        "score_sum": sum(a.score for a in aligned.values()),
        "preamble_parents": sum(
            a.parent is not None and a.parent.kind == "preamble"
            for a in aligned.values()
        ),
        "by_doc": {
            doc: {
                "chunks": sum(r[0] == doc for r in current),
                "without_parent": sum(
                    aligned[i].parent is None
                    for i, r in enumerate(current)
                    if r[0] == doc
                ),
            }
            for doc in run.DOCS
        },
    }


def outcome(context, q, scheme):
    metrics = [
        context_metrics(context, n, "A" if scheme == "A" else "B") for n in q.norms
    ]
    return Outcome(
        tuple(m[0] for m in metrics),
        tuple(m[1] for m in metrics),
        tuple(m[2] for m in metrics),
        len("\n".join(c.text for c in context)),
        tuple(c.chunk_index for c in context),
    )


def main() -> int:
    # Физический запрет сетевого fallback, включая загрузку файлов токенизатора.
    def offline(*args, **kwargs):
        raise RuntimeError("network disabled: step2 is offline")

    socket.socket.connect = offline
    socket.create_connection = offline
    paras = run.load_paragraphs()
    current = run.read_chunks(run.CURRENT_DB, set(run.DOCS))
    metadata = read_metadata()
    saved_norms = read_jsonl("step1_norms.jsonl")
    norms = {
        r["id"]: Norm(
            r["doc"],
            r["kind"],
            r["number"],
            tuple(r["paragraphs"]),
            int(r["id"].rsplit(":", 1)[1]),
        )
        for r in saved_norms
    }
    questions = [
        Question(
            r["question"],
            tuple(norms[n["id"]] for n in r["norms"]),
            r["status"],
            tuple(r["labels"]),
        )
        for r in read_jsonl("norm_labels.jsonl")
    ]
    old_parents = read_jsonl("step1_parents.jsonl")
    if len(current) != len(old_parents):
        raise RuntimeError("step1/current chunk count differs")
    for i, (doc, cid, text, heading) in enumerate(current):
        if (doc, cid) != (old_parents[i]["doc"], old_parents[i]["chunk_id"]):
            raise RuntimeError("step1/current chunk order differs")
    texts = [r[2] for r in current]
    qtexts = [q.text for q in questions]
    vectors = cached_vectors(texts + qtexts, run.OUT / "embed_cache.pkl")
    paths = [
        run.CURRENT_DB,
        run.OUT / "embed_cache.pkl",
        *(run.ROOT / "source_docs" / fname for fname, _ in run.DOCS.values()),
        *(
            run.OUT / f
            for f in (
                "step1_parents.jsonl",
                "step1_norms.jsonl",
                "norm_labels.jsonl",
                "step1_results.json",
            )
        ),
        Path(__file__),
        Path(__file__).with_name("align.py"),
        Path(run.__file__),
        Path(__file__).with_name("norms.py"),
        Path(__file__).with_name("chunkers.py"),
    ]
    hashes = {
        str(p.relative_to(run.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    # Только alignment-статистика: не смотрим retrieval при выборе параметров.
    candidates = {}
    for rollback in (0, 1):
        aligned = alignment_run(paras, current, metadata, rollback)
        candidates[rollback] = aligned
        print(
            f"alignment rollback={rollback}: {alignment_stats(aligned, current)}",
            flush=True,
        )
    # Выбор после ручного аудита зафиксирован до метрик; см. step2_report.
    chosen_rollback = 0
    aligned = candidates[chosen_rollback]
    a_chunks, b_chunks, parents = [], [], []
    for i, (doc, cid, text, heading) in enumerate(current):
        body = run.strip_heading(text, heading)
        old = norms.get(old_parents[i]["norm_id"])
        a = aligned[i]
        a_chunks.append(Chunk(doc, text, body, old))
        b_chunks.append(Chunk(doc, text, body, a.parent))
        parents.append(
            {
                "index": i,
                "doc": doc,
                "chunk_id": cid,
                "norm_id": a.parent.id if a.parent else None,
                "candidate_id": a.candidate.id if a.candidate else None,
                "score": a.score,
                "best_score": a.best_score,
                "metadata_bonus": a.bonus,
                "threshold": THRESHOLD,
                "rollback_allowed": chosen_rollback,
                "rollback": a.rollback,
                "reason": a.reason,
                "step1_parent": old_parents[i]["norm_id"],
                "metadata": {
                    k: metadata[(doc, cid)].get(k)
                    for k in ("article", "part", "heading_path")
                },
                "body_preview": body[:200],
            }
        )
    jsonl("step2_parents.jsonl", parents)
    print("alignment saved; ranking cache-only", flush=True)
    ranks = run.rank_lists(texts, vectors[: len(texts)], qtexts, vectors[len(texts) :])
    outcomes, details, special, losses, doc_metrics = {}, [], [], [], {}
    for retriever in ("hybrid", "dense", "bm25"):
        for scheme, budget in [
            (s, b) for b in (8000, 16000) for s in ("A", "B1", "B2")
        ] + [("Bdiag", None)]:
            key = f"{retriever}/{budget}/{scheme}"
            rows = []
            for qi, q in enumerate(questions):
                chunks = a_chunks if scheme == "A" else b_chunks
                ctx = select_context(ranks[retriever][qi], chunks, scheme, budget)
                out = outcome(ctx, q, scheme)
                rows.append(out)
                details.append(
                    {
                        "retriever": retriever,
                        "budget": budget,
                        "scheme": scheme,
                        "question_index": qi,
                        "question": q.text,
                        "chars": out.chars,
                        "selected_C_indices": out.chunks,
                        "context": [
                            {
                                "C_index": c.chunk_index,
                                "parent": c.parent.id if c.parent else None,
                                "chars": len(c.text),
                                "expanded": c.expanded,
                                "text_sha256": hashlib.sha256(
                                    c.text.encode()
                                ).hexdigest(),
                            }
                            for c in ctx
                        ],
                        "norms": [
                            {
                                "norm_id": n.id,
                                "found": f,
                                "complete": c,
                                "coverage": cov,
                            }
                            for n, f, c, cov in zip(
                                q.norms, out.found, out.complete, out.coverage
                            )
                        ],
                    }
                )
                for case, text, doc, kind, number in (
                    (
                        "G005",
                        "В каких случаях проводится внеплановый инструктаж?",
                        "2464.md",
                        "item",
                        "16",
                    ),
                    (
                        "G047",
                        "Кто входит в комиссию по расследованию несчастного случая на производстве?",
                        "тк рф.pdf",
                        "article",
                        "229",
                    ),
                ):
                    if q.text == text:
                        targets = [
                            n
                            for n in norms.values()
                            if (n.doc, n.kind, n.number) == (doc, kind, number)
                        ]
                        if len(targets) != 1:
                            raise RuntimeError(f"{case}: target count {len(targets)}")
                        f, c, cov = context_metrics(
                            ctx, targets[0], "A" if scheme == "A" else "B"
                        )
                        special.append(
                            {
                                "case": case,
                                "norm_id": targets[0].id,
                                "retriever": retriever,
                                "budget": budget,
                                "scheme": scheme,
                                "found": f,
                                "complete": c,
                                "coverage": cov,
                            }
                        )
            outcomes[key] = rows
            for doc in run.DOCS:
                subset = []
                for q, out in zip(questions, rows):
                    js = [j for j, n in enumerate(q.norms) if n.doc == doc]
                    if js:
                        subset.append(
                            Outcome(
                                tuple(out.found[j] for j in js),
                                tuple(out.complete[j] for j in js),
                                tuple(out.coverage[j] for j in js),
                                out.chars,
                                out.chunks,
                            )
                        )
                doc_metrics[f"{doc}/{key}"] = summarize(subset)
        for budget in (8000, 16000):
            aa, bb, dd = (
                outcomes[f"{retriever}/{b}/{s}"]
                for b, s in ((budget, "A"), (budget, "B1"), (None, "Bdiag"))
            )
            for qi, (q, a, b, d) in enumerate(zip(questions, aa, bb, dd)):
                per_norm = []
                for j, n in enumerate(q.norms):
                    reason = loss_reason(
                        a.found[j],
                        b.found[j],
                        d.found[j],
                        n,
                        ranks[retriever][qi],
                        b_chunks,
                        a_chunks,
                    )
                    if reason:
                        per_norm.append(reason)
                        losses.append(
                            {
                                "level": "norm",
                                "retriever": retriever,
                                "budget": budget,
                                "question_index": qi,
                                "question": q.text,
                                "norm_id": n.id,
                                "reason": reason,
                                "A_indices": a.chunks,
                                "B1_indices": b.chunks,
                                "top100_parentless": [
                                    int(i)
                                    for i in ranks[retriever][qi]
                                    if b_chunks[i].parent is None
                                ],
                            }
                        )
                if any(a.found) and not any(b.found):
                    reason = (
                        "budget"
                        if any(d.found)
                        else "missing_parent"
                        if "missing_parent" in per_norm
                        else "other"
                    )
                    losses.append(
                        {
                            "level": "question",
                            "retriever": retriever,
                            "budget": budget,
                            "question_index": qi,
                            "question": q.text,
                            "reason": reason,
                            "norm_ids": [n.id for n, f in zip(q.norms, a.found) if f],
                        }
                    )
    aggregates = {k: summarize(v) for k, v in outcomes.items()}
    # A обязан точно воспроизвести шаг 1, иначе сравнение прерывается.
    old_results = json.loads((run.OUT / "step1_results.json").read_text())
    for r in run.RETRIEVERS:
        for b in (8000, 16000):
            if aggregates[f"{r}/{b}/A"] != old_results["metrics"][f"{r}/{b}/A"]:
                raise RuntimeError("A baseline changed")
    apos = {
        f"{r}/{b}": summarize(
            [
                outcome(select_context(order, b_chunks, "A", b), q, "A")
                for order, q in zip(ranks[r], questions)
            ]
        )
        for r in run.RETRIEVERS
        for b in (8000, 16000)
    }
    legacy_window = {
        f"{r}/{b}": summarize(
            [
                outcome(old_select_context(order, b_chunks, "B", b), q, "B1")
                for order, q in zip(ranks[r], questions)
            ]
        )
        for r in run.RETRIEVERS
        for b in (8000, 16000)
    }
    children = [
        c
        for doc, ps in paras.items()
        for c in build_children(doc, run.DOC_TITLE[doc], ps)
    ]
    stats = k_stats([c.text for c in children], run.OUT / "embed_cache.pkl")
    stats.update(
        {
            "word_fallback_children": sum(c.word_fallback for c in children),
            "shortened_intro_children": sum(c.intro_shortened for c in children),
            "parents": len({c.parent_id for c in children}),
            "by_doc": {
                doc: sum(c.parent_id.startswith(doc + "::") for c in children)
                for doc in run.DOCS
            },
        }
    )
    both = [p for p in parents if p["norm_id"] and p["step1_parent"]]
    compare = {
        "both": len(both),
        "same": sum(p["norm_id"] == p["step1_parent"] for p in both),
        "old_without": sum(p["step1_parent"] is None for p in parents),
        "new_without": sum(p["norm_id"] is None for p in parents),
    }
    results = {
        "questions": len(questions),
        "norm_instances": sum(len(q.norms) for q in questions),
        "metrics": aggregates,
        "by_doc": doc_metrics,
        "special": special,
        "alignment": {
            str(k): alignment_stats(v, current) for k, v in candidates.items()
        },
        "chosen_rollback": chosen_rollback,
        "threshold": THRESHOLD,
        "rollback_penalty": ROLLBACK_PENALTY,
        "parent_comparison": compare,
        "A_pos_sensitivity": apos,
        "B_step1_window_pos_sensitivity": legacy_window,
        "B_step1_metrics": {
            k: v for k, v in old_results["metrics"].items() if k.endswith("/B")
        },
        "loss_counts": {
            f"{r}/{b}/{level}": dict(
                Counter(
                    x["reason"]
                    for x in losses
                    if x["retriever"] == r and x["budget"] == b and x["level"] == level
                )
            )
            for r in run.RETRIEVERS
            for b in (8000, 16000)
            for level in ("question", "norm")
        },
        "k_variant": stats,
        "input_sha256": hashes,
    }
    for p in paths:
        if (
            hashlib.sha256(p.read_bytes()).hexdigest()
            != hashes[str(p.relative_to(run.ROOT))]
        ):
            raise RuntimeError(f"input changed: {p}")
    jsonl("step2_details.jsonl", details)
    jsonl("step2_losses.jsonl", losses)
    jsonl("step2_k_children.jsonl", [asdict(c) for c in children])
    write_json(run.OUT / "step2_k_variant_stats.json", stats)
    write_json(run.OUT / "step2_results.json", results)
    report = render_report(results, parents, losses, current, candidates)
    (run.OUT / "step2_report.md").write_text(report)
    print(report[:6000], flush=True)
    return 0


def render_report(results, parents, losses, current, candidates):
    m = results["metrics"]
    a, b, d = (
        m[f"hybrid/{budget}/{scheme}"]
        for budget, scheme in ((8000, "A"), (8000, "B1"), (None, "Bdiag"))
    )
    lines = [
        f"Потеря found при 8000 {'сохраняется' if b['q_found'] < a['q_found'] else 'уходит'}, без бюджета {'уходит' if d['q_found'] >= a['q_found'] else 'сохраняется'}; выигрыш complete {'сохраняется' if b['q_complete'] > a['q_complete'] else 'не сохраняется'}.",
        "",
        "# Шаг 2 — позиционный родитель и бюджет",
        "",
        "Offline/cache-only; снимок build-8cs1plb3, без реранкера/генератора. A точно совпал со всеми агрегатами шага 1. "
        "Метки и выборка прочитаны из неизменённых norm_labels.jsonl/step1_norms.jsonl: 117 вопросов, 190 пар вопрос×норма. "
        "Вопрос засчитан при хотя бы одной норме; два вопроса без подтверждённых норм остаются в знаменателе. "
        "Complete = ≥90% уникальных слов нормы в объединении реально отправленных тел того же документа. "
        "Found: присвоенный родитель; у A дополнительно ≥60% слов нормы в одном C. Это разные и несовершенные критерии.",
        "",
        "B1 раскрывает **полный текст** любого родителя. В отличие от шага 1 скрытого окна для родителей >6000 нет; "
        "Bdiag раскрывает все уникальные родители top100 без бюджета. B2 — максимум 3000 знаков вокруг якоря, "
        "сначала точное совпадение последовательности токенов, иначе лучший overlap абзаца. "
        "Окно растёт целыми абзацами; если сам якорь >3000, fallback по предложениям/словам. "
        "При повторном попадании родителя сохраняется первое окно. Без родителя — исходный C. "
        "Строгий бюджет с переводами строк, остановка на первом переполнении; это может дать пустой контекст B1. "
        "A использует привязки шага 1, чтобы изменение самого found не выдавало себя за retrieval-эффект.",
        "",
        "## Основная таблица (hybrid — заглавный)",
        "",
        "| поиск | бюджет | схема | found вопросов | complete вопросов | found норм | complete норм | средний контекст |",
        "|---|---:|---|---|---|---|---|---:|",
    ]
    for r in ("hybrid", "dense", "bm25"):
        for budget in (8000, 16000, None):
            for scheme in ("Bdiag",) if budget is None else ("A", "B1", "B2"):
                v = m[f"{r}/{budget}/{scheme}"]
                lines.append(
                    f"| {r} | {budget or '∞ (top100)'} | {scheme} | {metric_cells(v)} | {v['mean_chars']:.0f} |"
                )
    lines += [
        "",
        "## Выравнивание: выбор до retrieval",
        "",
        "DP максимизирует сумму overlap (уникальные слова ≥4 знаков C / слов C) + подсказки metadata ≤0.06 "
        "+ специфичность ≤0.01 для словарных ничьих. Article +0.03, part +0.015, heading_path overlap ×0.015. "
        "Переход вперёд/на месте бесплатный; откат к предыдущей единице штрафуется 0.15. "
        "Откат ограничен одним относительно предыдущего состояния (не глобального максимума пути). "
        "Порог 0.60 оставлен как в шаге 1: путь не превращает слабое совпадение в родителя. "
        "Нет отказа за близость двух кандидатов: неоднозначность разрешает последовательность. "
        "В DP включены вводные parse_units: им разрешён родитель preamble, но они не добавлены к целевым нормам. "
        "Отказы empty_words/below_threshold/no_units, score, лучший локальный score, выбранный кандидат, "
        "metadata и rollback записаны в step2_parents.jsonl.",
        "",
        "| откат | без родителя | сумма overlap | откатов | preamble |",
        "|---:|---|---:|---:|---:|",
    ]
    for rollback, st in results["alignment"].items():
        lines.append(
            f"| {rollback} | {pct(st['without_parent'], st['chunks'])} | {st['score_sum']:.2f} | {st['rollbacks']} | {st['preamble_parents']} |"
        )
    cmp = results["parent_comparison"]
    lines += [
        "",
        f"Выбран откат {results['chosen_rollback']}: оба пути совпали, разрешение отката не использовано ни разу. "
        "Строгая монотонность проще и подтверждается порядком C. Порог 0.60 даёт 2.0% отказов; "
        "просмотр отказов показывает преимущественно чанки на стыке нескольких пунктов и служебный текст. "
        "Порог не снижался ради цели <5% и не выбирался по found.",
        f"Без родителя до: {pct(cmp['old_without'], len(parents))}; после: {pct(cmp['new_without'], len(parents))}. "
        f"Совпали среди привязанных обоими: {pct(cmp['same'], cmp['both'])}.",
        "",
        "| документ | без родителя до | после | preamble после |",
        "|---|---|---|---:|",
    ]
    for doc in run.DOCS:
        ps = [p for p in parents if p["doc"] == doc]
        lines.append(
            f"| {doc} | {pct(sum(p['step1_parent'] is None for p in ps), len(ps))} | "
            f"{pct(sum(p['norm_id'] is None for p in ps), len(ps))} | {sum(p['norm_id'] is not None and '::preamble:' in p['norm_id'] for p in ps)} |"
        )
    lines += [
        "",
        "### 10 расхождений (7 сменившихся родителей и 3 восстановленные привязки)",
        "",
        "| C | шаг 1 | шаг 2 | overlap | текст C |",
        "|---|---|---|---:|---|",
    ]
    differences = [p for p in parents if p["norm_id"] != p["step1_parent"]]
    examples = [p for p in differences if p["norm_id"] and p["step1_parent"]]
    for doc, marker in (
        ("тк рф.pdf", "::article:229:"),
        ("426-ФЗ.pdf", "::article:4:"),
        ("2464.md", "::item:"),
    ):
        recovered = [
            p
            for p in differences
            if p["doc"] == doc
            and p["step1_parent"] is None
            and p["norm_id"]
            and marker in p["norm_id"]
            and len(p["body_preview"]) > 100
            and p["score"] >= 0.9
            and not p["body_preview"].startswith("(В ")
        ]
        examples.extend(recovered[:1])
    for p in examples[:10]:
        lines.append(
            f"| {p['doc']}#{p['chunk_id']} | {p['step1_parent']} | {p['norm_id']} | {p['score']:.3f} | {p['body_preview'].replace('|', '/').replace(chr(10), ' ')} |"
        )
    lines += ["", "### Аудит откатов и отказов", ""]
    for p in [p for p in parents if p["rollback"]][:10]:
        lines.append(
            f"- Откат {p['doc']}#{p['chunk_id']}: {p['candidate_id']}, overlap={p['score']:.3f}; {p['body_preview'].replace(chr(10), ' ')}"
        )
    for p in [p for p in parents if p["reason"]][:10]:
        lines.append(
            f"- Отказ {p['doc']}#{p['chunk_id']}: {p['reason']}, overlap={p['score']:.3f}, local best={p['best_score']:.3f}; {p['body_preview'].replace(chr(10), ' ')}"
        )
    lines += [
        "",
        "### Чувствительность found A к новой привязке (тот же исходный контекст)",
        "",
        "| поиск | бюджет | A старые родители | A позиционные родители |",
        "|---|---:|---|---|",
    ]
    for key, v in results["A_pos_sensitivity"].items():
        r, budget = key.split("/")
        old = m[f"{key}/A"]
        lines.append(
            f"| {r} | {budget} | {pct(old['q_found'], 117)} | {pct(v['q_found'], 117)} |"
        )
    lines += [
        "",
        "### Изоляция эффекта привязки от смены длинного окна",
        "",
        "Дополнительный контроль: раскрытие в точности по expand_parent шага 1 (порог 6000, "
        "окно по 3000 с каждой стороны), изменены только родители. Это не B1 основной таблицы. "
        "Он отделяет эффект привязки от перехода к действительно полному родителю.",
        "",
        "| поиск | бюджет | B шага 1 found / complete | только новая привязка found / complete |",
        "|---|---:|---|---|",
    ]
    for key, v in results["B_step1_window_pos_sensitivity"].items():
        r, budget = key.split("/")
        old = results["B_step1_metrics"][f"{key}/B"]
        lines.append(
            f"| {r} | {budget} | {old['q_found']}/117 / {old['q_complete']}/117 | "
            f"{v['q_found']}/117 / {v['q_complete']}/117 |"
        )
    lines += [
        "",
        "## Причины потерь found (A нашёл, B1 нет)",
        "",
        "Классы взаимоисключающие: budget — целевая норма находится в Bdiag; missing_parent — в top100 есть "
        "parentless C, который имел эту норму родителем в шаге 1 либо покрыл бы ≥60% её слов; other — остальное (в том числе чужой "
        "присвоенный родитель или found A только по старой привязке). Для вопроса сначала проверяется any(Bdiag), "
        "затем missing_parent среди потерянных норм. Нормы и вопросы считаются отдельно. "
        "Это операционная диагностика, а не доказательство правильности нового родителя.",
        "",
        "| поиск | бюджет | уровень | бюджет | без родителя | прочее |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in ("hybrid", "dense", "bm25"):
        for budget in (8000, 16000):
            for level in ("question", "norm"):
                c = results["loss_counts"][f"{r}/{budget}/{level}"]
                lines.append(
                    f"| {r} | {budget} | {level} | {c.get('budget', 0)} | {c.get('missing_parent', 0)} | {c.get('other', 0)} |"
                )
    for reason in ("budget", "missing_parent", "other"):
        lines += ["", f"### Примеры: {reason}", ""]
        rows = [x for x in losses if x["reason"] == reason and x["level"] == "norm"]
        seen, selected = set(), []
        for x in rows:
            if x["question"] not in seen:
                selected.append(x)
                seen.add(x["question"])
        if len(selected) < 5:
            lines.append(
                f"Всего разных вопросов с потерей нормы этой категории: {len(selected)}; пяти примеров нет."
            )
        for x in selected[:5]:
            lines.append(
                f"- {x['retriever']}/{x['budget']}, {x['question']} — {x['norm_id']}; "
                f"C в A={x['A_indices']}, C в B1={x['B1_indices']}; parentless top100={x['top100_parentless']}."
            )
    lines += [
        "",
        "## G005 / G047 — именно п.16 / ст.229",
        "",
        "Общий успех G005 по п.17 не заменяет целевой п.16 в этой таблице.",
        "",
        "| кейс | норма | поиск | бюджет | схема | found | complete | покрытие |",
        "|---|---|---|---:|---|---|---|---:|",
    ]
    for x in sorted(
        results["special"],
        key=lambda x: (
            ("hybrid", "dense", "bm25").index(x["retriever"]),
            x["budget"] or 999999,
            x["case"],
            x["scheme"],
        ),
    ):
        lines.append(
            f"| {x['case']} | {x['norm_id']} | {x['retriever']} | {x['budget'] or '∞'} | {x['scheme']} | {x['found']} | {x['complete']} | {x['coverage']:.1%} |"
        )
    st = results["k_variant"]
    lines += [
        "",
        "## K — только dry-run",
        "",
        "Дети строятся внутри единицы parse_units, parent_id задан при построении, включая preamble. "
        "Общая длина ≤450 с коротким префиксом документ › раздел › статья/пункт. "
        "Границы абзацев/подпунктов, затем предложения; слишком длинное предложение режется по словам, "
        "слишком длинный токен — по символам. Вводная перед перечнем повторяется у каждого фрагмента подпункта, "
        "сокращается до ≤120 знаков и ≤1/3 доступного тела с явным многоточием. Оригинальная вводная также "
        "сохраняется отдельными детьми. Это компромисс строгого ≤450, а не гарантия полной вводной в каждом ребёнке.",
        "",
        f"Детей {st['children']}; медиана {st['median_chars']} знаков; максимум {st['max_chars']}; "
        f"уникальных текстов {st['unique_texts']}; отсутствуют в кеше {st['missing_unique_texts']}. "
        f"Новых токенов cl100k_base {st['estimated_tokens']:,}; тариф run.py ${st['price_per_m_tokens_usd']}/млн; "
        f"оценка **${st['estimated_usd']:.6f}**. Эмбеддинги НЕ выполнялись: **нужно разрешение Петра**.",
        f"Детей с fallback по словам {st['word_fallback_children']}, с сокращённой повторной вводной {st['shortened_intro_children']}. "
        "Нарезка K не участвовала в retrieval-замере.",
        "",
        "## По документам",
        "",
        "Вопросы с подтверждённой нормой этого документа, оцениваются только его нормы. "
        "Сумма вопросов по документам может превышать 117; сумма норм 190.",
        "",
        "| документ | поиск | бюджет | схема | found вопросов | complete вопросов | found норм | complete норм |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for doc in run.DOCS:
        for r in ("hybrid", "dense", "bm25"):
            for budget in (8000, 16000, None):
                for scheme in ("Bdiag",) if budget is None else ("A", "B1", "B2"):
                    v = results["by_doc"][f"{doc}/{r}/{budget}/{scheme}"]
                    lines.append(
                        f"| {doc} | {r} | {budget or '∞'} | {scheme} | {metric_cells(v)} |"
                    )
    lines += [
        "",
        "## Ограничения и воспроизведение",
        "",
        "Порог фиксирован по привязкам, без оптимизации по found. Низкая доля отказов не доказывает правильность "
        "выравнивания: повторяющиеся тексты, порядок PDF/Docling, границы приложений и ошибки parse_units остаются. "
        "Метки из C и спорные метки шага 1 не исправлены; статус/редакция НПА не проверялись. "
        "Found от parent_id может быть истинным даже при малом содержательном overlap окна B2; complete лексический, "
        "не юридическая полнота. Bdiag — диагностический потолок на top100, не рабочая конфигурация. "
        "Перенос эффекта на генерацию и весь индекс не измерен.",
        "",
        "Запуск: `.venv/bin/python -m eval.chunking_ab.step2`. Сокеты запрещены в main; при недостающих векторах C/вопросов "
        "cached_vectors прерывает прогон. K cache misses только считаются. "
        "step2_results.json: агрегаты, документы, контроль A, SHA256 входов; step2_details.jsonl: каждый контекст и "
        "покрытие норм; step2_losses.jsonl: каждый потерянный found; step2_parents.jsonl: все привязки; "
        "step2_k_children.jsonl/step2_k_variant_stats.json: воспроизводимый dry-run. Входы проверены на неизменность.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
