"""Step 3: observed routes and stratified step-2 metrics, offline/cache-only."""

# ANCHOR: saved route traces + fixed step-2 labels/contexts + cached vectors → step3_*.
# Reuses step1.summarize and step2.select_context/outcome; never imports src or LLM factory.
from __future__ import annotations

import ast
import hashlib
import json
import socket
from collections import Counter, defaultdict
from dataclasses import asdict

from eval.chunking_ab import run, step2
from eval.chunking_ab.norms import Chunk, Context, Norm
from eval.chunking_ab.routes import choose_route, key, scan_file
from eval.chunking_ab.step1 import (
    Outcome,
    Question,
    cached_vectors,
    metric_cells,
    summarize,
    write_json,
)

GROUPS = ("simple", "complex", "unknown")


def grouped_metrics(rows: list[Outcome], routes: list[str]) -> dict:
    if len(rows) != len(routes):
        raise ValueError("route/outcome count mismatch")
    return {
        g: summarize([r for r, route in zip(rows, routes) if route == g])
        for g in GROUPS
    }


def count_context(order: list[int], chunks: list[Chunk], top_k: int) -> list[Context]:
    return [
        Context(
            chunks[i].doc,
            chunks[i].text,
            chunks[i].body,
            chunks[i].parent,
            int(i),
            False,
        )
        for i in order[:top_k]
    ]


def config_default(name: str) -> int:
    tree = ast.parse((run.ROOT / "src/v7/config.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
    raise ValueError(name)


def main() -> int:
    def offline(*args, **kwargs):
        raise RuntimeError("network disabled: step3 is offline")

    socket.socket.connect = offline
    socket.create_connection = offline
    labels = step2.read_jsonl("norm_labels.jsonl")
    details = step2.read_jsonl("step2_details.jsonl")
    current = run.read_chunks(run.CURRENT_DB, set(run.DOCS))
    texts = [r[2] for r in current]
    qtexts = [r["question"] for r in labels]
    # Fail before writing any results if a vector is missing.
    vectors = cached_vectors(texts + qtexts, run.OUT / "embed_cache.pkl")
    candidates = sorted(
        {
            *(run.ROOT / "benchmarks").rglob("*.json*"),
            *(run.ROOT / "eval/runs").rglob("*.json*"),
            run.ROOT / "logs/events.jsonl",
        }
    )
    evidence = defaultdict(list)
    inventory = []
    for path in candidates:
        if (
            path.name.startswith("step3_")
            or not path.is_file()
            or path.suffix not in (".json", ".jsonl")
        ):
            continue
        # Step-0/1/2 contexts and generated children contain no runtime paths.
        if path.parent == run.OUT:
            continue
        rows = scan_file(path)
        inventory.append(
            {
                "file": str(path.relative_to(run.ROOT)),
                "observations": len(rows),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        for question, e in rows:
            evidence[key(question)].append(e)
    routes = [choose_route(q, evidence[key(q)]) for q in qtexts]
    route_names = [r.route for r in routes]
    norms = {
        r["id"]: Norm(
            r["doc"],
            r["kind"],
            r["number"],
            tuple(r["paragraphs"]),
            int(r["id"].rsplit(":", 1)[1]),
        )
        for r in step2.read_jsonl("step1_norms.jsonl")
    }
    questions = [
        Question(
            r["question"],
            tuple(norms[n["id"]] for n in r["norms"]),
            r["status"],
            tuple(r["labels"]),
        )
        for r in labels
    ]
    parents = step2.read_jsonl("step1_parents.jsonl")
    if len(parents) != len(current) or any(
        (r[0], r[1]) != (p["doc"], p["chunk_id"]) for r, p in zip(current, parents)
    ):
        raise RuntimeError("step1/current chunk identity differs")
    chunks = [
        Chunk(doc, text, run.strip_heading(text, heading), norms.get(p["norm_id"]))
        for (doc, cid, text, heading), p in zip(current, parents)
    ]
    metrics, all_rows = {}, defaultdict(list)
    for d in details:
        all_rows[f"{d['retriever']}/{d['budget']}/{d['scheme']}"].append(d)
    for name, rows in all_rows.items():
        rows.sort(key=lambda r: r["question_index"])
        if [r["question"] for r in rows] != qtexts:
            raise RuntimeError("step2 question identity differs")
        outcomes = [
            Outcome(
                tuple(n["found"] for n in r["norms"]),
                tuple(n["complete"] for n in r["norms"]),
                tuple(n["coverage"] for n in r["norms"]),
                r["chars"],
                tuple(r["selected_C_indices"]),
            )
            for r in rows
        ]
        metrics[name] = grouped_metrics(outcomes, route_names)
        # Every stratification must sum to the unchanged step2 aggregate.
        aggregate = summarize(outcomes)
        expected = json.loads((run.OUT / "step2_results.json").read_text())["metrics"][
            name
        ]
        if aggregate != expected:
            raise RuntimeError(f"step2 aggregate mismatch: {name}")
    ranks = run.rank_lists(texts, vectors[: len(texts)], qtexts, vectors[len(texts) :])
    top_k = config_default("FINAL_MERGE_TOP_K")
    count_details = []
    for retriever in run.RETRIEVERS:
        outcomes = []
        for qi, q in enumerate(questions):
            ctx = count_context(ranks[retriever][qi], chunks, top_k)
            out = step2.outcome(ctx, q, "A")
            outcomes.append(out)
            if route_names[qi] == "complex":
                count_details.append(
                    {
                        "question": q.text,
                        "retriever": retriever,
                        "top_k": top_k,
                        **asdict(out),
                    }
                )
        metrics[f"{retriever}/top{top_k}/A"] = grouped_metrics(outcomes, route_names)
    special = []
    # Step2 already selected the specific target norms, not any norm of G005/G047.
    saved_special = json.loads((run.OUT / "step2_results.json").read_text())["special"]
    case_questions = {}
    import csv

    with (run.ROOT / "tests/dataset.csv").open() as f:
        for row in csv.DictReader(f):
            if row.get("case_id") in ("G005", "G047"):
                case_questions[row["case_id"]] = row["question"]
    for row in saved_special:
        q = case_questions[row["case"]]
        route = routes[qtexts.index(q)]
        special.append({**row, "route": route.route})
    results = {
        "groups": dict(Counter(route_names)),
        "metrics": metrics,
        "special": special,
        "route_policy": "latest timestamp; exact question with whitespace normalization; latest ties conflicting => unknown",
        "b_validation": {
            "n": 0,
            "agreement": None,
            "reason": "no production simple cache (full corpus, expanded queries, gap packing); b not attempted",
        },
        "scanned_files": inventory,
        "historical_conflicts": sum(r.conflicting_history for r in routes),
        "final_merge_top_k": top_k,
        "input_sha256": {
            str(p.relative_to(run.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run.OUT / n
                for n in (
                    "step2_details.jsonl",
                    "step2_results.json",
                    "norm_labels.jsonl",
                    "step1_parents.jsonl",
                    "step1_norms.jsonl",
                )
            ]
        },
    }
    step2.jsonl(
        "step3_routes.jsonl",
        [dict(question_index=i, **asdict(r)) for i, r in enumerate(routes)],
    )
    step2.jsonl("step3_count_details.jsonl", count_details)
    write_json(run.OUT / "step3_results.json", results)
    report = render_report(results, routes)
    (run.OUT / "step3_report.md").write_text(report)
    print(report[:4500])
    return 0


def render_report(results, routes):
    m = results["metrics"]
    n = results["groups"].get("complex", 0)
    a = m["hybrid/8000/A"]["complex"]
    b = m["hybrid/8000/B2"]["complex"]
    a16, b16 = (m[f"hybrid/16000/{s}"]["complex"] for s in ("A", "B2"))
    lines = [
        f"На complex (n={n}) B2 даёт выигрыш complete {a['q_complete']}→{b['q_complete']} при 8000 и {a16['q_complete']}→{b16['q_complete']} при 16000 с потерей found {a['q_found']}→{b['q_found']} / {a16['q_found']}→{b16['q_found']}; "
        + ("вывод неустойчив (n<20)." if n < 20 else "эффект неоднозначен."),
        "",
        "# Шаг 3 — маршруты и группы",
        "",
        "## Достоверность маршрутов",
        "",
        f"Группы: {results['groups']}; unknown {results['groups'].get('unknown', 0)}/117 ({results['groups'].get('unknown', 0) / 117:.1%}). "
        f"Исторические расхождения: {results['historical_conflicts']} вопросов. Все свидетельства сохранены в step3_routes.jsonl; selected — выбранный файл и строка вопроса в том же объекте с path/stage.",
        "",
        "Источник (a): точное совпадение текста после нормализации пробелов; выбирается самое позднее датированное наблюдение, при конфликте на одной дате/времени — unknown. Дата: ts/timestamp/date объекта или контейнера, затем дата в пути. Это наблюдавшийся маршрут, а не гарантия маршрута сегодня. Старые снимки/флаги и стохастическое multi-query могут менять ветку. Путь из контейнера retrieval-only отчёта не наследуется дочерним вопросам; abstain не считается simple, но явный stage=complex доказывает посещение ветки.",
        "",
        f"Проверено файлов: {len(results['scanned_files'])} (инвентарь и SHA256 в step3_results.json). Источники: benchmarks/, eval/runs/, logs/events.jsonl. Локальные golden/LangSmith-экспорты в этих каталогах входят в поиск.",
        "",
        "Источник (b) не применялся: кеш шага 2 покрывает 6871 C семи документов против 8921 чанка снимка; rank_lists считает top100 RRF исходного вопроса, а production — dense/BM25 top12 по glossary-expanded запросу, возможно LLM multi-query, гарантия BM25 top3, visual enrichment и crossref-expansion. Косинус из eval не равен Chroma 1/(1+d). Подменять этим check_full_triage нельзя. Проверка (b) против (a): n=0, доля совпадений не определена. Источник (c): нет достоверного наблюдения либо конфликт последних.",
        "",
        "В рабочем дереве нет src/graph.py и src/router.py; актуальны src/v7/graph.py и src/v7/nodes/router.py. Router/plan — keyword-классификация и glossary, без LLM (router.py:53–110). rag_simple:75–86 допускает LLM multi-query при включённом флаге; plan сам LLM не требует. check_full_triage (hard_gates.py:124) детерминирован: top_score≥0.50, passage_count≥5, keyword_overlap_active≥0.15; require_multi_doc добавляет diversity. evaluate_triage:49–77 сначала enrich/pack/validate; gap из top5 закрывается добавлением в хвост (pack_context.py:149–188), затем decide_simple:68–113 эскалирует при незакрытых refs/enumeration, drift или плохом triage. Даже при достаточных трёх hard gates путь может стать complex.",
        "",
        "## Hybrid по группам",
        "",
        "Знаменатели как в шаге 2: успех вопроса = хотя бы одна его норма; вопросы без норм остаются в n. Нормы — пары вопрос×норма. Found A: parent_id либо ≥60% слов нормы в одном C; B: parent_id. Complete: ≥90% слов нормы в объединении тел того же документа. Метки и позиционные привязки не менялись; B2 — окно ≤3000, первый parent выигрывает, дедупликация по parent_id; B1 полный parent, Bdiag все уникальные родители top100 без бюджета. Bdiag приводится один раз на группу: от 8000/16000 не зависит.",
        "",
    ]

    def table(retrievers):
        out = [
            "| группа | поиск | бюджет | схема | found вопросов | complete вопросов | found норм | complete норм |",
            "|---|---|---:|---|---|---|---|---|",
        ]
        for g in GROUPS:
            for r in retrievers:
                for budget, schemes in (
                    (8000, ("A", "B1", "B2")),
                    (16000, ("A", "B1", "B2")),
                    (None, ("Bdiag",)),
                ):
                    for s in schemes:
                        out.append(
                            f"| {g} | {r} | {budget or '∞ top100'} | {s} | {metric_cells(m[f'{r}/{budget}/{s}'][g])} |"
                        )
        return out

    lines += table(("hybrid",))
    lines += [
        "",
        "## Complex: контроль A с лимитом числа чанков",
        "",
        "FINAL_MERGE_TOP_K=24 (config.py:55) ограничивает только merged-кандидата ДО упаковки: simple anchors≤12 + MMR оставшегося reranked complex pool (evaluate_complex.py:36–64). Потом pack_context добавляет crossref, режет до MAX_CHUNKS_FOR_LLM=30 и PACK_TOKEN_BUDGET=60000 (с заголовками). Кроме merged рассматриваются last_attempt и fallback_snapshot. generate_answer получает принятый final_context целиком; bridge не обрезает его. Поэтому фраза FACTS «24 чанка видит генератор» — упрощение: сохранённые прогоны действительно показывают n_passages=30.",
        "",
        "Ниже запрошенный контроль A@24 без лимита знаков, на том же eval hybrid/dense/bm25 порядке. Это честный контроль числа чанков в эксперименте, **не восстановленный production complex baseline**: нет reranker/MMR, полного корпуса, crossref и исходного final_context. Его нельзя выдавать за измерение реального генератора. Реальный baseline по имеющемуся кешу недоступен.",
        "",
        "| поиск | схема | found вопросов | complete вопросов | found норм | complete норм | средний контекст, знаки |",
        "|---|---|---|---|---|---|---:|",
    ]
    for r in ("hybrid", "dense", "bm25"):
        v = m[f"{r}/top{results['final_merge_top_k']}/A"]["complex"]
        lines.append(
            f"| {r} | A@24 proxy | {metric_cells(v)} | {v['mean_chars']:.0f} |"
        )
    lines += [
        "",
        "## G005 / G047: целевые п.16 / ст.229",
        "",
        "| кейс | путь | целевая норма | бюджет | схема | found | complete | покрытие |",
        "|---|---|---|---:|---|---|---|---:|",
    ]
    for s in results["special"]:
        if s["retriever"] == "hybrid" and s["scheme"] in ("A", "B2"):
            lines.append(
                f"| {s['case']} | {s['route']} | {s['norm_id']} | {s['budget']} | {s['scheme']} | {s['found']} | {s['complete']} | {s['coverage']:.1%} |"
            )
    lines += [
        "",
        "G005 наблюдался на complex: B2 не находит целевой п.16 ни при 8000, ни при 16000; прирост общего complete вопроса по другой норме не решает этот кейс. G047 наблюдался на simple: при 16000 B2 снижает покрытие ст.229 с 91.6% до 74.4% и теряет complete. При флаге только для complex G047 останется на текущем A.",
        "",
        "## Как включить флагом",
        "",
        "Предлагаемый флаг COMPLEX_PARENT_CONTEXT_ENABLE: применять раскрытие к кандидатам complex в evaluate_complex._prepare после выбора/rerank/MMR и enrichment, перед pack_context и validate_context; отдельная ветка для packed fallback, сохраняя исходный snapshot неизменным. Не менять rag_simple и его triage. В rag_complex сохранять child для поиска и реранкера. Подстановка после валидации только в генераторе нарушит контракт: verdict должен проверять именно тот текст, который получит генератор.",
        "",
        "Параметры: parent_max_chars=3000, явный бюджет контекста (8000/16000 знаков для сравнения либо production token budget с заголовками), дедупликация по (snapshot, source, parent_id), стабильный порядок и политика первого окна, fallback к child без parent. Для длинной статьи — ограниченное окно вокруг позиции ребёнка, не полный текст. Раскрытие и crossref должны делить один финальный бюджет, обрезка до validate; provenance исходных child и их score сохранять отдельно от parent.",
        "",
        "При индексации нужны snapshot/source, parent_id, тип/номер статьи/пункта, parent_text или адрес хранилища, начало/конец родителя и child внутри исходного текста, child→parent по позиции, article/part/heading_path, URL/страница для цитат. Parent назначается при разборе/индексации, не лексическим поиском во время запроса. Здесь src и индексы не менялись.",
        "",
        "## Ограничения",
        "",
        "Группы отобраны по исторически наблюдавшимся маршрутам, не случайно; unknown нельзя распределять пропорционально 21%. Complete лексический, found от parent_id может быть true при малом покрытии окна. Нельзя заключить о качестве ответов генератора или юридической полноте. Схема B2 раскрывает текущие C; новый мелкий индекс K не измерен. Статус/редакция НПА в этом техническом замере не проверялись.",
        "",
        "## Приложение: dense / bm25",
        "",
    ]
    lines += table(("dense", "bm25"))
    lines += [
        "",
        "## Воспроизведение",
        "",
        "`.venv/bin/python -m eval.chunking_ab.step3`. Сокеты блокируются, cached_vectors останавливает запуск при cache miss. src не импортируется. Агрегаты шага 2 пересчитываются через step1.summarize и проверяются на точное совпадение; A@24 использует step2.outcome/context_metrics. step3_routes.jsonl — все вопросы/наблюдения/источники; step3_results.json — агрегаты/инвентарь; step3_count_details.jsonl — A@24 для complex.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
