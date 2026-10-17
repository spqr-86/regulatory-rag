"""A/B нарезок законов: поиск по одному корпусу, разные чанки (протокол: см. ниже).

# ANCHOR: протокол и правило решения — docs/evaluation/experiments/2026-10-17-law-chunking-ab.md.
# Вход: source_docs/ (7 документов), снимок индекса C, метки eval/data/golden_retrieval_labeled*.jsonl.
# Выход: eval/runs/chunking_ab_2026-10-17/{results.json,report.md}. Платные вызовы — только
# эмбеддинги text-embedding-3-small, только с --yes; результаты кешируются по хэшу текста.
# Метка = текст размеченного чанка старого индекса; чанк нарезки покрывает её, если содержит
# ≥60% её слов (≥4 букв). Бюджет контекста B — суммарная длина выбранных чанков в знаках.

    .venv/bin/python -m eval.chunking_ab.run --dry-run
    .venv/bin/python -m eval.chunking_ab.run --yes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.chunking_ab.chunkers import (  # noqa: E402
    chunk_paragraphs,
    chunk_structural_parts,
    chunk_window,
    paragraphs_from_text,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval/runs/chunking_ab_2026-10-17"
LABEL_DB = ROOT / "chroma_db/chroma.sqlite3"
CURRENT_DB = ROOT / "chroma_db.snapshots/build-8cs1plb3/chroma.sqlite3"
QUESTION_FILES = (
    "eval/data/golden_retrieval_labeled.jsonl",
    "eval/data/golden_retrieval_labeled_ext.jsonl",
)
# ключ — имя source в индексе; значение — (файл source_docs, wrapped)
DOCS = {
    "2464.md": ("2464.md", False),
    "тк рф.pdf": ("тк рф.pdf", True),
    "426-ФЗ.pdf": ("426-ФЗ.pdf", True),
    "О пожарной безопасности.pdf": ("О пожарной безопасности.pdf", True),
    "1479 ппр.pdf": ("1479 ппр.pdf", True),
    "782н.pdf": ("782н.pdf", True),
    "776н.pdf": ("776н.pdf", True),
}
# метки 2464 живут под старым именем в chroma_db
LABEL_SOURCE_ALIAS = {"2464.pdf": "2464.md"}
DOC_TITLE = {
    "2464.md": "Постановление Правительства РФ № 2464",
    "тк рф.pdf": "Трудовой кодекс РФ",
    "426-ФЗ.pdf": "ФЗ № 426-ФЗ о специальной оценке условий труда",
    "О пожарной безопасности.pdf": "ФЗ № 69-ФЗ О пожарной безопасности",
    "1479 ппр.pdf": "Правила противопожарного режима (ПП № 1479)",
    "782н.pdf": "Правила по охране труда при работе на высоте (782н)",
    "776н.pdf": "Положение о СУОТ (776н)",
}
VARIANTS = ["C", "W350", "P", "S900", "S1800", "S3000", "S1800-np"]
RETRIEVERS = ["dense", "bm25", "hybrid"]
BUDGETS = [4000, 8000, 16000]
HEADLINE = ("hybrid", 8000)
COVER_THRESHOLD = 0.6
RRF_K = 60
PRICE_PER_M_TOKENS = 0.02
WORD_RE = re.compile(r"[а-яёa-z0-9]{4,}")


# ── данные ────────────────────────────────────────────────────────────────


def read_chunks(db: Path, sources: set[str]) -> list[tuple[str, str, str, str]]:
    """(source, chunk_id, text, parent_section) из sqlite-индекса chroma, нужные source."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows: dict[int, dict] = defaultdict(dict)
    for i, k, v in con.execute(
        "select id,key,coalesce(string_value,cast(int_value as text)) from embedding_metadata"
        " where key in ('source','chunk_id','chroma:document','parent_section')"
    ):
        rows[i][k] = v
    out = [
        (m["source"], m["chunk_id"], m["chroma:document"], m.get("parent_section", ""))
        for m in rows.values()
        if m.get("source") in sources and "chunk_id" in m and "chroma:document" in m
    ]
    out.sort(key=lambda t: (t[0], int(t[1])))
    return out


def pdf_text(path: Path) -> str:
    import pymupdf

    return "\n".join(p.get_text() for p in pymupdf.open(path))


def load_paragraphs() -> dict[str, list[str]]:
    out = {}
    for name, (fname, wrapped) in DOCS.items():
        path = ROOT / "source_docs" / fname
        text = path.read_text(encoding="utf-8") if fname.endswith(".md") else pdf_text(path)
        out[name] = paragraphs_from_text(text, wrapped=wrapped)
    return out


def strip_heading(text: str, heading: str) -> str:
    """Тело чанка индекса без заголовка раздела, который прод дописывает для эмбеддинга."""
    if heading and text.startswith(heading):
        return text[len(heading) :].lstrip("\n ")
    return text


def build_variants(paras: dict[str, list[str]]) -> dict[str, list[tuple[str, str, str]]]:
    """variant -> [(doc, text для эмбеддинга, тело без префикса)]."""
    variants: dict[str, list[tuple[str, str, str]]] = {v: [] for v in VARIANTS}
    for src, _, text, heading in read_chunks(CURRENT_DB, set(DOCS)):
        variants["C"].append((src, text, strip_heading(text, heading)))
    for doc, ps in paras.items():
        title = DOC_TITLE[doc]
        full = "\n".join(ps)
        variants["W350"] += [(doc, c, c) for c in chunk_window(full)]
        variants["P"] += [(doc, c, c) for c in chunk_paragraphs(ps)]
        for name, limit in (("S900", 900), ("S1800", 1800), ("S3000", 3000)):
            for head, body in chunk_structural_parts(ps, title, limit):
                variants[name].append((doc, f"{head}\n{body}", body))
        for head, body in chunk_structural_parts(ps, title, 1800):
            variants["S1800-np"].append((doc, body, body))
    return variants


def words(text: str) -> frozenset[str]:
    return frozenset(WORD_RE.findall(text.lower()))


def load_questions() -> tuple[list[dict], dict[str, dict]]:
    """Уникальные вопросы с метками-доказательствами (в областях корпуса)."""
    label_text = {
        f"{s}#{c}": strip_heading(t, h)
        for s, c, t, h in read_chunks(LABEL_DB, set(DOCS) | set(LABEL_SOURCE_ALIAS))
    }
    seen, questions = set(), []
    for f in QUESTION_FILES:
        for line in (ROOT / f).read_text(encoding="utf-8").splitlines():
            q = json.loads(line)
            if q["question"] in seen:
                continue
            seen.add(q["question"])
            labels = []
            for rid in q["relevant_chunk_ids"]:
                src = rid.split("#")[0]
                doc = LABEL_SOURCE_ALIAS.get(src, src)
                w = words(label_text.get(rid, ""))
                if doc in DOCS and len(w) >= 8:
                    labels.append({"id": rid, "doc": doc, "words": w})
            if labels:
                questions.append(
                    {"question": q["question"], "labels": labels, "doc": labels[0]["doc"]}
                )
    return questions, label_text


# ── метрики нарезки ───────────────────────────────────────────────────────


def chunk_quality(chunks: list[tuple[str, str, str]]) -> dict:
    """Длины — по тексту для эмбеддинга; оборванность и сироты — по телу."""
    lens = [len(t) for _, t, _ in chunks]
    cut = sum(1 for _, _, b in chunks if b.rstrip()[-1:] not in ".;:!?»)\"…")
    sub = re.compile(r"^(?:[а-яё]|\d+)\)\s")
    # сирота: тело начинается с подпункта «а)», то есть вводная осталась в другом чанке
    orphan = sum(1 for _, _, b in chunks if sub.match(b.lstrip()))
    return {
        "chunks": len(chunks),
        "median_chars": int(statistics.median(lens)),
        "mean_chars": int(statistics.mean(lens)),
        "p90_chars": int(sorted(lens)[int(0.9 * (len(lens) - 1))]),
        "cut_mid_sentence_pct": round(100 * cut / len(chunks), 1),
        "orphan_subitem_pct": round(100 * orphan / len(chunks), 1),
    }


# ── поиск ─────────────────────────────────────────────────────────────────


def stem_tokens(text: str) -> list[str]:
    return [w[:5] for w in re.findall(r"[а-яёa-z0-9]+", text.lower()) if len(w) > 1]


def embed_all(texts: list[str], cache_path: Path, estimate_only: bool) -> tuple[dict, int, int]:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    cache: dict[str, np.ndarray] = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    key = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()  # noqa: E731
    todo = sorted({t for t in texts if key(t) not in cache}, key=key)
    tokens = sum(len(enc.encode(t, disallowed_special=())) for t in todo)
    if estimate_only or not todo:
        return cache, len(todo), tokens
    from src.infra.llm_factory import get_embedding_model

    model = get_embedding_model()
    for i in range(0, len(todo), 64):
        batch = todo[i : i + 64]
        vecs = model.embed_documents(batch)
        for t, v in zip(batch, vecs):
            cache[key(t)] = np.asarray(v, dtype=np.float32)
        cache_path.write_bytes(pickle.dumps(cache))
        print(f"  embedded {min(i + 64, len(todo))}/{len(todo)}", flush=True)
    return cache, len(todo), tokens


def rank_lists(texts: list[str], vecs: np.ndarray, qtexts: list[str], qvecs: np.ndarray) -> dict[str, list[list[int]]]:
    from rank_bm25 import BM25Okapi

    vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    qn = qvecs / np.linalg.norm(qvecs, axis=1, keepdims=True)
    dense = [list(np.argsort(-(vn @ q))[:100]) for q in qn]
    bm = BM25Okapi([stem_tokens(t) for t in texts])
    bm25 = [list(np.argsort(-bm.get_scores(stem_tokens(q)))[:100]) for q in qtexts]
    hybrid = []
    for d, b in zip(dense, bm25):
        score: dict[int, float] = defaultdict(float)
        for lst in (d, b):
            for r, i in enumerate(lst, 1):
                score[i] += 1 / (RRF_K + r)
        hybrid.append(sorted(score, key=lambda i: -score[i])[:100])
    return {"dense": dense, "bm25": bm25, "hybrid": hybrid}


def select_budget(order: list[int], lengths: list[int], budget: int) -> list[int]:
    chosen, total = [], 0
    for i in order:
        if chosen and total + lengths[i] > budget:
            break
        chosen.append(i)
        total += lengths[i]
    return chosen


def label_cover_sets(cw: list[frozenset[str]], label: frozenset[str]) -> set[frozenset[int]]:
    """Наборы чанков, покрывающих метку: один чанк ≥60% слов либо два соседних в тексте."""
    frac = {}
    for i, w in enumerate(cw):
        f = len(w & label) / len(label)
        if f >= 0.2:
            frac[i] = f
    sets: set[frozenset[int]] = set()
    for i, f in frac.items():
        if f >= COVER_THRESHOLD:
            sets.add(frozenset((i,)))
        if i + 1 in frac and len((cw[i] | cw[i + 1]) & label) / len(label) >= COVER_THRESHOLD:
            sets.add(frozenset((i, i + 1)))
    return sets


def evaluate(variant_chunks, questions, ranks):
    """per[variant][retriever][budget|topK] = [(hit, label_recall, n_chunks)] по вопросам."""
    out = {}
    for v, chunks in variant_chunks.items():
        cw = [words(b) for _, _, b in chunks]
        lens = [len(t) for _, t, _ in chunks]
        cover = {}
        oracle = []
        for qi, q in enumerate(questions):
            labs = []
            for li, lab in enumerate(q["labels"]):
                cs = label_cover_sets(cw, lab["words"])
                cover[(qi, li)] = cs
                labs.append(bool(cs))
            oracle.append(labs)

        def got_for(qi, chosen):
            return [
                any(cs <= chosen for cs in cover[(qi, li)])
                for li in range(len(questions[qi]["labels"]))
            ]

        per = {}
        for r in RETRIEVERS:
            per[r] = {}
            for B in BUDGETS:
                rows = []
                for qi in range(len(questions)):
                    chosen = set(select_budget(ranks[v][r][qi], lens, B))
                    got = got_for(qi, chosen)
                    rows.append((any(got), sum(got) / len(got), len(chosen)))
                per[r][B] = rows
            for k in (5, 12):
                rows = []
                for qi in range(len(questions)):
                    got = got_for(qi, set(ranks[v][r][qi][:k]))
                    rows.append((any(got), sum(got) / len(got), k))
                per[r][f"top{k}"] = rows
        out[v] = {"per": per, "oracle": oracle}
    return out


def bootstrap_diff(a: list[bool], b: list[bool], n: int = 10000, seed: int = 7):
    rng = random.Random(seed)
    m = len(a)
    diffs = []
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(sum(a[i] for i in idx) / m - sum(b[i] for i in idx) / m)
    diffs.sort()
    return (sum(a) - sum(b)) / m, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


# ── отчёт ─────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="разрешить платные эмбеддинги")
    args = ap.parse_args()
    if not (args.dry_run or args.yes):
        ap.error("нужен --dry-run или --yes")
    OUT.mkdir(parents=True, exist_ok=True)

    paras = load_paragraphs()
    variants = build_variants(paras)
    questions, _ = load_questions()
    print(f"вопросов с метками в корпусе: {len(questions)}")
    quality = {v: chunk_quality(c) for v, c in variants.items()}
    for v, q in quality.items():
        print(v, q)

    # покрытие текста структурными вариантами: каждый абзац лежит в каком-то чанке
    cover_gaps = {}
    for name in ("S900", "S1800", "S3000"):
        blob_by_doc: dict[str, str] = defaultdict(str)
        for d, b, _ in variants[name]:
            blob_by_doc[d] += "\n" + b
        norm = lambda x: " ".join(x.split())  # noqa: E731
        blobs = {d: norm(b) for d, b in blob_by_doc.items()}
        missing = sum(1 for d, ps in paras.items() for p in ps if norm(p)[:60] not in blobs[d])
        cover_gaps[name] = missing
    print("абзацев без следа в чанках:", cover_gaps)

    qtexts = [q["question"] for q in questions]
    all_texts = [t for c in variants.values() for _, t, _ in c] + qtexts
    cache, n_new, tokens = embed_all(all_texts, OUT / "embed_cache.pkl", estimate_only=args.dry_run)
    print(f"к эмбеддингу новых текстов: {n_new}, токенов ≈ {tokens:,}, цена ≈ ${tokens / 1e6 * PRICE_PER_M_TOKENS:.3f}")
    if args.dry_run:
        # проверка оракула без эмбеддингов
        ranks = {v: {r: [list(range(len(c)))[:100] for _ in questions] for r in RETRIEVERS} for v, c in variants.items()}
        ev = evaluate(variants, questions, ranks)
        for v in VARIANTS:
            orc = ev[v]["oracle"]
            lab_cov = sum(x for q in orc for x in q) / sum(len(q) for q in orc)
            q_cov = sum(any(q) for q in orc) / len(orc)
            print(f"оракул {v}: меток покрыто {lab_cov:.1%}, вопросов с покрытой меткой {q_cov:.1%}")
        return 0

    key = lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()  # noqa: E731
    qvecs = np.stack([cache[key(t)] for t in qtexts])
    ranks = {}
    for v, c in variants.items():
        texts = [t for _, t, _ in c]
        vecs = np.stack([cache[key(t)] for t in texts])
        ranks[v] = rank_lists(texts, vecs, qtexts, qvecs)
    ev = evaluate(variants, questions, ranks)

    keep = [qi for qi in range(len(questions)) if all(any(ev[v]["oracle"][qi]) for v in VARIANTS)]
    print(f"вопросов, покрытых во всех вариантах: {len(keep)}/{len(questions)}")

    results = {"n_questions": len(questions), "n_kept": len(keep), "quality": quality,
               "cover_gaps": cover_gaps, "oracle": {}, "metrics": {}, "diff_vs_C": {}, "by_doc": {}}
    for v in VARIANTS:
        orc = ev[v]["oracle"]
        results["oracle"][v] = {
            "labels_covered": round(sum(x for q in orc for x in q) / sum(len(q) for q in orc), 3),
            "questions_covered": round(sum(any(q) for q in orc) / len(orc), 3),
        }
        results["metrics"][v] = {}
        for r in RETRIEVERS:
            results["metrics"][v][r] = {}
            for B in BUDGETS + ["top5", "top12"]:
                rows = [ev[v]["per"][r][B][qi] for qi in keep]
                results["metrics"][v][r][str(B)] = {
                    "hit": round(sum(x[0] for x in rows) / len(rows), 3),
                    "label_recall": round(sum(x[1] for x in rows) / len(rows), 3),
                    "mean_chunks": round(sum(x[2] for x in rows) / len(rows), 1),
                }
    for v in VARIANTS[1:]:
        results["diff_vs_C"][v] = {}
        for r in RETRIEVERS:
            for B in BUDGETS:
                a = [ev[v]["per"][r][B][qi][0] for qi in keep]
                b = [ev["C"]["per"][r][B][qi][0] for qi in keep]
                d, lo, hi = bootstrap_diff(a, b)
                results["diff_vs_C"][v][f"{r}/{B}"] = [round(d, 3), round(lo, 3), round(hi, 3)]
    r, B = HEADLINE
    docs_kept = defaultdict(list)
    for qi in keep:
        docs_kept[questions[qi]["doc"]].append(qi)
    for v in VARIANTS:
        results["by_doc"][v] = {
            d: {"n": len(qis), "hit": round(sum(ev[v]["per"][r][B][qi][0] for qi in qis) / len(qis), 3)}
            for d, qis in docs_kept.items()
        }
    (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(render_report(results), encoding="utf-8")
    print(f"готово: {OUT}/report.md")
    return 0


def render_report(res: dict) -> str:
    L = [f"# Отчёт A/B нарезок (вопросов в сравнении: {res['n_kept']} из {res['n_questions']})", ""]
    L += ["## Качество нарезки", "", "| вариант | чанков | медиана | p90 | оборвано посреди предложения, % | подпункт без вводной, % |", "|---|---|---|---|---|---|"]
    for v, q in res["quality"].items():
        L.append(f"| {v} | {q['chunks']} | {q['median_chars']} | {q['p90_chars']} | {q['cut_mid_sentence_pct']} | {q['orphan_subitem_pct']} |")
    L += ["", f"Абзацев без следа в чанках: {res['cover_gaps']}", "", "## Оракул (доля меток, покрытых хоть одним чанком)", ""]
    L += ["| вариант | меток | вопросов |", "|---|---|---|"]
    for v, o in res["oracle"].items():
        L.append(f"| {v} | {o['labels_covered']:.1%} | {o['questions_covered']:.1%} |")
    for r in RETRIEVERS:
        L += ["", f"## Доля вопросов с покрытой нормой при бюджете контекста — {r}", ""]
        L += ["| вариант | B=4000 | B=8000 | B=16000 | top5 | top12 | чанков при 8000 |", "|---|---|---|---|---|---|---|"]
        for v in VARIANTS:
            m = res["metrics"][v][r]
            L.append(f"| {v} | {m['4000']['hit']:.1%} | {m['8000']['hit']:.1%} | {m['16000']['hit']:.1%} | {m['top5']['hit']:.1%} | {m['top12']['hit']:.1%} | {m['8000']['mean_chunks']} |")
    L += ["", "## Разница с C (п.п., 95% бутстреп-интервал), hybrid", "", "| вариант | B=4000 | B=8000 | B=16000 |", "|---|---|---|---|"]
    for v, d in res["diff_vs_C"].items():
        cells = [f"{d[f'hybrid/{B}'][0]*100:+.1f} [{d[f'hybrid/{B}'][1]*100:+.1f}; {d[f'hybrid/{B}'][2]*100:+.1f}]" for B in BUDGETS]
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    L += ["", "## По документам (hybrid, B=8000)", ""]
    docs = sorted({d for v in res["by_doc"].values() for d in v})
    L += ["| документ | n | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---|" * len(VARIANTS)]
    for d in docs:
        n = res["by_doc"]["C"][d]["n"]
        L.append(f"| {d} | {n} | " + " | ".join(f"{res['by_doc'][v][d]['hit']:.0%}" for v in VARIANTS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
