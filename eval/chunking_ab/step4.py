"""Шаг 4: реальный контекст генератора (граф V7, только поиск) против замены на родителей.

# ANCHOR: вход step4_real_graph.jsonl (маршрут и final_context реального графа),
# norm_labels.jsonl, step1_norms.jsonl, step2_parents.jsonl. Схемы: A — passages как есть;
# B2 — каждый passage заменён родителем ≤3000 знаков (дедуп по родителю, порядок сохранён);
# B2eq — B2, обрезанный до длины контекста A того же вопроса (равный объём).
# Группы: complex (путь посетил complex) и simple. Платных вызовов нет.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.chunking_ab.norms import Norm  # noqa: E402
from eval.chunking_ab.run import words  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "eval/runs/chunking_ab_2026-10-17"
CAP = 3000


def jl(name: str) -> list[dict]:
    return [json.loads(x) for x in (OUT / name).read_text(encoding="utf-8").splitlines() if x]


def window(norm: Norm, body: str, cap: int = CAP) -> str:
    if len(norm.text) <= cap:
        return norm.text
    target = words(body)
    scores = [len(words(p) & target) for p in norm.paragraphs]
    first = max(range(len(scores)), key=scores.__getitem__)
    lo, hi = first, first + 1
    size = len(norm.paragraphs[first])
    while True:
        grew = False
        if lo > 0 and size + len(norm.paragraphs[lo - 1]) + 1 <= cap:
            lo -= 1
            size += len(norm.paragraphs[lo]) + 1
            grew = True
        if hi < len(norm.paragraphs) and size + len(norm.paragraphs[hi]) + 1 <= cap:
            size += len(norm.paragraphs[hi]) + 1
            hi += 1
            grew = True
        if not grew:
            return "\n".join(norm.paragraphs[lo:hi])


def build_b2(passages: list[dict], parent_of: dict, norms: dict[str, Norm]) -> list[tuple[str, str, str | None]]:
    out, seen = [], set()
    for p in passages:
        pid = parent_of.get((p["source"], str(p["chunk_id"])))
        if pid and pid in norms:
            if pid in seen:
                continue
            seen.add(pid)
            out.append((p["source"], window(norms[pid], p["text"]), pid))
        else:
            out.append((p["source"], p["text"], None))
    return out


def truncate_to(ctx: list[tuple[str, str, str | None]], limit: int):
    out, total = [], 0
    for src, text, pid in ctx:
        if total + len(text) > limit and out:
            break
        out.append((src, text, pid))
        total += len(text) + 1
    return out


def evaluate(ctx, norm: Norm, parents_known: bool) -> tuple[bool, bool]:
    local = [c for c in ctx if c[0] == norm.doc]
    covered: set[str] = set()
    found = False
    for _, text, pid in local:
        w = words(text)
        covered |= w
        found |= pid == norm.id or (
            pid is None and bool(norm.words) and len(w & norm.words) / len(norm.words) >= 0.6
        )
    complete = bool(norm.words) and len(covered & norm.words) / len(norm.words) >= 0.9
    return found, complete


def main() -> int:
    graph = {r["question"]: r for r in jl("step4_real_graph.jsonl")}
    norm_rows = {r["id"]: r for r in jl("step1_norms.jsonl")}
    norms = {
        i: Norm(r["doc"], r["kind"], r["number"], tuple(r["paragraphs"]), int(i.rsplit(":", 1)[1]))
        for i, r in norm_rows.items()
    }
    parent_of = {(r["doc"], str(r["chunk_id"])): r["norm_id"] for r in jl("step2_parents.jsonl") if r.get("norm_id")}
    labels = jl("norm_labels.jsonl")

    agg: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0, 0]))  # group→scheme→[q,qf,qc,nf,nc,nnorms]
    res = {}
    sizes = defaultdict(lambda: defaultdict(list))
    detail = []
    for lab in labels:
        g = graph.get(lab["question"])
        if not g:
            continue
        group = "complex" if g["complex_visited"] else "simple"
        qnorms = [norms[n["id"]] for n in lab["norms"] if n["id"] in norms]
        if g["outcome"] != "ready":
            group = f"{group}-{g['outcome']}"
        passages = g["final_context"]
        a_ctx = [(p["source"], p["text"], parent_of.get((p["source"], str(p["chunk_id"])))) for p in passages]
        b2 = build_b2(passages, parent_of, norms)
        a_len = sum(len(c[1]) for c in a_ctx)
        schemes = {"A": a_ctx, "B2": b2, "B2eq": truncate_to(b2, a_len)}
        for s, ctx in schemes.items():
            sizes[group][s].append(sum(len(c[1]) for c in ctx))
            fs = [evaluate(ctx, n, True) for n in qnorms]
            row = agg[group][s]
            row[0] += 1
            row[1] += any(f for f, _ in fs)
            row[2] += any(c for _, c in fs)
            row[3] += sum(f for f, _ in fs)
            row[4] += sum(c for _, c in fs)
            agg[group][s + "_n"] = agg[group].get(s + "_n", 0)
            res.setdefault(group, {}).setdefault("norms", 0)
        res.setdefault(group, {})["norms"] = res[group].get("norms", 0) + len(qnorms)
        res[group]["questions"] = res[group].get("questions", 0) + 1
        detail.append({"question": lab["question"], "group": group, "norms": [n.id for n in qnorms],
                       **{s: [evaluate(ctx, n, True) for n in qnorms] for s, ctx in schemes.items()}})
    lines = ["# Шаг 4 — реальный контекст генератора (граф V7, только поиск), A против B2", ""]
    lines.append(f"Вопросов с результатом графа: {len(graph)}; индекс: см. step4_real_graph_meta.json.")
    for group in sorted(res):
        lines += ["", f"## Группа {group}: вопросов {res[group]['questions']}, норм {res[group]['norms']}", "",
                  "| схема | найдена (вопросы) | целиком (вопросы) | найдена (нормы) | целиком (нормы) | средний контекст, знаков |", "|---|---|---|---|---|---|"]
        for s in ("A", "B2", "B2eq"):
            q, qf, qc, nf, nc = agg[group][s]
            mean = sum(sizes[group][s]) / max(len(sizes[group][s]), 1)
            lines.append(f"| {s} | {qf}/{q} | {qc}/{q} | {nf}/{res[group]['norms']} | {nc}/{res[group]['norms']} | {mean:.0f} |")
    (OUT / "step4_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "step4_details.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in detail) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
