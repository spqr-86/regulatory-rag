"""Produce a detailed Markdown analysis from failure_traces JSON."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    src = Path("benchmarks/failure_traces_2026-05-29.json")
    out = Path("docs/plans/2026-05-29-eval-failures-detailed.md")
    traces = json.loads(src.read_text())

    lines: list[str] = []
    lines.append("# Детальный разбор проблемных вопросов eval — 2026-05-29")
    lines.append("")
    lines.append(
        "Для каждого вопроса: GT, baseline ответ и оценка судьи, top-12 retrieved chunks с источниками и скорами, диагностика."
    )
    lines.append("")
    lines.append(f"Всего: {len(traces)} вопросов")
    lines.append("")
    lines.append("---")
    lines.append("")

    for i, t in enumerate(traces, 1):
        q = t["question"]
        gt = t.get("ground_truth", "")
        bs = t.get("baseline_score", -1)
        breason = t.get("baseline_reasoning", "")
        banswer = t.get("baseline_answer", "")
        trace = t.get("trace", {})
        passages = trace.get("passages", [])
        new_answer = trace.get("answer", "")
        path = trace.get("path", "?")
        stages = "/".join(trace.get("stages", []))

        lines.append(f"## {i}. {q}")
        lines.append("")
        lines.append(
            f"**Baseline score:** {bs}/10  |  **Path:** {path}  |  **Stages:** {stages}"
        )
        lines.append("")
        lines.append("**Ground truth:**")
        lines.append(f"> {gt[:600]}")
        lines.append("")
        lines.append("**Baseline answer (FlashRank → CrossEncoder eval):**")
        lines.append(f"> {banswer[:600]}")
        lines.append("")
        if breason:
            lines.append("**Judge reasoning:**")
            lines.append(f"> {breason[:400]}")
            lines.append("")

        if new_answer != banswer and new_answer:
            lines.append("**New trace answer (re-run):**")
            lines.append(f"> {new_answer[:600]}")
            lines.append("")

        lines.append(f"**Retrieved passages (top {min(len(passages), 12)}):**")
        lines.append("")
        lines.append("| # | Score | Vec | Source | Section | Snippet |")
        lines.append("|---|-------|-----|--------|---------|---------|")
        for j, p in enumerate(passages[:12], 1):
            src_short = (p.get("src") or "?").replace(".pdf", "")[:18]
            section = (
                (p.get("section") or "?")[:60].replace("|", "\\|").replace("\n", " ")
            )
            snippet = (p.get("text") or "")[:120].replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| {j} | {p.get('score', 0):.3f} | {p.get('vector_score', 0):.3f} | {src_short} | {section} | {snippet} |"
            )
        lines.append("")
        lines.append("**Диагностика:** _(заполняется вручную)_")
        lines.append("")
        lines.append("---")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote → {out}")
    print(f"Length: {len(lines)} lines")


if __name__ == "__main__":
    main()
