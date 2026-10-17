"""Реальный граф V7 в режиме «только поиск» на вопросах эксперимента (шаг 4).

# ANCHOR: вход — вопросы из norm_labels.jsonl; выход — step4_real_graph.jsonl: маршрут,
# итог, passages финального контекста (источник, chunk_id, текст). Платные вызовы: только
# эмбеддинги вопросов (расширение запроса LLM выключено: V8_ENABLE_MULTI_QUERY=False).
# Генерации нет. Индекс — тот, что в .env (CHROMA_DB_PATH), снимок пишется в вывод.

    .venv/bin/python -m eval.chunking_ab.real_graph --limit 3
    .venv/bin/python -m eval.chunking_ab.real_graph
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

OUT = Path(__file__).resolve().parents[2] / "eval/runs/chunking_ab_2026-10-17"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from eval.run_retrieval_eval import init_engine
    from src.v7.retrieval import retrieve_context

    questions = []
    seen = set()
    for line in (OUT / "norm_labels.jsonl").read_text(encoding="utf-8").splitlines():
        q = json.loads(line)["question"]
        if q not in seen:
            seen.add(q)
            questions.append(q)
    if args.limit:
        questions = questions[: args.limit]

    runtime = init_engine()
    rows = []
    for i, q in enumerate(questions, 1):
        res = retrieve_context(q, runtime=runtime)
        stages = [a.get("stage") for a in res.attempts]
        rows.append(
            {
                "question": q,
                "outcome": res.outcome,
                "route": res.route,
                "reason": res.reason,
                "stages": stages,
                "complex_visited": "complex" in stages,
                "elapsed_ms": round(res.elapsed_ms),
                "final_context": [
                    {
                        "source": (p.get("metadata") or {}).get("source"),
                        "chunk_id": p.get("chunk_id"),
                        "text": p.get("page_content") or p.get("text") or "",
                    }
                    for p in res.final_context
                ],
                "attempt_passages": [
                    [
                        {
                            "source": (p.get("metadata") or {}).get("source"),
                            "chunk_id": p.get("chunk_id"),
                        }
                        for p in (a.get("passages") or [])
                    ]
                    for a in res.attempts
                ],
            }
        )
        print(i, len(questions), res.outcome, stages, flush=True)
    meta = {"chroma": os.environ.get("CHROMA_DB_PATH"), "n": len(rows)}
    path = OUT / (
        "step4_real_graph_smoke.jsonl" if args.limit else "step4_real_graph.jsonl"
    )
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (OUT / "step4_real_graph_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
