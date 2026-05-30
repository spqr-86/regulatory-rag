"""Trace all problematic questions from baseline eval through the pipeline.

For each question, captures:
- final_passages with sources and scores
- retrieval_attempts (stages taken)
- generated answer
- side-by-side with ground_truth

Output: JSON file with full trace data + markdown summary.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("LANGSMITH_TRACING_V2", "false")
os.environ.setdefault("RERANKER_BACKEND", "crossencoder")

from dotenv import load_dotenv

load_dotenv()

from src.infra.llm_factory import apply_ipv6_patch_for_googleapis  # noqa: E402

apply_ipv6_patch_for_googleapis()

from src.backends.vector_store import get_vector_store_backend  # noqa: E402
from src.v7.bridge import init_v7_pipeline  # noqa: E402
from src.v7.graph import build_graph  # noqa: E402

# In-scope questions with correctness < 7 from eval_v7_2026-05-29.jsonl
TARGET_QUESTIONS = [
    "Как проводится расследование несчастного случая на производстве?",
    "В течение какого срока работодатель обязан сообщить в ФСС о несчастном случае на производстве?",
    "Нужна ли каска бухгалтеру?",
    "Что делать при изменении технологии производства?",
    "Какие документы должны быть разработаны при выявлении вредных факторов?",
    "Работодатель вправе допустить работника к работе до прохождения первичного инструктажа, если тот имеет большой опыт — верно?",
    "Какова процедура проведения специальной оценки условий труда (СОУТ) по новой методике?",
    "Какие размеры штрафов за нарушения охраны труда в 2025-2026?",
    "Что такое декларация пожарной безопасности и кто её разрабатывает?",
    "Какие критерии используются при отнесении рабочих мест к классам условий труда и какие гарантии положены работникам?",
    "Организация внедрила новую технологию производства. Какова процедура оценки рисков и обучения?",
    "Что включает в себя медицинский осмотр работника по новым правилам?",
    "Какие требования предъявляются к персоналу, ответственному за пожарную безопасность?",
    "Какие основные функции выполняет специалист по охране труда?",
    "Какие документы необходимо иметь при проверке пожарной безопасности?",
    "Как организовать систему управления охраной труда (СУОТ) согласно Приказу Минтруда?",
    "Работник отказывается выполнять работу, считая её опасной. Каковы его права и обязанности работодателя?",
    "Какой штраф за нарушение требований охраны труда?",
    "Кто входит в комиссию по расследованию несчастного случая на производстве?",
    "СОУТ можно не проводить если рабочие места признаны безопасными по результатам аттестации — верно?",
    "Кто проходит обучение по программе А обучения требований охраны труда?",
]


def load_baseline() -> dict[str, dict]:
    """Map question → record from baseline eval."""
    with open("benchmarks/eval_v7_2026-05-29.jsonl") as f:
        data = json.load(f)
    return {r["question"]: r for r in data["results"]}


def trace_one(graph, question: str) -> dict:
    t0 = time.time()
    state = graph.invoke({"query": question})
    elapsed = round(time.time() - t0, 2)

    final = state.get("final_passages") or []
    attempts = state.get("retrieval_attempts") or []
    stages = [a.get("stage", "?") for a in attempts]

    passages_summary = []
    for p in final[:12]:
        meta = p.get("metadata", {})
        passages_summary.append(
            {
                "src": meta.get("source", "?"),
                "section": (
                    meta.get("parent_section") or meta.get("heading_path") or "?"
                )[:200],
                "score": round(float(p.get("score", 0.0)), 4),
                "vector_score": round(float(p.get("vector_score", 0.0)), 4),
                "text": (p.get("text", "") or "")[:400].replace("\n", " "),
            }
        )

    return {
        "answer": state.get("answer", ""),
        "stages": stages,
        "path": "complex" if "complex" in stages else "simple",
        "n_passages": len(final),
        "elapsed_sec": elapsed,
        "passages": passages_summary,
    }


def main() -> None:
    print("Loading baseline...")
    baseline = load_baseline()
    print(f"  {len(baseline)} baseline records")

    print("Initializing pipeline...")
    vs = get_vector_store_backend(load_existing=True)
    init_v7_pipeline(vs)
    graph = build_graph().compile()

    out_path = Path("benchmarks/failure_traces_2026-05-29.json")
    traces = []

    for i, q in enumerate(TARGET_QUESTIONS, 1):
        print(f"\n[{i}/{len(TARGET_QUESTIONS)}] {q[:80]}")
        b = baseline.get(q, {})
        try:
            trace = trace_one(graph, q)
        except Exception as exc:
            print(f"   ERROR: {exc}")
            traces.append(
                {
                    "question": q,
                    "error": str(exc),
                }
            )
            continue

        traces.append(
            {
                "question": q,
                "ground_truth": b.get("ground_truth", ""),
                "baseline_answer": b.get("answer", ""),
                "baseline_score": b.get("correctness_score", -1),
                "baseline_reasoning": b.get("correctness_reasoning", ""),
                "trace": trace,
            }
        )
        print(
            f"   path={trace['path']} stages={'/'.join(trace['stages'])} elapsed={trace['elapsed_sec']:.1f}s"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traces, ensure_ascii=False, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
