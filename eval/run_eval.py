"""V8 Eval runner — deterministic metrics, no LLM judge.

Парсит tests/dataset.csv, прогоняет V7 pipeline, считает:
  - completeness      (ключевые леммы ground_truth найдены в ответе)
  - abstain_rate      (доля отказов)
  - false_abstain     (domain-запрос → отказ = ошибка)
  - correct_abstain   (OOS-запрос → отказ = правильно)
  - avg_top_score     (информативно, baseline)
  - avg_passage_count (информативно, baseline)

Usage:
    cd /home/petr/projects/safety-incident-analyzer
    source venv/bin/activate
    python eval/run_eval.py
    python eval/run_eval.py --limit 5             # быстрый smoke test
    python eval/run_eval.py --output eval/baselines/v7_baseline.json
    python eval/run_eval.py --no-pipeline         # только dataset parse + print
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Lazy SQLite3 fix (same as app.py) ─────────────────────────────────────────
try:
    import pysqlite3

    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from eval.metrics import (
    compute_abstain_rate,
    compute_citation_doc_match,
    compute_citation_in_retrieval,
    compute_citation_rate,
    compute_completeness,
    compute_correct_abstain_rate,
    compute_false_abstain_rate,
    compute_inversion_detected,
    compute_inversion_rate,
    compute_retrieval_stats,
)

DATASET_PATH = Path(__file__).parent.parent / "tests" / "dataset.csv"
BASELINES_DIR = Path(__file__).parent / "baselines"


def _is_oos(oos_type: str) -> bool:
    """True если вопрос out_of_scope (система должна отказать).

    false_premise — отдельная категория: система должна ответить с коррекцией предпосылки,
    поэтому НЕ считается OOS для метрики correct_abstain_rate.
    """
    return oos_type == "out_of_scope"


# ── Dataset ───────────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            q = row.get("question", "").strip()
            gt = row.get("ground_truth", "").strip()
            if not q:
                print(f"  [SKIP] Row {i}: empty question")
                continue
            oos_type = row.get("oos_type", "").strip()
            rows.append(
                {
                    "row_num": i,
                    "question": q,
                    "ground_truth": gt,
                    "oos_type": oos_type,
                    "is_oos": _is_oos(oos_type),
                    "must_not_contain": row.get("must_not_contain", "").strip(),
                }
            )
    return rows


# ── Pipeline runner ────────────────────────────────────────────────────────────


def run_query(graph: Any, question: str) -> dict:
    """Прогоняет один запрос через V7 граф, возвращает структурированный результат."""
    start = time.time()
    state = graph.invoke({"query": question})
    elapsed = round(time.time() - start, 2)

    answer = state.get("answer", "") or ""
    final_passages = state.get("final_passages") or []
    retrieval_attempts = state.get("retrieval_attempts") or []

    stages = [a.get("stage", "") for a in retrieval_attempts]
    path = "complex" if "complex" in stages else "simple"

    # top_score из последнего retrieval attempt
    top_score = 0.0
    if retrieval_attempts:
        top_score = retrieval_attempts[-1].get("top_score", 0.0) or 0.0

    return {
        "answer": answer,
        "path": path,
        "elapsed_sec": elapsed,
        "top_score": top_score,
        "passage_count": len(final_passages),
        "final_passages": final_passages,
    }


def init_pipeline():
    """Инициализирует V7 граф (requires ChromaDB)."""
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import init_v7_from_chroma
    from src.v7.graph import build_graph

    vector_store = get_vector_store_backend(load_existing=True)
    init_v7_from_chroma(vector_store)
    return build_graph().compile()


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    limit: int | None = None,
    output: Path | None = None,
    no_pipeline: bool = False,
    label: str = "v7",
    delay: float = 3.0,
) -> dict:
    print(f"Loading dataset: {DATASET_PATH}")
    dataset = load_dataset(DATASET_PATH)
    if limit:
        dataset = dataset[:limit]
    oos_count = sum(1 for r in dataset if r["is_oos"])
    fp_count = sum(1 for r in dataset if r.get("oos_type") == "false_premise")
    print(
        f"  {len(dataset)} questions ({oos_count} out_of_scope, {fp_count} false_premise)\n"
    )

    if no_pipeline:
        print("--no-pipeline: skipping graph execution")
        return {}

    print("Initializing V7 pipeline...")
    graph = init_pipeline()
    print("  Ready.\n")

    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        is_oos = item["is_oos"]
        print(f"[{i:2d}/{len(dataset)}] {'[OOS] ' if is_oos else ''}{question[:65]}...")

        try:
            run_result = run_query(graph, question)
        except Exception as e:
            print(f"       ERROR: {e}")
            results.append(
                {
                    **item,
                    "answer": "",
                    "error": str(e),
                    "path": "error",
                    "top_score": 0.0,
                    "passage_count": 0,
                    "elapsed_sec": 0.0,
                    "completeness": 0.0,
                }
            )
            continue

        answer = run_result["answer"]
        passages = run_result["final_passages"]
        completeness = compute_completeness(ground_truth, answer)
        must_not_contain = item.get("must_not_contain", "")
        inversion = compute_inversion_detected(must_not_contain, answer)
        cit_rate = compute_citation_rate(answer)
        cit_in_ret = compute_citation_in_retrieval(answer, passages)
        cit_doc_match = compute_citation_doc_match(answer, passages)

        record = {
            **item,
            "answer": answer,
            "path": run_result["path"],
            "elapsed_sec": run_result["elapsed_sec"],
            "top_score": run_result["top_score"],
            "passage_count": run_result["passage_count"],
            "completeness": round(completeness, 4),
            "abstained": not bool(answer.strip()),
            "inversion_detected": inversion,
            "citation_rate": round(cit_rate, 4),
            "citation_in_retrieval": round(cit_in_ret, 4),
            "citation_doc_match": round(cit_doc_match, 4),
        }
        results.append(record)

        status = "ABSTAIN" if record["abstained"] else f"comp={completeness:.2f}"
        print(
            f"       {status} | path={run_result['path']} | {run_result['elapsed_sec']}s"
        )

        if delay > 0 and i < len(dataset):
            time.sleep(delay)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    answered = [r for r in results if not r.get("abstained") and "error" not in r]
    domain_results = [r for r in results if not r.get("is_oos")]
    oos_results = [r for r in results if r.get("is_oos")]

    avg_completeness = (
        sum(r["completeness"] for r in answered) / len(answered) if answered else 0.0
    )
    # Completeness на enumeration queries (содержат "кто", "какие", "в каких случаях")
    enum_results = [r for r in answered if _is_enumeration(r["question"])]
    enum_completeness = (
        sum(r["completeness"] for r in enum_results) / len(enum_results)
        if enum_results
        else 0.0
    )

    retrieval_stats = compute_retrieval_stats(results)
    latency_stats = _compute_latency_stats(results)

    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "dataset": str(DATASET_PATH),
        "dataset_size": len(dataset),
        "pipeline_runs": len(results),
        "metrics": {
            "completeness_mean": round(avg_completeness, 4),
            "completeness_enumeration": round(enum_completeness, 4),
            "abstain_rate": round(compute_abstain_rate(results), 4),
            "false_abstain_rate": round(compute_false_abstain_rate(results), 4),
            "correct_abstain_rate": round(compute_correct_abstain_rate(results), 4),
            "inversion_rate": round(compute_inversion_rate(results), 4),
            "citation_rate_mean": round(
                (
                    sum(r.get("citation_rate", 0.0) for r in answered) / len(answered)
                    if answered
                    else 0.0
                ),
                4,
            ),
            "citation_in_retrieval_mean": round(
                (
                    sum(r.get("citation_in_retrieval", 0.0) for r in answered)
                    / len(answered)
                    if answered
                    else 0.0
                ),
                4,
            ),
            "citation_doc_match_mean": round(
                (
                    sum(r.get("citation_doc_match", 0.0) for r in answered)
                    / len(answered)
                    if answered
                    else 0.0
                ),
                4,
            ),
            **retrieval_stats,
            **latency_stats,
        },
        "results": results,
    }

    # ── Print table ────────────────────────────────────────────────────────────
    m = summary["metrics"]
    print(f"\n{'=' * 55}")
    print(
        f"Eval: {label} | {len(results)} queries | {datetime.now().strftime('%H:%M')}"
    )
    print(f"{'─' * 55}")
    print(
        f"  Completeness (all):        {m['completeness_mean']:.3f}  (target >= 0.70)"
    )
    print(
        f"  Completeness (enumeration):{m['completeness_enumeration']:.3f}  (target >= 0.70)"
    )
    print(f"  Abstain rate:              {m['abstain_rate']:.3f}")
    print(f"  False abstain:             {m['false_abstain_rate']:.3f}  (target = 0)")
    print(
        f"  Correct abstain (OOS):     {m['correct_abstain_rate']:.3f}  (target = 1.0)"
    )
    print(
        f"  Inversion rate:            {m['inversion_rate']:.3f}  (target = 0, checked {sum(1 for r in results if r.get('must_not_contain', '').strip())} rows)"
    )
    print(
        f"  Citation rate:             {m['citation_rate_mean']:.3f}  (% sentences with citation)"
    )
    print(
        f"  Citation in retrieval:     {m['citation_in_retrieval_mean']:.3f}  (target = 1.0)"
    )
    print(
        f"  Citation doc match:        {m['citation_doc_match_mean']:.3f}  (informative)"
    )
    print(f"  Latency p50:               {m['latency_p50']:.1f}s")
    print(
        f"  Latency p95:               {m['latency_p95']:.1f}s  (informative, n={len(results)})"
    )
    print(f"  Latency max:               {m['latency_max']:.1f}s")
    print(f"  Latency >10s:              {m['latency_count_over_10s']} queries")
    print(f"  Avg top_score:             {m['avg_top_score']:.4f}")
    print(f"  Avg passage_count:         {m['avg_passage_count']:.1f}")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {output}")

    return summary


def _compute_latency_stats(results: list[dict]) -> dict:
    """Latency статистика по всем запросам."""
    latencies = sorted(r.get("elapsed_sec", 0.0) for r in results if "error" not in r)
    if not latencies:
        return {
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_max": 0.0,
            "latency_count_over_10s": 0,
        }
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[min(int(n * 0.95), n - 1)]
    return {
        "latency_p50": round(p50, 2),
        "latency_p95": round(p95, 2),
        "latency_max": round(latencies[-1], 2),
        "latency_count_over_10s": sum(1 for t in latencies if t > 10.0),
    }


def _is_enumeration(question: str) -> bool:
    import re

    patterns = [
        r"\bкто\b",
        r"\bкакие\b",
        r"\bв\s+каких\b",
        r"\bперечисли",
        r"\bкаким\b",
    ]
    q = question.lower()
    return any(re.search(p, q) for p in patterns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V7/V8 RAG pipeline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--label", type=str, default="v7")
    parser.add_argument(
        "--no-pipeline",
        action="store_true",
        help="Only parse dataset, skip graph execution",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="Seconds to sleep between questions (default 3.0, use 0 to disable)",
    )
    args = parser.parse_args()

    if args.output is None and not args.no_pipeline:
        args.output = BASELINES_DIR / f"{args.label}_baseline.json"

    run(
        limit=args.limit,
        output=args.output,
        no_pipeline=args.no_pipeline,
        label=args.label,
        delay=args.delay,
    )
