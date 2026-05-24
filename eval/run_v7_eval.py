"""Evaluation runner for V7 RAG pipeline.

Runs the golden dataset through the V7 graph and measures:
  - faithfulness       — are claims grounded in retrieved context?
  - answer_relevance   — does the answer address the question?
  - correctness        — does the answer match the ground truth? (LLM-judge, 0-10)
  - false_sufficiency_rate — % of simple-path answers that scored badly (< threshold)

Usage:
    cd /home/petr/projects/safety-incident-analyzer
    source venv/bin/activate
    python eval/run_v7_eval.py
    python eval/run_v7_eval.py --limit 5          # quick smoke test
    python eval/run_v7_eval.py --skip-judge       # pipeline only, no LLM judge (~$0)
    python eval/run_v7_eval.py --output benchmarks/eval_v7_custom.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.advanced_generation_metrics import (
    evaluate_answer_relevance,
    evaluate_faithfulness,
)
from src.backends.vector_store import get_vector_store_backend
from src.llm_factory import get_gemini_llm
from src.v7.bridge import init_v7_pipeline
from src.v7.graph import build_graph

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(__file__).parent.parent / "tests" / "dataset.csv"
DEFAULT_OUTPUT = (
    Path(__file__).parent.parent
    / "benchmarks"
    / f"eval_v7_{date.today().isoformat()}.jsonl"
)

# False-sufficiency: "simple" path answer with correctness < this threshold → false positive
FALSE_SUFFICIENCY_THRESHOLD = 5.0  # out of 10


# ── Dataset ───────────────────────────────────────────────────────────────────


def load_dataset(path: Path) -> list[dict[str, str]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            gt = row.get("ground_truth", "").strip()
            if q and gt:
                rows.append({"question": q, "ground_truth": gt})
    return rows


# ── Graph runner ──────────────────────────────────────────────────────────────


def run_query(graph, question: str) -> dict[str, Any]:
    """Run one question through V7 graph, return structured result."""
    start = time.time()
    state = graph.invoke({"query": question})
    elapsed = round(time.time() - start, 2)

    answer = state.get("answer", "")
    final_passages = state.get("final_passages") or []
    retrieval_attempts = state.get("retrieval_attempts") or []

    # Determine path taken: simple or complex
    stages = [a.get("stage", "unknown") for a in retrieval_attempts]
    path = "complex" if "complex" in stages else "simple"

    # Build context string from retrieved passages
    context = "\n\n".join(p.get("text", "") for p in final_passages if p.get("text"))

    return {
        "answer": answer,
        "context": context,
        "path": path,
        "elapsed_sec": elapsed,
        "retrieval_attempts": len(retrieval_attempts),
    }


# ── Correctness judge ─────────────────────────────────────────────────────────


def evaluate_correctness(
    question: str, ground_truth: str, answer: str, llm
) -> dict[str, Any]:
    """LLM-as-judge: how close is answer to ground_truth? Returns score 0-10."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """Ты — строгий судья качества ответов. Оцени, насколько Ответ соответствует Эталонному ответу.

Критерии:
- 9-10: Ответ полный, точный, содержит все ключевые факты эталона.
- 7-8: Ответ в целом верный, но упускает некоторые детали.
- 5-6: Ответ частично верный, значимые детали отсутствуют.
- 3-4: Ответ поверхностный или содержит ошибки.
- 0-2: Ответ неверный или не по теме.

Верни JSON: {{"score": <число 0-10>, "reasoning": "<краткое объяснение>"}}

Вопрос: {question}
Эталонный ответ: {ground_truth}
Ответ для оценки: {answer}

JSON:"""
    )

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"question": question, "ground_truth": ground_truth, "answer": answer}
    )

    try:
        import re

        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "correctness_score": float(data.get("score", 0.0)),
                "correctness_reasoning": data.get("reasoning", ""),
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    return {
        "correctness_score": 0.0,
        "correctness_reasoning": f"parse error: {response[:100]}",
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    limit: int | None = None,
    output: Path = DEFAULT_OUTPUT,
    skip_judge: bool = False,
) -> None:
    print("Loading dataset...")
    dataset = load_dataset(DATASET_PATH)
    if limit:
        dataset = dataset[:limit]
    print(f"  {len(dataset)} questions")

    print("Initializing V7 graph...")
    vector_store = get_vector_store_backend(load_existing=True)
    init_v7_pipeline(vector_store)
    graph = build_graph().compile()
    print("  Graph ready.")

    judge_llm = None
    if skip_judge:
        print("  [--skip-judge] LLM judge disabled — pipeline only, $0 cost.\n")
    else:
        print("Loading judge LLM...")
        judge_llm = get_gemini_llm(temperature=0.0)
        print("  Judge ready.\n")

    results = []
    for i, item in enumerate(dataset, 1):
        question = item["question"]
        ground_truth = item["ground_truth"]
        print(f"[{i}/{len(dataset)}] {question[:70]}...")

        # Run graph
        try:
            run_result = run_query(graph, question)
        except Exception as e:
            print(f"  ERROR running graph: {e}")
            results.append({"question": question, "error": str(e)})
            continue

        answer = run_result["answer"]
        context = run_result["context"]
        path = run_result["path"]

        if not answer:
            print(f"  WARNING: empty answer (path={path})")
            results.append(
                {
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": "",
                    "path": path,
                    "error": "empty answer",
                }
            )
            continue

        if skip_judge:
            record = {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "path": path,
                "elapsed_sec": run_result["elapsed_sec"],
                "retrieval_attempts": run_result["retrieval_attempts"],
            }
            results.append(record)
            print(f"  path={path} | elapsed={run_result['elapsed_sec']:.1f}s")
            continue

        # Evaluate with LLM judge
        try:
            faithfulness = evaluate_faithfulness(question, context, answer, judge_llm)
        except Exception as e:
            faithfulness = {"faithfulness_score": 0.0, "faithfulness_reasoning": str(e)}

        try:
            relevance = evaluate_answer_relevance(question, answer, judge_llm)
        except Exception as e:
            print(f"  WARNING: relevance eval failed: {e}")
            relevance = {"answer_relevance_score": 0.0}

        try:
            correctness = evaluate_correctness(
                question, ground_truth, answer, judge_llm
            )
        except Exception as e:
            correctness = {"correctness_score": 0.0, "correctness_reasoning": str(e)}

        record = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "path": path,
            "elapsed_sec": run_result["elapsed_sec"],
            "retrieval_attempts": run_result["retrieval_attempts"],
            **faithfulness,
            **relevance,
            **correctness,
        }
        results.append(record)

        print(
            f"  path={path} | "
            f"faith={faithfulness.get('faithfulness_score', 0):.2f} | "
            f"rel={relevance.get('answer_relevance_score', 0):.2f} | "
            f"correct={correctness.get('correctness_score', 0):.1f}/10"
        )

    # Aggregate
    valid = [r for r in results if "error" not in r and r.get("answer")]
    n = len(valid)

    if n == 0:
        print("\nNo valid results to aggregate.")
        return

    avg_elapsed = sum(r.get("elapsed_sec", 0) for r in valid) / n
    complex_rate = sum(1 for r in valid if r.get("path") == "complex") / n
    abstain_count = sum(
        1
        for r in valid
        if not r.get("answer") or r.get("answer", "").startswith("Не могу")
    )

    aggregate: dict[str, Any] = {
        "complex_path_rate": round(complex_rate, 3),
        "mean_elapsed_sec": round(avg_elapsed, 2),
        "answered": n,
        "abstained": abstain_count,
    }

    if not skip_judge:
        avg_faith = sum(r.get("faithfulness_score", 0) for r in valid) / n
        avg_rel = sum(r.get("answer_relevance_score", 0) for r in valid) / n
        avg_correct = sum(r.get("correctness_score", 0) for r in valid) / n
        simple_path = [r for r in valid if r.get("path") == "simple"]
        false_sufficiency_cases = [
            r
            for r in simple_path
            if r.get("correctness_score", 10) < FALSE_SUFFICIENCY_THRESHOLD
        ]
        false_sufficiency_rate = (
            len(false_sufficiency_cases) / len(simple_path) if simple_path else 0.0
        )
        aggregate.update(
            {
                "faithfulness": round(avg_faith, 3),
                "answer_relevance": round(avg_rel, 3),
                "correctness_mean": round(avg_correct, 2),
                "false_sufficiency_rate": round(false_sufficiency_rate, 3),
            }
        )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "skip_judge": skip_judge,
        "dataset": str(DATASET_PATH),
        "dataset_size": len(dataset),
        "valid_results": n,
        "aggregate": aggregate,
        "false_sufficiency_threshold": FALSE_SUFFICIENCY_THRESHOLD,
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 55}")
    print(f"Results ({n}/{len(dataset)} valid)")
    if not skip_judge:
        print(
            f"  Faithfulness:          {aggregate['faithfulness']:.3f}  (target >0.85)"
        )
        print(
            f"  Answer Relevance:      {aggregate['answer_relevance']:.3f}  (target >0.85)"
        )
        print(
            f"  Correctness:           {aggregate['correctness_mean']:.1f}/10  (target >7.5)"
        )
        print(
            f"  False-sufficiency:     {aggregate['false_sufficiency_rate']:.1%}  (target <10%)"
        )
        if false_sufficiency_cases:
            print(f"\nFalse-sufficiency cases ({len(false_sufficiency_cases)}):")
            for r in false_sufficiency_cases:
                print(
                    f"  - [{r.get('correctness_score', 0):.1f}] {r['question'][:60]}..."
                )
    else:
        print("  [skip-judge mode] Quality metrics not computed.")
    print(f"  Complex path rate:     {complex_rate:.1%}")
    print(f"  Mean latency:          {avg_elapsed:.1f}s")
    print(f"\nSaved → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate V7 RAG pipeline")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of questions"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path"
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM judge (no faithfulness/correctness scoring). Pipeline runs only. Cost ~$0.",
    )
    args = parser.parse_args()
    run(limit=args.limit, output=args.output, skip_judge=args.skip_judge)
