"""
Полная оценка RAG системы: retrieval + generation + pipeline метрики.

Использование:
    python eval/run_full_evaluation.py [--dataset DATASET_PATH] [--output OUTPUT_PATH]
"""

import sys
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent.parent))

from eval.advanced_generation_metrics import (
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_citation_quality,
)
from eval.custom_evaluators import check_correctness
from src.llm_factory import get_llm
from src.final_chain import create_final_hybrid_chain
from agents.multiagent_rag import MultiAgentRAGWorkflow


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Загружает датасет из CSV."""
    dataset = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dataset = list(reader)
    return dataset


def evaluate_single_query(
    question: str,
    ground_truth: str,
    chain,
    retriever,
    llm,
    mode: str = "rag",
) -> Dict[str, Any]:
    """
    Оценивает один запрос (retrieval + generation).

    Returns:
        Dict с метриками и результатами
    """
    result = {}

    # 1. Retrieval
    retrieval_start = time.time()
    retrieved_docs = retriever.invoke(
        question
    )  # Используем invoke вместо get_relevant_documents
    retrieval_time = time.time() - retrieval_start

    result["retrieval_time"] = retrieval_time
    result["num_retrieved_docs"] = len(retrieved_docs)
    result["retrieved_doc_ids"] = [str(i) for i in range(len(retrieved_docs))]

    # 2. Generation
    generation_start = time.time()
    try:
        if mode == "mas":
            # Multi-Agent RAG (через LangGraph ReAct agent)
            res = chain.invoke(question)
            answer = res.get("final_answer", "")
            thought = res.get("draft_answer", "")
            # Контекст из найденных агентом чанков, fallback на retriever
            chunks = res.get("chunks_found", [])
            if chunks:
                context = "\n\n".join(c.get("text", "") for c in chunks)
            else:
                context = "\n\n".join([d.page_content for d in retrieved_docs[:20]])
        else:
            # Стандартный RAG (цепочка LangChain)
            response = chain.invoke({"question": question})
            answer = (
                response.get("output", "")
                if isinstance(response, dict)
                else str(response)
            )
            context = response.get("context", "") if isinstance(response, dict) else ""
            thought = ""

    except Exception as e:
        print(f"  ❌ Ошибка генерации ({mode}): {e}")
        answer = ""
        context = ""
        thought = ""

    generation_time = time.time() - generation_start
    total_time = retrieval_time + generation_time

    result["generation_time"] = generation_time
    result["total_time"] = total_time
    result["answer"] = answer
    result["context"] = context
    result["thought"] = thought

    # 3. Generation метрики
    if answer:
        # Faithfulness
        try:
            faith_metrics = evaluate_faithfulness(question, context, answer, llm)
            result.update(faith_metrics)
        except Exception as e:
            print(f"  ⚠️  Faithfulness eval failed: {e}")
            result["faithfulness_score"] = 0.0

        # Answer relevance
        try:
            rel_metrics = evaluate_answer_relevance(question, answer, llm)
            result.update(rel_metrics)
        except Exception as e:
            print(f"  ⚠️  Relevance eval failed: {e}")
            result["answer_relevance_score"] = 0.0

        # Citation quality
        cite_metrics = evaluate_citation_quality(answer, context, [])
        result.update(cite_metrics)

        # Correctness (используем существующий evaluator)
        # Эмулируем run и example для check_correctness
        class MockRun:
            def __init__(self, output):
                self.outputs = {"output": output}

        class MockExample:
            def __init__(self, question, ground_truth):
                self.inputs = {"question": question}
                self.outputs = {"ground_truth": ground_truth}

        try:
            correctness = check_correctness(
                MockRun(answer), MockExample(question, ground_truth)
            )
            result["correctness_score"] = correctness.get("score", 0.0)
            result["correctness_comment"] = correctness.get("comment", "")
        except Exception as e:
            print(f"  ⚠️  Correctness eval failed: {e}")
            result["correctness_score"] = 0.0

    return result


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Агрегирует метрики по всем запросам."""
    import numpy as np

    metrics = {}

    # Числовые метрики для усреднения
    numeric_keys = [
        "retrieval_time",
        "generation_time",
        "total_time",
        "num_retrieved_docs",
        "faithfulness_score",
        "answer_relevance_score",
        "correctness_score",
        "citation_count",
        "unique_citation_count",
    ]

    for key in numeric_keys:
        values = [r.get(key, 0.0) for r in results if key in r]
        if values:
            metrics[f"mean_{key}"] = float(np.mean(values))
            metrics[f"std_{key}"] = float(np.std(values))
            metrics[f"min_{key}"] = float(np.min(values))
            metrics[f"max_{key}"] = float(np.max(values))
            metrics[f"p95_{key}"] = float(np.percentile(values, 95))

    # Boolean метрики
    has_citations_count = sum(1 for r in results if r.get("has_citations", False))
    metrics["citation_rate"] = has_citations_count / len(results) if results else 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Полная оценка RAG системы")
    parser.add_argument(
        "--mode",
        choices=["rag", "mas"],
        default="rag",
        help="Режим работы: обычный RAG (rag) или многоагентный (mas)",
    )
    parser.add_argument(
        "--dataset",
        default="tests/dataset.csv",
        help="Путь к датасету (CSV)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Путь для сохранения результатов (JSONL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество вопросов для тестирования",
    )
    args = parser.parse_args()

    # Формируем путь вывода по умолчанию, если не задан
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"benchmarks/eval_{args.mode}_{timestamp}.jsonl"

    print(f"🚀 Запуск оценки в режиме: {args.mode.upper()}")
    print(f"📂 Датасет: {args.dataset}")

    # Инициализация
    print("\n🔧 Инициализация компонентов...")
    llm = get_llm()
    rag_chain, retriever, agent_retriever = create_final_hybrid_chain()

    if args.mode == "mas":
        print("🤖 Загрузка Multi-Agent RAG Workflow...")
        chain = MultiAgentRAGWorkflow(agent_retriever)
    else:
        chain = rag_chain

    # Загрузка датасета
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[: args.limit]
    print(f"📊 Загружено {len(dataset)} вопросов")

    # Оценка каждого вопроса
    print("\n📝 Оценка запросов...")
    results = []

    for i, item in enumerate(dataset, 1):
        question = item["question"].replace("[cite_start]", "").strip()
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{len(dataset)}] {question[:60]}...")

        result = evaluate_single_query(
            question, ground_truth, chain, retriever, llm, mode=args.mode
        )

        # Debug: Print full answer
        print(f"  🤖 Ответ: {result.get('answer', '')[:200]}...")
        if result.get("thought"):
            print(f"  💭 Мысли: {result.get('thought', '')[:200]}...")

        result["question"] = question
        result["ground_truth"] = ground_truth
        results.append(result)

        # Краткий вывод
        print(f"  ✅ Correctness: {result.get('correctness_score', 0):.1f}/10")
        print(f"  📊 Faithfulness: {result.get('faithfulness_score', 0):.2f}")
        print(f"  ⏱️  Время: {result.get('total_time', 0):.2f}s")

    # Агрегация метрик
    print("\n📈 Агрегация метрик...")
    agg_metrics = aggregate_metrics(results)

    # Вывод результатов
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЕ МЕТРИКИ")
    print("=" * 60)

    print("\n🎯 Correctness:")
    print(f"  Среднее:     {agg_metrics.get('mean_correctness_score', 0):.2f}/10")
    print(
        f"  Мин/Макс:    {agg_metrics.get('min_correctness_score', 0):.1f} / {agg_metrics.get('max_correctness_score', 0):.1f}"
    )

    print("\n✅ Faithfulness:")
    print(f"  Среднее:     {agg_metrics.get('mean_faithfulness_score', 0):.3f}")
    print(f"  Мин:         {agg_metrics.get('min_faithfulness_score', 0):.3f}")

    print("\n🔗 Answer Relevance:")
    print(f"  Среднее:     {agg_metrics.get('mean_answer_relevance_score', 0):.3f}")

    print("\n📚 Citations:")
    print(f"  Citation rate:        {agg_metrics.get('citation_rate', 0):.1%}")
    print(f"  Среднее на ответ:     {agg_metrics.get('mean_citation_count', 0):.1f}")

    print("\n⏱️  Performance:")
    print(f"  Среднее время:        {agg_metrics.get('mean_total_time', 0):.2f}s")
    print(f"  P95 время:            {agg_metrics.get('p95_total_time', 0):.2f}s")
    print(f"  Retrieval время:      {agg_metrics.get('mean_retrieval_time', 0):.2f}s")
    print(f"  Generation время:     {agg_metrics.get('mean_generation_time', 0):.2f}s")

    # Сохранение результатов
    print(f"\n💾 Сохранение результатов в {args.output}...")

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "dataset_size": len(dataset),
        "aggregate_metrics": agg_metrics,
        "detailed_results": results,
    }

    # Сохранение в JSONL (append)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

    print("✅ Готово!")

    # Проверка целевых метрик
    print("\n🎯 Проверка целевых метрик:")
    checks = [
        ("Correctness > 7.0", agg_metrics.get("mean_correctness_score", 0) > 7.0),
        ("Faithfulness > 0.85", agg_metrics.get("mean_faithfulness_score", 0) > 0.85),
        (
            "Answer Relevance > 0.80",
            agg_metrics.get("mean_answer_relevance_score", 0) > 0.80,
        ),
        ("P95 Latency < 15s", agg_metrics.get("p95_total_time", 999) < 15.0),
    ]

    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")

    if all(passed for _, passed in checks):
        print("\n🎉 Все целевые метрики достигнуты!")
    else:
        print("\n⚠️  Некоторые метрики требуют улучшения")


if __name__ == "__main__":
    main()
