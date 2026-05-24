"""
Метрики для оценки качества Retrieval компонента RAG системы.

Основные метрики:
- Hit Rate @ K: Найден ли хотя бы один релевантный документ в топ-K
- MRR (Mean Reciprocal Rank): Средняя обратная позиция первого релевантного документа
- NDCG @ K: Normalized Discounted Cumulative Gain
- Precision @ K: Точность в топ-K результатах
- Recall @ K: Полнота в топ-K результатах
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict


def hit_rate_at_k(
    retrieved_docs: List[str], relevant_docs: List[str], k: int = 10
) -> float:
    """
    Hit Rate @ K: Найден ли хотя бы один релевантный документ в топ-K.

    Args:
        retrieved_docs: Список ID извлеченных документов (упорядочен по релевантности)
        relevant_docs: Список ID релевантных документов (ground truth)
        k: Количество топ документов для проверки

    Returns:
        1.0 если хотя бы один релевантный документ найден в топ-K, иначе 0.0
    """
    top_k = retrieved_docs[:k]
    return 1.0 if any(doc in relevant_docs for doc in top_k) else 0.0


def mean_reciprocal_rank(
    retrieved_docs_list: List[List[str]], relevant_docs_list: List[List[str]]
) -> float:
    """
    MRR (Mean Reciprocal Rank): Средняя обратная позиция первого релевантного документа.

    Args:
        retrieved_docs_list: Список списков извлеченных документов для каждого запроса
        relevant_docs_list: Список списков релевантных документов для каждого запроса

    Returns:
        MRR score от 0.0 до 1.0
    """
    reciprocal_ranks = []

    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        for i, doc in enumerate(retrieved, start=1):
            if doc in relevant:
                reciprocal_ranks.append(1.0 / i)
                break
        else:
            reciprocal_ranks.append(0.0)

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def precision_at_k(
    retrieved_docs: List[str], relevant_docs: List[str], k: int = 10
) -> float:
    """
    Precision @ K: Доля релевантных документов среди топ-K извлеченных.

    Args:
        retrieved_docs: Список ID извлеченных документов
        relevant_docs: Список ID релевантных документов
        k: Количество топ документов

    Returns:
        Precision от 0.0 до 1.0
    """
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_in_top_k / k if k > 0 else 0.0


def recall_at_k(
    retrieved_docs: List[str], relevant_docs: List[str], k: int = 10
) -> float:
    """
    Recall @ K: Доля найденных релевантных документов от всех релевантных.

    Args:
        retrieved_docs: Список ID извлеченных документов
        relevant_docs: Список ID релевантных документов
        k: Количество топ документов

    Returns:
        Recall от 0.0 до 1.0
    """
    if not relevant_docs:
        return 0.0

    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_in_top_k / len(relevant_docs)


def dcg_at_k(relevances: List[float], k: int = 10) -> float:
    """
    DCG @ K (Discounted Cumulative Gain).

    Args:
        relevances: Список оценок релевантности (0 = нерелевантен, 1+ = релевантен)
        k: Количество топ позиций

    Returns:
        DCG score
    """
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+1)) для i от 0 до k-1
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)


def ndcg_at_k(
    retrieved_docs: List[str],
    relevant_docs: Dict[str, float],
    k: int = 10,
) -> float:
    """
    NDCG @ K (Normalized Discounted Cumulative Gain).

    Args:
        retrieved_docs: Список ID извлеченных документов (упорядочен)
        relevant_docs: Словарь {doc_id: relevance_score}, где score обычно 0, 1 или 2
                      (0 = нерелевантен, 1 = частично релевантен, 2 = полностью релевантен)
        k: Количество топ позиций

    Returns:
        NDCG от 0.0 до 1.0
    """
    # Получаем relevance scores для извлеченных документов
    retrieved_relevances = [relevant_docs.get(doc, 0.0) for doc in retrieved_docs[:k]]

    # Ideal DCG: сортируем все релевантные документы по убыванию релевантности
    ideal_relevances = sorted(relevant_docs.values(), reverse=True)[:k]

    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    retrieved_docs: List[str],
    relevant_docs: List[str] | Dict[str, float],
    k: int = 10,
) -> Dict[str, float]:
    """
    Вычисляет все retrieval метрики для одного запроса.

    Args:
        retrieved_docs: Список ID извлеченных документов (упорядочен по релевантности)
        relevant_docs: Список ID релевантных документов ИЛИ словарь {doc_id: relevance_score}
        k: Количество топ документов для оценки

    Returns:
        Словарь с метриками
    """
    # Конвертируем в список для метрик, которым нужен просто список
    if isinstance(relevant_docs, dict):
        relevant_list = [doc for doc, score in relevant_docs.items() if score > 0]
    else:
        relevant_list = relevant_docs

    metrics = {
        f"hit_rate@{k}": hit_rate_at_k(retrieved_docs, relevant_list, k),
        f"precision@{k}": precision_at_k(retrieved_docs, relevant_list, k),
        f"recall@{k}": recall_at_k(retrieved_docs, relevant_list, k),
    }

    # NDCG требует scoring, если у нас есть
    if isinstance(relevant_docs, dict):
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs, relevant_docs, k)

    return metrics


def evaluate_retrieval_batch(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[List[str] | Dict[str, float]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Вычисляет средние retrieval метрики для батча запросов.

    Args:
        retrieved_docs_list: Список списков извлеченных документов
        relevant_docs_list: Список списков/словарей релевантных документов
        k: Количество топ документов

    Returns:
        Словарь со средними метриками + MRR
    """
    all_metrics = []

    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        metrics = evaluate_retrieval(retrieved, relevant, k)
        all_metrics.append(metrics)

    # Усредняем метрики
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # Добавляем MRR (требует специальной обработки)
    relevant_lists = [
        list(rel.keys()) if isinstance(rel, dict) else rel for rel in relevant_docs_list
    ]
    avg_metrics["mrr"] = mean_reciprocal_rank(retrieved_docs_list, relevant_lists)

    return avg_metrics


# Пример использования
if __name__ == "__main__":
    # Пример 1: Простой список релевантных документов
    retrieved = ["doc_3", "doc_1", "doc_5", "doc_2", "doc_7"]
    relevant = ["doc_1", "doc_2", "doc_4"]

    metrics = evaluate_retrieval(retrieved, relevant, k=5)
    print("Метрики для примера 1:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Пример 2: С оценками релевантности для NDCG
    relevant_scored = {
        "doc_1": 2.0,  # highly relevant
        "doc_2": 1.0,  # somewhat relevant
        "doc_4": 2.0,  # highly relevant
    }

    metrics_scored = evaluate_retrieval(retrieved, relevant_scored, k=5)
    print("\nМетрики для примера 2 (с NDCG):")
    for metric, value in metrics_scored.items():
        print(f"  {metric}: {value:.3f}")

    # Пример 3: Батч запросов
    retrieved_batch = [
        ["doc_3", "doc_1", "doc_5"],
        ["doc_2", "doc_4", "doc_1"],
    ]
    relevant_batch = [["doc_1", "doc_2"], ["doc_1", "doc_3"]]

    batch_metrics = evaluate_retrieval_batch(retrieved_batch, relevant_batch, k=3)
    print("\nСредние метрики для батча:")
    for metric, value in batch_metrics.items():
        print(f"  {metric}: {value:.3f}")
