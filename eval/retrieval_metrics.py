"""
Retrieval quality metrics for the RAG pipeline.

Metrics:
- Hit Rate @ K: is at least one relevant document present in the top-K results?
- MRR (Mean Reciprocal Rank): average reciprocal rank of the first relevant document
- NDCG @ K: Normalized Discounted Cumulative Gain
- Precision @ K: fraction of top-K results that are relevant
- Recall @ K: fraction of relevant documents found in top-K
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict


def hit_rate_at_k(
    retrieved_docs: List[str], relevant_docs: List[str], k: int = 10
) -> float:
    """
    Hit Rate @ K: is at least one relevant document present in the top-K results?

    Args:
        retrieved_docs: List of retrieved document IDs ordered by relevance
        relevant_docs: List of relevant document IDs (ground truth)
        k: Number of top documents to check

    Returns:
        1.0 if at least one relevant document is in top-K, else 0.0
    """
    top_k = retrieved_docs[:k]
    return 1.0 if any(doc in relevant_docs for doc in top_k) else 0.0


def mean_reciprocal_rank(
    retrieved_docs_list: List[List[str]], relevant_docs_list: List[List[str]]
) -> float:
    """
    MRR (Mean Reciprocal Rank): average reciprocal rank of the first relevant document.

    Args:
        retrieved_docs_list: List of retrieved document ID lists, one per query
        relevant_docs_list: List of relevant document ID lists, one per query

    Returns:
        MRR score from 0.0 to 1.0
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
    Precision @ K: fraction of top-K retrieved documents that are relevant.

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Number of top documents

    Returns:
        Precision from 0.0 to 1.0
    """
    top_k = retrieved_docs[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
    return relevant_in_top_k / k if k > 0 else 0.0


def recall_at_k(
    retrieved_docs: List[str], relevant_docs: List[str], k: int = 10
) -> float:
    """
    Recall @ K: fraction of all relevant documents found in top-K.

    Args:
        retrieved_docs: List of retrieved document IDs
        relevant_docs: List of relevant document IDs
        k: Number of top documents

    Returns:
        Recall from 0.0 to 1.0
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
        relevances: List of relevance scores (0 = not relevant, 1+ = relevant)
        k: Number of top positions

    Returns:
        DCG score
    """
    relevances = np.array(relevances[:k])
    if relevances.size == 0:
        return 0.0

    # DCG = sum(rel_i / log2(i+1)) for i from 0 to k-1
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
        retrieved_docs: List of retrieved document IDs ordered by rank
        relevant_docs: Dict {doc_id: relevance_score}, where score is typically 0, 1, or 2
                      (0 = not relevant, 1 = partially relevant, 2 = fully relevant)
        k: Number of top positions

    Returns:
        NDCG from 0.0 to 1.0
    """
    # Get relevance scores for retrieved documents
    retrieved_relevances = [relevant_docs.get(doc, 0.0) for doc in retrieved_docs[:k]]

    # Ideal DCG: sort all relevant documents by descending relevance
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
    Compute all retrieval metrics for a single query.

    Args:
        retrieved_docs: List of retrieved document IDs ordered by relevance
        relevant_docs: List of relevant document IDs OR dict {doc_id: relevance_score}
        k: Number of top documents to evaluate

    Returns:
        Dict with metric values
    """
    # Convert to list for metrics that only need a plain list
    if isinstance(relevant_docs, dict):
        relevant_list = [doc for doc, score in relevant_docs.items() if score > 0]
    else:
        relevant_list = relevant_docs

    metrics = {
        f"hit_rate@{k}": hit_rate_at_k(retrieved_docs, relevant_list, k),
        f"precision@{k}": precision_at_k(retrieved_docs, relevant_list, k),
        f"recall@{k}": recall_at_k(retrieved_docs, relevant_list, k),
    }

    # NDCG requires scoring dict
    if isinstance(relevant_docs, dict):
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_docs, relevant_docs, k)

    return metrics


def evaluate_retrieval_batch(
    retrieved_docs_list: List[List[str]],
    relevant_docs_list: List[List[str] | Dict[str, float]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Compute averaged retrieval metrics over a batch of queries.

    Args:
        retrieved_docs_list: List of retrieved document ID lists
        relevant_docs_list: List of relevant document ID lists or scoring dicts
        k: Number of top documents

    Returns:
        Dict with averaged metrics plus MRR
    """
    all_metrics = []

    for retrieved, relevant in zip(retrieved_docs_list, relevant_docs_list):
        metrics = evaluate_retrieval(retrieved, relevant, k)
        all_metrics.append(metrics)

    # Average per-metric values
    avg_metrics = {}
    if all_metrics:
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    # MRR requires special handling (needs all queries together)
    relevant_lists = [
        list(rel.keys()) if isinstance(rel, dict) else rel for rel in relevant_docs_list
    ]
    avg_metrics["mrr"] = mean_reciprocal_rank(retrieved_docs_list, relevant_lists)

    return avg_metrics


# Usage example
if __name__ == "__main__":
    # Example 1: plain list of relevant documents
    retrieved = ["doc_3", "doc_1", "doc_5", "doc_2", "doc_7"]
    relevant = ["doc_1", "doc_2", "doc_4"]

    metrics = evaluate_retrieval(retrieved, relevant, k=5)
    print("Example 1 metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    # Example 2: with relevance scores for NDCG
    relevant_scored = {
        "doc_1": 2.0,  # highly relevant
        "doc_2": 1.0,  # somewhat relevant
        "doc_4": 2.0,  # highly relevant
    }

    metrics_scored = evaluate_retrieval(retrieved, relevant_scored, k=5)
    print("\nExample 2 metrics (with NDCG):")
    for metric, value in metrics_scored.items():
        print(f"  {metric}: {value:.3f}")

    # Example 3: batch of queries
    retrieved_batch = [
        ["doc_3", "doc_1", "doc_5"],
        ["doc_2", "doc_4", "doc_1"],
    ]
    relevant_batch = [["doc_1", "doc_2"], ["doc_1", "doc_3"]]

    batch_metrics = evaluate_retrieval_batch(retrieved_batch, relevant_batch, k=3)
    print("\nBatch averaged metrics:")
    for metric, value in batch_metrics.items():
        print(f"  {metric}: {value:.3f}")
