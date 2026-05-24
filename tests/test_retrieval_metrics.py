"""
Unit тесты для retrieval метрик.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest
from eval.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_retrieval,
    evaluate_retrieval_batch,
)


class TestHitRate:
    def test_hit_found(self):
        """Тест: релевантный документ найден в топ-K"""
        retrieved = ["doc_3", "doc_1", "doc_5"]
        relevant = ["doc_1", "doc_2"]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0

    def test_hit_not_found(self):
        """Тест: релевантный документ НЕ найден в топ-K"""
        retrieved = ["doc_3", "doc_4", "doc_5"]
        relevant = ["doc_1", "doc_2"]
        assert hit_rate_at_k(retrieved, relevant, k=3) == 0.0

    def test_hit_beyond_k(self):
        """Тест: релевантный документ есть, но за пределами K"""
        retrieved = ["doc_3", "doc_4", "doc_1"]  # doc_1 на позиции 3
        relevant = ["doc_1"]
        assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
        assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


class TestMRR:
    def test_mrr_first_position(self):
        """Тест: релевантный документ на первой позиции"""
        retrieved = [["doc_1", "doc_2", "doc_3"]]
        relevant = [["doc_1"]]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_mrr_second_position(self):
        """Тест: релевантный документ на второй позиции"""
        retrieved = [["doc_2", "doc_1", "doc_3"]]
        relevant = [["doc_1"]]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_mrr_average(self):
        """Тест: среднее по нескольким запросам"""
        retrieved = [
            ["doc_1", "doc_2"],  # RR = 1.0
            ["doc_2", "doc_1"],  # RR = 0.5
        ]
        relevant = [["doc_1"], ["doc_1"]]
        expected = (1.0 + 0.5) / 2
        assert mean_reciprocal_rank(retrieved, relevant) == expected

    def test_mrr_not_found(self):
        """Тест: релевантный документ не найден"""
        retrieved = [["doc_2", "doc_3"]]
        relevant = [["doc_1"]]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0


class TestPrecisionRecall:
    def test_precision_perfect(self):
        """Тест: идеальная точность"""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = ["doc_1", "doc_2", "doc_3"]
        assert precision_at_k(retrieved, relevant, k=3) == 1.0

    def test_precision_half(self):
        """Тест: 50% точность"""
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = ["doc_1", "doc_3"]
        assert precision_at_k(retrieved, relevant, k=4) == 0.5

    def test_recall_perfect(self):
        """Тест: идеальная полнота"""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = ["doc_1", "doc_2"]
        assert recall_at_k(retrieved, relevant, k=3) == 1.0

    def test_recall_partial(self):
        """Тест: частичная полнота"""
        retrieved = ["doc_1", "doc_3"]
        relevant = ["doc_1", "doc_2", "doc_4"]  # 1 из 3 найден
        assert recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)

    def test_recall_empty_relevant(self):
        """Тест: нет релевантных документов"""
        retrieved = ["doc_1", "doc_2"]
        relevant = []
        assert recall_at_k(retrieved, relevant, k=2) == 0.0


class TestNDCG:
    def test_ndcg_perfect_ranking(self):
        """Тест: идеальное ранжирование"""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant = {"doc_1": 2.0, "doc_2": 1.0, "doc_3": 1.0}
        # Идеальный порядок: doc_1 (2.0) > doc_2/doc_3 (1.0)
        assert ndcg_at_k(retrieved, relevant, k=3) == 1.0

    def test_ndcg_worst_ranking(self):
        """Тест: худшее ранжирование"""
        retrieved = ["doc_3", "doc_2", "doc_1"]
        relevant = {"doc_1": 2.0, "doc_2": 1.0, "doc_3": 0.0}
        # Худший порядок
        ndcg = ndcg_at_k(retrieved, relevant, k=3)
        assert 0.0 <= ndcg < 1.0  # Не идеально

    def test_ndcg_no_relevant(self):
        """Тест: нет релевантных документов"""
        retrieved = ["doc_1", "doc_2"]
        relevant = {"doc_3": 1.0, "doc_4": 1.0}
        assert ndcg_at_k(retrieved, relevant, k=2) == 0.0


class TestEvaluateRetrieval:
    def test_full_evaluation(self):
        """Тест: полная оценка с всеми метриками"""
        retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
        relevant = ["doc_1", "doc_3", "doc_5"]

        metrics = evaluate_retrieval(retrieved, relevant, k=4)

        # Проверяем наличие всех метрик
        assert "hit_rate@4" in metrics
        assert "precision@4" in metrics
        assert "recall@4" in metrics

        # Hit rate = 1.0 (doc_1 и doc_3 найдены)
        assert metrics["hit_rate@4"] == 1.0

        # Precision = 2/4 = 0.5
        assert metrics["precision@4"] == 0.5

        # Recall = 2/3 ≈ 0.667
        assert metrics["recall@4"] == pytest.approx(2 / 3)

    def test_with_scored_relevance(self):
        """Тест: оценка с NDCG"""
        retrieved = ["doc_1", "doc_2", "doc_3"]
        relevant_scored = {"doc_1": 2.0, "doc_2": 1.0}

        metrics = evaluate_retrieval(retrieved, relevant_scored, k=3)

        # Должен быть NDCG
        assert "ndcg@3" in metrics
        assert 0.0 <= metrics["ndcg@3"] <= 1.0


class TestEvaluateRetrievalBatch:
    def test_batch_evaluation(self):
        """Тест: батч оценка"""
        retrieved_batch = [
            ["doc_1", "doc_2", "doc_3"],
            ["doc_2", "doc_1", "doc_4"],
        ]
        relevant_batch = [["doc_1", "doc_2"], ["doc_1", "doc_3"]]

        metrics = evaluate_retrieval_batch(retrieved_batch, relevant_batch, k=3)

        # Должны быть все метрики + MRR
        assert "hit_rate@3" in metrics
        assert "precision@3" in metrics
        assert "recall@3" in metrics
        assert "mrr" in metrics

        # Hit rate должен быть 1.0 (в обоих запросах есть релевантные в топ-3)
        assert metrics["hit_rate@3"] == 1.0

        # MRR: (1.0 + 0.5) / 2 = 0.75
        assert metrics["mrr"] == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
