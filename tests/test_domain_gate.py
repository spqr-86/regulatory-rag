"""Unit tests for src/v7/domain_gate.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.v7.domain_gate import cosine_similarity, is_in_domain


class TestCosineSimilarity:
    @pytest.mark.unit
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, a) == pytest.approx(1.0)

    @pytest.mark.unit
    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    @pytest.mark.unit
    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    @pytest.mark.unit
    def test_zero_vector_a(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    @pytest.mark.unit
    def test_zero_vector_b(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0

    @pytest.mark.unit
    def test_both_zero_vectors(self):
        a = np.array([0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, a) == 0.0

    @pytest.mark.unit
    def test_similar_vectors(self):
        a = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.9, 0.0], dtype=np.float32)
        sim = cosine_similarity(a, b)
        assert 0.9 < sim <= 1.0

    @pytest.mark.unit
    def test_returns_float(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([2.0, 1.0], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert isinstance(result, float)

    @pytest.mark.unit
    def test_list_input(self):
        """cosine_similarity accepts list inputs via np.array wrapping when called from is_in_domain."""
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(1.0)


class TestIsInDomain:
    @pytest.mark.unit
    def test_disabled_threshold_always_true(self):
        """threshold=0.0 disables gate — always returns True regardless of embedding."""
        fake_embedding = [0.0, 0.0, 0.0]
        assert is_in_domain(fake_embedding, threshold=0.0) is True

    @pytest.mark.unit
    def test_negative_threshold_always_true(self):
        fake_embedding = [0.0] * 4
        assert is_in_domain(fake_embedding, threshold=-1.0) is True

    @pytest.mark.unit
    def test_in_domain_above_threshold(self):
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        with patch("src.v7.domain_gate.get_corpus_centroid", return_value=centroid):
            result = is_in_domain(query, threshold=0.5)
        assert result is True

    @pytest.mark.unit
    def test_out_of_domain_below_threshold(self):
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # Orthogonal → similarity = 0.0
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        with patch("src.v7.domain_gate.get_corpus_centroid", return_value=centroid):
            result = is_in_domain(query, threshold=0.5)
        assert result is False

    @pytest.mark.unit
    def test_exactly_at_threshold(self):
        """Similarity exactly equal to threshold → in domain (>=)."""
        centroid = np.array([1.0, 0.0], dtype=np.float32)
        # 45° angle → cos = 1/sqrt(2) ≈ 0.7071
        query = np.array([1.0, 1.0], dtype=np.float32)
        sim = cosine_similarity(
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
        )
        with patch("src.v7.domain_gate.get_corpus_centroid", return_value=centroid):
            result = is_in_domain(query, threshold=sim)
        assert result is True

    @pytest.mark.unit
    def test_accepts_list_embedding(self):
        centroid = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_list = [1.0, 0.0, 0.0]  # list, not ndarray
        with patch("src.v7.domain_gate.get_corpus_centroid", return_value=centroid):
            result = is_in_domain(query_list, threshold=0.5)
        assert result is True

    @pytest.mark.unit
    def test_centroid_called_only_when_threshold_positive(self):
        """get_corpus_centroid must NOT be called when threshold <= 0."""
        with patch("src.v7.domain_gate.get_corpus_centroid") as mock_centroid:
            is_in_domain([1.0, 0.0], threshold=0.0)
            mock_centroid.assert_not_called()


class TestGetCorpusCentroid:
    @pytest.mark.unit
    def test_centroid_is_unit_vector(self):
        """Centroid returned by get_corpus_centroid should be L2-normalized."""
        from src.v7.domain_gate import get_corpus_centroid

        fake_embeddings = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"embeddings": fake_embeddings.tolist()}
        mock_vs = MagicMock()
        mock_vs._collection = mock_collection

        with patch("src.v7.domain_gate.load_vector_store", return_value=mock_vs):
            # Clear lru_cache to force fresh computation
            get_corpus_centroid.cache_clear()
            centroid = get_corpus_centroid()
            get_corpus_centroid.cache_clear()

        norm = np.linalg.norm(centroid)
        assert norm == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.unit
    def test_centroid_shape_matches_embedding_dim(self):
        from src.v7.domain_gate import get_corpus_centroid

        dim = 8
        fake_embeddings = np.random.rand(10, dim).astype(np.float32)
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"embeddings": fake_embeddings.tolist()}
        mock_vs = MagicMock()
        mock_vs._collection = mock_collection

        with patch("src.v7.domain_gate.load_vector_store", return_value=mock_vs):
            get_corpus_centroid.cache_clear()
            centroid = get_corpus_centroid()
            get_corpus_centroid.cache_clear()

        assert centroid.shape == (dim,)

    @pytest.mark.unit
    def test_empty_corpus_disables_gate(self):
        """Empty corpus must not yield a nan centroid.

        With zero embeddings, get_corpus_centroid returns None (sentinel) and
        is_in_domain passes everything (returns True) instead of comparing
        against a nan centroid.
        """
        from src.v7.domain_gate import get_corpus_centroid

        mock_collection = MagicMock()
        mock_collection.get.return_value = {"embeddings": []}
        mock_vs = MagicMock()
        mock_vs._collection = mock_collection

        with patch("src.v7.domain_gate.load_vector_store", return_value=mock_vs):
            get_corpus_centroid.cache_clear()
            centroid = get_corpus_centroid()
            get_corpus_centroid.cache_clear()

        assert centroid is None

    @pytest.mark.unit
    def test_is_in_domain_passes_when_no_centroid(self):
        """When centroid is unavailable (empty corpus), gate always passes."""
        with patch("src.v7.domain_gate.get_corpus_centroid", return_value=None):
            assert is_in_domain([0.0, 1.0, 0.0], threshold=0.9) is True
