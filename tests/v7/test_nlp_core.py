"""Tests for src/v7/nlp_core.py — Stage 1."""

from __future__ import annotations

import pytest

from src.v7.nlp_core import (
    BM25Index,
    compute_doc_diversity,
    compute_keyword_overlap,
    extract_keywords,
    merge_all_passages,
    mmr_select,
    rrf_merge,
)

# ─── extract_keywords ─────────────────────────────────────────────────────


class TestExtractKeywords:
    @pytest.mark.unit
    def test_basic_lemmatization(self):
        """Lemmatization of Russian word forms."""
        keywords = extract_keywords("Проверка ограждений на соответствие требованиям")
        assert "проверка" in keywords
        assert "ограждение" in keywords
        # "соответствие" and "требование" are stop words — must not appear
        assert "соответствие" not in keywords
        assert "требование" not in keywords

    @pytest.mark.unit
    def test_lemmatization_equivalence(self):
        """Different word forms produce the same lemma."""
        kw1 = extract_keywords("ограждения")
        kw2 = extract_keywords("ограждение")
        kw3 = extract_keywords("ограждений")
        assert kw1 == kw2 == kw3

    @pytest.mark.unit
    def test_preserves_document_numbers(self):
        """Regulatory document numbers are preserved as-is."""
        keywords = extract_keywords("СП 1.13130 и ГОСТ 12.1.004-91")
        assert "1.13130" in keywords
        assert "12.1.004-91" in keywords

    @pytest.mark.unit
    def test_preserves_bare_doc_numbers(self):
        """Bare regulatory doc numbers (no dots) survive extraction.

        Order numbers like 2464 ("приказ 2464") and 29н ("приказ 29н") must not
        be dropped — they are high-signal for keyword overlap.
        """
        keywords = extract_keywords("что в приказе 2464 и 29н")
        assert "2464" in keywords
        assert "29н" in keywords

    @pytest.mark.unit
    def test_returns_set(self):
        """Returns a set, not a list."""
        result = extract_keywords("тест")
        assert isinstance(result, set)

    @pytest.mark.unit
    def test_filters_short_words(self):
        """Words shorter than 3 characters are filtered out."""
        keywords = extract_keywords("на в из по от до")
        assert len(keywords) == 0

    @pytest.mark.unit
    def test_empty_text(self):
        keywords = extract_keywords("")
        assert keywords == set()

    @pytest.mark.unit
    def test_stop_words_filtered(self):
        """Stop words do not appear in the result."""
        keywords = extract_keywords("для этого также можно быть")
        assert "мочь" not in keywords
        assert "быть" not in keywords


# ─── BM25Index ─────────────────────────────────────────────────────────────


class TestBM25Index:
    @pytest.fixture
    def corpus(self):
        return [
            {
                "chunk_id": "c1",
                "text": "Требования к ограждениям лестничных клеток",
                "doc_id": "d1",
            },
            {
                "chunk_id": "c2",
                "text": "Пожарная безопасность зданий и сооружений",
                "doc_id": "d2",
            },
            {
                "chunk_id": "c3",
                "text": "Высота ограждения балконов не менее 1200 мм",
                "doc_id": "d1",
            },
            {
                "chunk_id": "c4",
                "text": "Нормы проектирования жилых зданий",
                "doc_id": "d3",
            },
        ]

    @pytest.mark.unit
    def test_build_and_search(self, corpus):
        """BM25Index builds and returns results."""
        index = BM25Index(corpus)
        results = index.search("ограждения", top_k=2)
        assert len(results) <= 2
        assert all("bm25_score" in r for r in results)

    @pytest.mark.unit
    def test_relevance_order(self, corpus):
        """Results are sorted by bm25_score descending."""
        index = BM25Index(corpus)
        results = index.search("ограждения", top_k=4)
        scores = [r["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.unit
    def test_lemmatized_matching(self, corpus):
        """Different word forms retrieve the same documents."""
        index = BM25Index(corpus)
        r1 = index.search("ограждения", top_k=4)
        r2 = index.search("ограждение", top_k=4)
        ids1 = {r["chunk_id"] for r in r1 if r["bm25_score"] > 0}
        ids2 = {r["chunk_id"] for r in r2 if r["bm25_score"] > 0}
        assert ids1 == ids2

    @pytest.mark.unit
    def test_filter_by_doc_id(self, corpus):
        """Filtering by metadata fields."""
        index = BM25Index(corpus)
        results = index.search("ограждения", top_k=4, filters={"doc_id": "d1"})
        assert all(r["doc_id"] == "d1" for r in results)

    @pytest.mark.unit
    def test_empty_query(self, corpus):
        index = BM25Index(corpus)
        results = index.search("", top_k=4)
        assert results == []

    @pytest.mark.unit
    def test_score_field_set(self, corpus):
        """score equals bm25_score when score was absent in the source passage."""
        index = BM25Index(corpus)
        results = index.search("пожарная безопасность", top_k=2)
        for r in results:
            assert r["score"] == r["bm25_score"]


# ─── rrf_merge ─────────────────────────────────────────────────────────────


class TestRRFMerge:
    @pytest.mark.unit
    def test_merge_two_rankings(self):
        """RRF correctly merges two rankings."""
        list1 = [
            {"chunk_id": "a", "text": "doc a"},
            {"chunk_id": "b", "text": "doc b"},
        ]
        list2 = [
            {"chunk_id": "b", "text": "doc b"},
            {"chunk_id": "c", "text": "doc c"},
        ]
        merged = rrf_merge(list1, list2, top_k=3, k=60)
        assert len(merged) == 3
        # "b" appears in both lists — should have highest RRF score
        assert merged[0]["chunk_id"] == "b"
        assert all("rrf_score" in r for r in merged)

    @pytest.mark.unit
    def test_dedup_by_chunk_id(self):
        """A single chunk_id is not duplicated in the result."""
        list1 = [{"chunk_id": "x", "text": "same"}]
        list2 = [{"chunk_id": "x", "text": "same"}]
        merged = rrf_merge(list1, list2, top_k=5, k=60)
        assert len(merged) == 1

    @pytest.mark.unit
    def test_respects_top_k(self):
        items = [{"chunk_id": f"c{i}", "text": f"doc {i}"} for i in range(10)]
        merged = rrf_merge(items, top_k=3, k=60)
        assert len(merged) == 3

    @pytest.mark.unit
    def test_empty_lists(self):
        merged = rrf_merge([], [], top_k=5, k=60)
        assert merged == []

    @pytest.mark.unit
    def test_uses_config_k_by_default(self):
        """When k=None, the value is taken from v7_config.RRF_K."""
        list1 = [{"chunk_id": "a", "text": "test"}]
        merged = rrf_merge(list1, top_k=5)
        assert len(merged) == 1
        # Score should be 1/(RRF_K + 1) = 1/61 ≈ 0.01639
        assert merged[0]["rrf_score"] > 0


# ─── mmr_select ────────────────────────────────────────────────────────────


class TestMMRSelect:
    @pytest.mark.unit
    def test_returns_all_if_fewer_than_top_k(self):
        passages = [{"doc_id": "d1", "score": 0.9}]
        result = mmr_select(passages, top_k=5, lambda_param=0.7)
        assert len(result) == 1

    @pytest.mark.unit
    def test_respects_top_k(self):
        passages = [{"doc_id": f"d{i}", "score": 0.9 - i * 0.1} for i in range(10)]
        result = mmr_select(passages, top_k=3, lambda_param=0.7)
        assert len(result) == 3

    @pytest.mark.unit
    def test_diversity_penalty(self):
        """Documents from different doc_ids are preferred over duplicates."""
        passages = [
            {"doc_id": "d1", "score": 0.9, "chunk_id": "c1"},
            {"doc_id": "d1", "score": 0.85, "chunk_id": "c2"},
            {"doc_id": "d2", "score": 0.80, "chunk_id": "c3"},
        ]
        result = mmr_select(passages, top_k=2, lambda_param=0.5)
        doc_ids = [r["doc_id"] for r in result]
        # With lambda=0.5 diversity penalty is significant, d2 must be selected
        assert "d2" in doc_ids


# ─── compute_keyword_overlap ──────────────────────────────────────────────


class TestComputeKeywordOverlap:
    @pytest.mark.unit
    def test_full_overlap(self):
        passages = [{"text": "Высота ограждения лестниц"}]
        score = compute_keyword_overlap("ограждение лестница", passages)
        assert score == 1.0

    @pytest.mark.unit
    def test_no_overlap(self):
        passages = [{"text": "Пожарная безопасность"}]
        score = compute_keyword_overlap("ограждение лестница", passages)
        assert score == 0.0

    @pytest.mark.unit
    def test_empty_query_returns_one(self):
        """Empty query → 1.0 (no keywords to check)."""
        score = compute_keyword_overlap("", [{"text": "test"}])
        assert score == 1.0


# ─── compute_doc_diversity ────────────────────────────────────────────────


class TestComputeDocDiversity:
    @pytest.mark.unit
    def test_single_doc(self):
        passages = [{"doc_id": "d1"}, {"doc_id": "d1"}]
        unique, ratio = compute_doc_diversity(passages)
        assert unique == 1
        assert ratio == 1.0

    @pytest.mark.unit
    def test_diverse_docs(self):
        passages = [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}]
        unique, ratio = compute_doc_diversity(passages)
        assert unique == 3
        assert abs(ratio - 1 / 3) < 0.01

    @pytest.mark.unit
    def test_empty(self):
        unique, ratio = compute_doc_diversity([])
        assert unique == 0
        assert ratio == 1.0


# ─── merge_all_passages ───────────────────────────────────────────────────


class TestMergeAllPassages:
    @pytest.mark.unit
    def test_dedup_by_chunk_id(self):
        attempts = [
            {"passages": [{"chunk_id": "c1", "score": 0.9, "doc_id": "d1"}]},
            {"passages": [{"chunk_id": "c1", "score": 0.9, "doc_id": "d1"}]},
        ]
        result = merge_all_passages(attempts, top_k=5, mmr_lambda=0.7)
        assert len(result) == 1

    @pytest.mark.unit
    def test_merges_across_attempts(self):
        attempts = [
            {"passages": [{"chunk_id": "c1", "score": 0.9, "doc_id": "d1"}]},
            {"passages": [{"chunk_id": "c2", "score": 0.8, "doc_id": "d2"}]},
        ]
        result = merge_all_passages(attempts, top_k=5, mmr_lambda=0.7)
        assert len(result) == 2

    @pytest.mark.unit
    def test_empty_attempts(self):
        assert merge_all_passages([], top_k=5) == []
