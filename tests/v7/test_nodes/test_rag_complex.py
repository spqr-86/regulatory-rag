"""Tests for rag_complex node — hybrid candidate pool before rerank.

Baseline 04.09.2026 показал: complex теряет 22 из 26 промахов@12 не на сортировке,
а на составе пула — золотого чанка там нет вовсе, потому что пул строился одним
плотным поиском. BM25 находит его в 19 случаях из 26. Эти тесты держат состав пула.
"""

from __future__ import annotations

import pytest

from src.v7.nodes import rag_complex as rc
from src.v7.nodes.rag_complex import rag_complex


def _passage(source: str, chunk_id: int, score: float = 0.5, **extra) -> dict:
    p = {
        "text": f"{source}#{chunk_id} текст фрагмента",
        "metadata": {"source": source, "chunk_id": chunk_id},
        "chunk_id": chunk_id,
        "score": score,
        "vector_score": score,
        "doc_id": source,
    }
    p.update(extra)
    return p


def _bm25_passage(source: str, chunk_id: int, score: float = 0.9) -> dict:
    """BM25 result shape: score squashed to [0,1], no vector_score."""
    p = _passage(source, chunk_id, score=score)
    p.pop("vector_score")
    p["bm25_score"] = 42.0
    return p


def _make_state(query="как часто проводятся медицинские осмотры", **overrides):
    state = {
        "query": query,
        "active_query": query,
        "filters": None,
        "plan": {"top_k": 12, "threshold": 0.4},
        "retrieval_id": "test_rid_complex",
        "retrieval_attempts": [],
    }
    state.update(overrides)
    return state


def _identities(passages):
    from src.v7.nlp_core import passage_identity

    return [passage_identity(p) for p in passages]


@pytest.fixture
def wired(monkeypatch):
    """Dense search without the gold chunk; BM25 that finds it."""
    dense = [_passage("29н.pdf", i, score=0.9 - i * 0.001) for i in range(60)]
    gold = _bm25_passage("29н.pdf", 777)

    monkeypatch.setattr(rc, "_vector_search", lambda **kw: list(dense))
    monkeypatch.setattr(rc, "_section_fetch_fn", None)
    monkeypatch.setattr(rc, "_rerank_fn", None)
    monkeypatch.setattr(rc, "bm25_search", lambda **kw: [gold])
    return {"dense": dense, "gold": gold, "gold_id": "29н.pdf#777"}


class TestHybridPool:
    @pytest.mark.unit
    def test_bm25_only_chunk_enters_pool(self, wired):
        """Чанк, который находит только BM25, обязан попасть в пул кандидатов."""
        result = rag_complex(_make_state())
        passages = result["retrieval_attempts"][0]["passages"]
        assert wired["gold_id"] in _identities(passages)

    @pytest.mark.unit
    def test_dense_results_survive(self, wired):
        """Гибрид не должен выбивать плотные результаты из пула."""
        result = rag_complex(_make_state())
        ids = set(_identities(result["retrieval_attempts"][0]["passages"]))
        assert {"29н.pdf#0", "29н.pdf#1", "29н.pdf#2"} <= ids

    @pytest.mark.unit
    def test_no_duplicates_by_identity(self, monkeypatch):
        """Один и тот же чанк из обоих списков схлопывается в один."""
        shared = _passage("29н.pdf", 5)
        monkeypatch.setattr(
            rc, "_vector_search", lambda **kw: [shared, _passage("29н.pdf", 6)]
        )
        monkeypatch.setattr(rc, "_section_fetch_fn", None)
        monkeypatch.setattr(rc, "_rerank_fn", None)
        monkeypatch.setattr(
            rc, "bm25_search", lambda **kw: [_bm25_passage("29н.pdf", 5)]
        )

        ids = _identities(
            rag_complex(_make_state())["retrieval_attempts"][0]["passages"]
        )
        assert len(ids) == len(set(ids))

    @pytest.mark.unit
    def test_bm25_survives_candidate_cap(self, monkeypatch):
        """Пул больше cap: отбор идёт по слитому рангу, а не по vector_score.

        BM25-кандидат не имеет vector_score; сортировка по нему выбрасывала бы
        его первым и обнуляла весь смысл гибридного пула.
        """
        dense = [_passage("тк рф.pdf", i, score=0.9 - i * 0.001) for i in range(60)]
        extra = [_passage("тк рф.pdf", 1000 + i, score=0.1) for i in range(80)]
        gold = _bm25_passage("тк рф.pdf", 777)

        monkeypatch.setattr(rc, "_vector_search", lambda **kw: list(dense))
        monkeypatch.setattr(rc, "_section_fetch_fn", lambda passages: list(extra))
        monkeypatch.setattr(rc, "_rerank_fn", None)
        monkeypatch.setattr(rc, "bm25_search", lambda **kw: [gold])

        passages = rag_complex(_make_state())["retrieval_attempts"][0]["passages"]
        from src.v7.config import v7_config

        assert len(passages) <= v7_config.RERANK_CANDIDATE_CAP
        assert "тк рф.pdf#777" in _identities(passages)

    @pytest.mark.unit
    def test_top_score_anchored_to_vector(self, monkeypatch):
        """Порог считается по плотному сходству: BM25-score его не поднимает."""
        monkeypatch.setattr(
            rc, "_vector_search", lambda **kw: [_passage("29н.pdf", 1, score=0.42)]
        )
        monkeypatch.setattr(rc, "_section_fetch_fn", None)
        monkeypatch.setattr(rc, "_rerank_fn", None)
        monkeypatch.setattr(
            rc, "bm25_search", lambda **kw: [_bm25_passage("29н.pdf", 2, score=0.99)]
        )

        attempt = rag_complex(_make_state())["retrieval_attempts"][0]
        assert attempt["top_score"] == pytest.approx(0.42)

    @pytest.mark.unit
    def test_bm25_failure_falls_back_to_dense(self, monkeypatch):
        """Упавший BM25 не роняет путь — остаётся плотный пул."""

        def _boom(**kw):
            raise RuntimeError("bm25 index not initialized")

        dense = [_passage("29н.pdf", i) for i in range(5)]
        monkeypatch.setattr(rc, "_vector_search", lambda **kw: list(dense))
        monkeypatch.setattr(rc, "_section_fetch_fn", None)
        monkeypatch.setattr(rc, "_rerank_fn", None)
        monkeypatch.setattr(rc, "bm25_search", _boom)

        passages = rag_complex(_make_state())["retrieval_attempts"][0]["passages"]
        assert len(passages) == 5

    @pytest.mark.unit
    def test_rerank_receives_hybrid_pool(self, wired):
        """Реранкер получает пул целиком, включая BM25-кандидатов."""
        seen: list[list[str]] = []

        def _rerank(query, passages, top_k):
            seen.append(_identities(passages))
            return list(passages)[:top_k]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(rc, "_rerank_fn", _rerank)
            rag_complex(_make_state())

        assert seen and wired["gold_id"] in seen[0]


class TestUnchangedContract:
    @pytest.mark.unit
    def test_dedup_skips_existing_attempt(self, wired):
        state = _make_state(
            retrieval_attempts=[
                {"retrieval_id": "test_rid_complex", "stage": "complex", "passages": []}
            ]
        )
        assert rag_complex(state) == {}

    @pytest.mark.unit
    def test_attempt_shape(self, wired):
        result = rag_complex(_make_state())
        attempt = result["retrieval_attempts"][0]
        assert attempt["stage"] == "complex"
        assert "metrics" in attempt and "attempt_plan" in attempt
        assert result["plan"]["rerank"] is True
