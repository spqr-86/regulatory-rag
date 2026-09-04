"""Tests for rag_simple node."""

from __future__ import annotations

import pytest

from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes.rag_simple import rag_simple


def _make_state(active_query="ограждение лестница", **overrides):
    state = {
        "query": active_query,
        "active_query": active_query,
        "plan": {
            "top_k": 5,
            "rerank": False,
            "timeout_ms": 100,
            "threshold": 0.45,
            "min_passages": 2,
            "min_keyword_overlap": 0.2,
        },
        "retrieval_id": "test_rid_001",
        "retrieval_attempts": [],
    }
    state.update(overrides)
    return state


class TestRagSimple:
    @pytest.mark.unit
    def test_returns_attempt(self):
        """With default stub (empty results), still returns attempt structure."""
        result = rag_simple(_make_state())
        attempts = result.get("retrieval_attempts", [])
        assert len(attempts) == 1
        assert attempts[0]["stage"] == "simple"
        assert attempts[0]["retrieval_id"] == "test_rid_001"
        assert "status_message" in result

    @pytest.mark.unit
    def test_dedup_skips_existing(self):
        """Skip if same retrieval_id + stage already exists."""
        existing_attempt = {
            "retrieval_id": "test_rid_001",
            "stage": "simple",
            "passages": [],
            "top_score": 0.0,
        }
        state = _make_state(retrieval_attempts=[existing_attempt])
        result = rag_simple(state)
        assert result == {}

    @pytest.mark.unit
    def test_attempt_has_metrics(self):
        result = rag_simple(_make_state())
        attempt = result["retrieval_attempts"][0]
        assert "metrics" in attempt
        assert "attempt_plan" in attempt
        assert attempt["metrics"]["retrieval_type"] == "hybrid_rrf"


class TestBm25GuaranteeIdentity:
    """The BM25 top-3 guarantee must key on passage identity, not the bare
    ``chunk_id`` — that is a per-source counter and collides across documents."""

    def setup_method(self):
        rag_simple_mod.set_vector_search(rag_simple_mod._default_vector_search)

    def teardown_method(self):
        rag_simple_mod.set_vector_search(rag_simple_mod._default_vector_search)

    @staticmethod
    def _passage(source: str, chunk_id: str, text: str, score: float) -> dict:
        return {
            "text": text,
            "score": score,
            "chunk_id": chunk_id,
            "metadata": {"source": source},
        }

    @pytest.mark.unit
    def test_bm25_chunk_survives_chunk_id_collision(self, monkeypatch):
        """A BM25 hit is not dropped because another document has the same chunk_id."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": False,
                    "V8_EXPAND_N": 0,
                    "V8_ENABLE_EVIDENCE_ASSESS": False,
                    "V8_SIMPLE_RERANK_TOP_K": 5,
                    "RRF_K": 60,
                },
            )(),
        )

        # Same bare chunk_id "50" in two different documents.
        vector_only = self._passage(
            "426-ФЗ.pdf", "50", "СОУТ проводится совместно", 0.9
        )
        shared_a = self._passage("426-ФЗ.pdf", "7", "статья 3 общие положения", 0.8)
        shared_b = self._passage("426-ФЗ.pdf", "9", "статья 4 работодатель", 0.7)
        bm25_only = self._passage(
            "29н.pdf", "50", "периодичность осмотра водителей", 0.6
        )

        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.bm25_search",
            lambda **kwargs: [shared_a, shared_b, bm25_only],
        )
        rag_simple_mod.set_vector_search(
            lambda **kwargs: [vector_only, shared_a, shared_b]
        )

        state = _make_state()
        state["plan"]["top_k"] = 3
        passages = rag_simple(state)["retrieval_attempts"][0]["passages"]

        identities = {
            f"{(p.get('metadata') or {}).get('source', '')}#{p.get('chunk_id')}"
            for p in passages
        }
        assert "426-ФЗ.pdf#50" in identities, "vector-only chunk must survive the merge"
        assert "29н.pdf#50" in identities, (
            "BM25 top-3 chunk was dropped as a duplicate of a same-numbered chunk "
            "from a different document"
        )
