"""Tests for rag_simple multi-query expand (V8 Epic 3)."""

from __future__ import annotations

import pytest

from src.v7.nodes import rag_simple as rag_simple_mod
from src.v7.nodes.rag_simple import rag_simple

# ─── Helpers ──────────────────────────────────────────────────────────────


def _make_passage(text: str, score: float = 0.5, chunk_id: str | None = None) -> dict:
    return {
        "text": text,
        "score": score,
        "chunk_id": chunk_id or text[:20].replace(" ", "_"),
        "metadata": {},
    }


def _make_state(active_query: str = "требования к ограждениям", **overrides) -> dict:
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
        "retrieval_id": "test_rid_multi",
        "retrieval_attempts": [],
    }
    state.update(overrides)
    return state


def _restore_rag_simple():
    """Reset module-level injected functions to defaults."""
    rag_simple_mod.set_vector_search(rag_simple_mod._default_vector_search)
    rag_simple_mod.set_expand_fn(None)


# ─── Tests ────────────────────────────────────────────────────────────────


class TestMultiQueryExpand:
    def setup_method(self):
        _restore_rag_simple()

    def teardown_method(self):
        _restore_rag_simple()

    @pytest.mark.unit
    def test_expand_fn_not_called_when_flag_disabled(self, monkeypatch):
        """When V8_ENABLE_MULTI_QUERY=False, expand_fn must not be called."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": False,
                    "V8_EXPAND_N": 3,
                },
            )(),
        )

        calls = []

        def _expand(query: str, n: int) -> list[str]:
            calls.append(query)
            return ["alt query 1", "alt query 2"]

        rag_simple_mod.set_expand_fn(_expand)
        rag_simple(_make_state())
        assert calls == [], "expand_fn must not be called when flag is disabled"

    @pytest.mark.unit
    def test_expand_fn_called_with_active_query(self, monkeypatch):
        """When enabled, expand_fn is called with active_query and n=V8_EXPAND_N."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": True,
                    "V8_EXPAND_N": 3,
                },
            )(),
        )

        calls = []

        def _expand(query: str, n: int) -> list[str]:
            calls.append((query, n))
            return []

        rag_simple_mod.set_expand_fn(_expand)
        rag_simple(_make_state(active_query="высота ограждения лестницы"))
        assert len(calls) == 1
        assert calls[0] == ("высота ограждения лестницы", 3)

    @pytest.mark.unit
    def test_multi_query_merges_passages_from_all_queries(self, monkeypatch):
        """Passages from all expanded queries are merged via RRF."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": True,
                    "V8_EXPAND_N": 2,
                    "RRF_K": 60,
                },
            )(),
        )

        # Original query → passage A
        # Expanded query → passage B (unique, not found by original)
        passage_a = _make_passage(
            "ограждение лестницы требования ГОСТ", score=0.6, chunk_id="chunk_a"
        )
        passage_b = _make_passage(
            "защитный барьер высота нормы", score=0.5, chunk_id="chunk_b"
        )

        search_calls = []

        def _vector_search(
            query: str, filters=None, top_k: int = 5, **kwargs
        ) -> list[dict]:
            search_calls.append(query)
            if "ограждение" in query:
                return [passage_a]
            if "барьер" in query:
                return [passage_b]
            return []

        def _expand(query: str, n: int) -> list[str]:
            return ["защитный барьер высота нормы"]

        rag_simple_mod.set_vector_search(_vector_search)
        rag_simple_mod.set_expand_fn(_expand)

        result = rag_simple(_make_state(active_query="ограждение лестница высота"))
        attempts = result["retrieval_attempts"]
        assert len(attempts) == 1

        texts = {p["text"] for p in attempts[0]["passages"]}
        assert passage_a["text"] in texts, "Original query passage must be included"
        assert passage_b["text"] in texts, "Expanded query passage must be included"

    @pytest.mark.unit
    def test_multi_query_deduplicates_passages(self, monkeypatch):
        """Same passage found by multiple queries must appear only once."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": True,
                    "V8_EXPAND_N": 2,
                    "RRF_K": 60,
                },
            )(),
        )

        shared = _make_passage(
            "общий фрагмент про охрану труда", score=0.7, chunk_id="shared_chunk"
        )

        def _vector_search(
            query: str, filters=None, top_k: int = 5, **kwargs
        ) -> list[dict]:
            return [shared]

        def _expand(query: str, n: int) -> list[str]:
            return ["другая формулировка вопроса"]

        rag_simple_mod.set_vector_search(_vector_search)
        rag_simple_mod.set_expand_fn(_expand)

        result = rag_simple(_make_state())
        passages = result["retrieval_attempts"][0]["passages"]
        chunk_ids = [p.get("chunk_id") for p in passages]
        assert (
            chunk_ids.count("shared_chunk") == 1
        ), "Duplicate passages must be deduped"

    @pytest.mark.unit
    def test_expand_fn_failure_falls_back_to_single_query(self, monkeypatch):
        """If expand_fn raises, gracefully fall back to single-query mode."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": True,
                    "V8_EXPAND_N": 3,
                    "RRF_K": 60,
                },
            )(),
        )

        passage = _make_passage("ограждение лестниц", score=0.6, chunk_id="chunk_x")

        def _vector_search(
            query: str, filters=None, top_k: int = 5, **kwargs
        ) -> list[dict]:
            return [passage]

        def _expand_broken(query: str, n: int) -> list[str]:
            raise RuntimeError("LLM expand failed")

        rag_simple_mod.set_vector_search(_vector_search)
        rag_simple_mod.set_expand_fn(_expand_broken)

        # Must not raise, must return a valid attempt
        result = rag_simple(_make_state())
        assert "retrieval_attempts" in result
        assert len(result["retrieval_attempts"]) == 1

    @pytest.mark.unit
    def test_top_score_from_original_query_vector_results(self, monkeypatch):
        """top_score must reflect only the original query's vector results.

        Expanded queries may find low-score passages. top_score drives the
        hard gate threshold — must stay anchored to the primary query.
        """
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": True,
                    "V8_EXPAND_N": 2,
                    "RRF_K": 60,
                },
            )(),
        )

        original_passage = _make_passage(
            "прямой ответ на вопрос", score=0.72, chunk_id="orig"
        )
        expanded_passage = _make_passage(
            "косвенный фрагмент низкий score", score=0.1, chunk_id="exp"
        )

        call_count = [0]

        def _vector_search(
            query: str, filters=None, top_k: int = 5, **kwargs
        ) -> list[dict]:
            call_count[0] += 1
            if call_count[0] == 1:
                return [original_passage]  # original query
            return [expanded_passage]  # expanded query

        def _expand(query: str, n: int) -> list[str]:
            return ["косвенная формулировка"]

        rag_simple_mod.set_vector_search(_vector_search)
        rag_simple_mod.set_expand_fn(_expand)

        result = rag_simple(_make_state())
        attempt = result["retrieval_attempts"][0]
        assert (
            abs(attempt["top_score"] - 0.72) < 0.01
        ), f"top_score must be from original query (0.72), got {attempt['top_score']}"

    @pytest.mark.unit
    def test_no_expand_fn_injected_single_query_mode(self, monkeypatch):
        """Without expand_fn, multi-query flag has no effect — single query mode."""
        monkeypatch.setattr(
            "src.v7.nodes.rag_simple.v7_config",
            type(
                "cfg",
                (),
                {
                    "V8_ENABLE_MULTI_QUERY": True,
                    "V8_EXPAND_N": 3,
                    "RRF_K": 60,
                },
            )(),
        )

        # No expand_fn injected (None by default after restore)
        result = rag_simple(_make_state())
        assert "retrieval_attempts" in result
