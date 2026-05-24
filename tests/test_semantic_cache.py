from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.caches import BaseCache
from langchain_core.outputs import Generation

from src.infra.semantic_cache import SemanticCache


@pytest.fixture
def cache(tmp_path):
    """SemanticCache with a deterministic mocked embedder."""
    cache_file = tmp_path / "semantic_cache.json"

    # Map specific queries to fixed embeddings so similarity is predictable.
    vectors = {
        "hello world": np.array([1.0, 0.0, 0.0]),
        "hello there": np.array([0.99, 0.14, 0.0]),  # cos≈0.99 with "hello world"
        "completely different topic": np.array([0.0, 0.0, 1.0]),
    }

    def encode(text):
        return vectors.get(text, np.array([0.0, 1.0, 0.0]))

    with patch("src.infra.semantic_cache.SentenceTransformer") as mock_st:
        model = MagicMock()
        model.encode.side_effect = encode
        mock_st.return_value = model

        sc = SemanticCache(
            threshold=0.93,
            cache_file=str(cache_file),
        )
    return sc


def test_is_base_cache_instance(cache):
    assert isinstance(cache, BaseCache)


def test_update_then_lookup_semantic_hit(cache):
    gens = [Generation(text="42")]
    cache.update("hello world", "llm-a", gens)

    result = cache.lookup("hello there", "llm-a")  # semantically close
    assert result is not None
    assert len(result) == 1
    assert result[0].text == "42"


def test_lookup_miss_returns_none(cache):
    cache.update("hello world", "llm-a", [Generation(text="42")])
    assert cache.lookup("completely different topic", "llm-a") is None


def test_lookup_empty_cache_returns_none(cache):
    assert cache.lookup("anything", "llm-a") is None


def test_namespace_isolation(cache):
    """Entries under one llm_string must not be returned for another."""
    cache.update("hello world", "llm-a", [Generation(text="a-answer")])
    assert cache.lookup("hello world", "llm-b") is None
    assert cache.lookup("hello world", "llm-a")[0].text == "a-answer"


def test_update_persists_multiple_generations(cache):
    gens = [Generation(text="first"), Generation(text="second")]
    cache.update("hello world", "llm-x", gens)
    result = cache.lookup("hello world", "llm-x")
    assert [g.text for g in result] == ["first", "second"]


def test_legacy_get_add_still_works(cache):
    cache.add("hello world", "legacy-answer")
    assert cache.get("hello there") == "legacy-answer"
    assert cache.get("completely different topic") is None


def test_legacy_and_basecache_share_default_namespace(cache):
    """Legacy add writes under default ns; lookup with empty llm_string sees it."""
    cache.add("hello world", "plain-string")
    result = cache.lookup("hello world", "")
    assert result is not None
    assert result[0].text == "plain-string"


def test_clear_empties_cache(cache):
    cache.update("hello world", "ns", [Generation(text="x")])
    cache.clear()
    assert cache.lookup("hello world", "ns") is None
    assert cache.sentences == []


def test_persistence_round_trip(tmp_path):
    cache_file = tmp_path / "sc.json"
    vectors = {"q": np.array([1.0, 0.0])}

    def encode(text):
        return vectors.get(text, np.array([0.0, 1.0]))

    with patch("src.infra.semantic_cache.SentenceTransformer") as mock_st:
        model = MagicMock()
        model.encode.side_effect = encode
        mock_st.return_value = model

        sc1 = SemanticCache(threshold=0.9, cache_file=str(cache_file))
        sc1.update("q", "ns", [Generation(text="persisted")])

        # New instance loads from disk
        sc2 = SemanticCache(threshold=0.9, cache_file=str(cache_file))
        result = sc2.lookup("q", "ns")
        assert result is not None
        assert result[0].text == "persisted"
