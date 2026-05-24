"""Unit tests for v7 bridge.make_rerank_fn (CARD-5.3).

Verifies the thin adapter over langchain's FlashrankRerank:
- vector_score is preserved on output (frozen pre-rerank)
- top_n is respected
- standard FlashrankRerank wiring is used (Ranker injected as client)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_ranker():
    """Ranker mock whose rerank() returns deterministic scored passages."""
    from flashrank import Ranker

    ranker = MagicMock(spec=Ranker)

    def _rerank(request):
        # FlashRank returns list of {id, score, text, meta} sorted by score desc.
        # Mirror input passages but reverse order to prove the adapter respects
        # rerank-imposed ordering, not original passage order.
        results = []
        for p in reversed(request.passages):
            results.append(
                {
                    "id": p["id"],
                    "score": 0.1 * (p["id"] + 1),
                    "text": p["text"],
                    "meta": p.get("meta", {}),
                }
            )
        return results

    ranker.rerank.side_effect = _rerank
    return ranker


def test_make_rerank_fn_preserves_vector_score(mock_ranker):
    from src.v7 import bridge

    with patch("flashrank.Ranker", return_value=mock_ranker):
        rerank = bridge.make_rerank_fn(model_name="test-model", cache_dir="/tmp/x")

    passages = [
        {"text": "alpha", "metadata": {"source": "A"}, "score": 0.91},
        {"text": "beta", "metadata": {"source": "B"}, "score": 0.42},
        {"text": "gamma", "metadata": {"source": "C"}, "score": 0.77},
    ]
    result = rerank("query", passages, top_k=3)

    assert len(result) == 3
    # vector_score must be the ORIGINAL pre-rerank score, never overwritten
    by_text = {r["text"]: r for r in result}
    assert by_text["alpha"]["vector_score"] == 0.91
    assert by_text["beta"]["vector_score"] == 0.42
    assert by_text["gamma"]["vector_score"] == 0.77
    # score field carries the (rounded) FlashRank relevance score
    for r in result:
        assert isinstance(r["score"], float)
        assert r["score"] != r["vector_score"]
    # metadata is preserved (and internal helper key stripped)
    assert by_text["alpha"]["metadata"]["source"] == "A"
    assert "_v7_passage_idx" not in by_text["alpha"]["metadata"]
    assert "relevance_score" not in by_text["alpha"]["metadata"]


def test_make_rerank_fn_respects_top_k(mock_ranker):
    from src.v7 import bridge

    with patch("flashrank.Ranker", return_value=mock_ranker):
        rerank = bridge.make_rerank_fn()

    passages = [{"text": f"p{i}", "metadata": {}, "score": 0.5} for i in range(10)]
    result = rerank("q", passages, top_k=3)
    assert len(result) == 3


def test_make_rerank_fn_empty_passages(mock_ranker):
    from src.v7 import bridge

    with patch("flashrank.Ranker", return_value=mock_ranker):
        rerank = bridge.make_rerank_fn()
    assert rerank("q", [], top_k=5) == []
    mock_ranker.rerank.assert_not_called()


def test_make_rerank_fn_uses_langchain_flashrank(mock_ranker):
    """Adapter must delegate to langchain's FlashrankRerank, not roll its own."""
    from src.v7 import bridge

    with (
        patch("flashrank.Ranker", return_value=mock_ranker),
        patch("langchain_community.document_compressors.FlashrankRerank") as mock_cls,
    ):
        instance = MagicMock()
        instance.compress_documents.return_value = []
        mock_cls.return_value = instance

        rerank = bridge.make_rerank_fn()
        rerank("query", [{"text": "x", "metadata": {}, "score": 0.5}], top_k=2)

        mock_cls.assert_called_once()
        kwargs = mock_cls.call_args.kwargs
        assert kwargs.get("client") is mock_ranker
        assert kwargs.get("top_n") == 2
        instance.compress_documents.assert_called_once()
