"""Tests for department Q&A wiring: hybrid scoped search and structured model call."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.department_qa.contract import ModelAnswer
from src.department_qa.wiring import make_hybrid_search_fn, make_model_fn
from src.v7.scope_filter import build_scope_filters


@pytest.mark.unit
def test_hybrid_search_passes_chroma_where_and_bm25_filters():
    _, internal = build_scope_filters("unit_1")
    store = MagicMock()
    store.similarity_search_with_score.return_value = [
        (
            Document(
                page_content="вектор",
                metadata={"source": "a.md", "chunk_id": 0, "audience": "company"},
            ),
            0.2,
        )
    ]
    bm25 = MagicMock(
        return_value=[
            {
                "text": "bm25",
                "metadata": {"source": "b.md", "chunk_id": 1},
                "chunk_id": 1,
            }
        ]
    )

    hits = make_hybrid_search_fn(store, bm25)("осмотр", filters=internal, top_k=5)

    where = store.similarity_search_with_score.call_args.kwargs["filter"]
    assert "$and" in where
    assert bm25.call_args.kwargs["filters"] == internal
    assert {h["text"] for h in hits} == {"вектор", "bm25"}


def _structured(responses):
    runnable = MagicMock()
    runnable.invoke.side_effect = responses
    llm = MagicMock()
    llm.with_structured_output.return_value = runnable
    return llm, runnable


GOOD = ModelAnswer(answer="ok")


@pytest.mark.unit
def test_model_fn_uses_json_schema_structured_output():
    llm, _ = _structured([{"parsed": GOOD, "parsing_error": None, "raw": None}])
    assert make_model_fn(llm)("prompt") == GOOD
    args, kwargs = llm.with_structured_output.call_args
    assert args[0] is ModelAnswer
    assert kwargs["method"] == "json_schema" and kwargs["include_raw"] is True


@pytest.mark.unit
def test_model_fn_retries_once_with_error_text():
    err = ValueError("answer: field required")
    llm, runnable = _structured(
        [
            {"parsed": None, "parsing_error": err, "raw": None},
            {"parsed": GOOD, "parsing_error": None, "raw": None},
        ]
    )
    assert make_model_fn(llm)("prompt") == GOOD
    retry_messages = runnable.invoke.call_args_list[1].args[0]
    assert "field required" in retry_messages[-1].content


@pytest.mark.unit
def test_model_fn_raises_after_second_invalid_output():
    err = ValueError("bad")
    llm, _ = _structured([{"parsed": None, "parsing_error": err, "raw": None}] * 2)
    with pytest.raises(ValueError):
        make_model_fn(llm)("prompt")
