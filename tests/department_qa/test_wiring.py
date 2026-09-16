"""Tests for department Q&A wiring: hybrid scoped search and structured model call."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.department_qa.contract import ModelAnswer, ModelAnswerV1, VerificationResult
from src.department_qa.wiring import (
    build_mode_config,
    ensure_store_matches,
    make_hybrid_search_fn,
    make_model_fn,
)
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


CFG = SimpleNamespace(
    DEPARTMENT_V1_CHROMA_DB_PATH="./chroma_db_dept",
    DEPARTMENT_V1_COLLECTION="department_demo",
    DEPARTMENT_V2_CHROMA_DB_PATH="./chroma_db_dept_v2",
    DEPARTMENT_V2_COLLECTION="department_demo_v2",
)


@pytest.mark.unit
def test_mode_v1_bundle_without_profiles():
    with patch("src.department_qa.wiring.load_profiles") as loader:
        config = build_mode_config(
            "v1", manifest=MagicMock(), source_dir="docs", cfg=CFG
        )
    loader.assert_not_called()
    assert (config.chroma_db_path, config.collection) == (
        "./chroma_db_dept",
        "department_demo",
    )
    assert config.prompt_version == "v1"
    assert config.schema is ModelAnswerV1
    assert config.profiles == {}


@pytest.mark.unit
def test_mode_v2_bundle_loads_profiles():
    manifest = MagicMock()
    with patch(
        "src.department_qa.wiring.load_profiles", return_value={"u": "p"}
    ) as loader:
        config = build_mode_config("v2", manifest=manifest, source_dir="docs", cfg=CFG)
    loader.assert_called_once_with(manifest, "docs")
    assert (config.chroma_db_path, config.collection) == (
        "./chroma_db_dept_v2",
        "department_demo_v2",
    )
    assert config.prompt_version == "v4"
    assert config.schema is ModelAnswer
    assert config.profiles == {"u": "p"}


@pytest.mark.unit
def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_mode_config("v3", manifest=MagicMock(), source_dir="docs", cfg=CFG)


@pytest.mark.unit
@pytest.mark.parametrize(
    "path, collection",
    [
        ("./chroma_db_dept", "department_demo_v2"),
        ("./chroma_db_dept_v2", "department_demo"),
    ],
    ids=["wrong-path", "wrong-collection"],
)
def test_store_must_match_mode(path, collection):
    with patch("src.department_qa.wiring.load_profiles", return_value={}):
        config = build_mode_config(
            "v2", manifest=MagicMock(), source_dir="docs", cfg=CFG
        )
    with pytest.raises(RuntimeError, match="DEPARTMENT_QA_MODE=v2"):
        ensure_store_matches(config, path, collection)


@pytest.mark.unit
def test_model_fn_uses_requested_schema():
    llm, _ = _structured(
        [{"parsed": ModelAnswerV1(answer="ok"), "parsing_error": None, "raw": None}]
    )
    make_model_fn(llm, schema=ModelAnswerV1)("prompt")
    assert llm.with_structured_output.call_args.args[0] is ModelAnswerV1


@pytest.mark.unit
def test_model_fn_supports_verification_schema():
    verdict = VerificationResult(verdict="missing", missing_fields=["люди в здании"])
    llm, _ = _structured([{"parsed": verdict, "parsing_error": None, "raw": None}])
    assert make_model_fn(llm, schema=VerificationResult)("prompt") == verdict
    assert llm.with_structured_output.call_args.args[0] is VerificationResult


@pytest.mark.unit
def test_model_fn_records_raw_output_of_every_attempt():
    records = []
    err = ValueError("bad")
    llm, _ = _structured(
        [
            {"parsed": None, "parsing_error": err, "raw": AIMessage(content="{broken")},
            {
                "parsed": GOOD,
                "parsing_error": None,
                "raw": AIMessage(content='{"answer": "ok"}'),
            },
        ]
    )
    make_model_fn(llm, recorder=records.append)("prompt")
    assert records == [
        {"attempt": 1, "raw": "{broken", "parsing_error": "bad", "usage": None},
        {"attempt": 2, "raw": '{"answer": "ok"}', "parsing_error": None, "usage": None},
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    ["chroma_db_dept_v2", "./chroma_db_dept_v2/", os.path.abspath("chroma_db_dept_v2")],
    ids=["no-dot", "trailing-slash", "absolute"],
)
def test_store_path_spelling_does_not_matter(path):
    with patch("src.department_qa.wiring.load_profiles", return_value={}):
        config = build_mode_config(
            "v2", manifest=MagicMock(), source_dir="docs", cfg=CFG
        )
    ensure_store_matches(config, path, "department_demo_v2")


@pytest.mark.unit
def test_model_fn_records_token_usage():
    raw = AIMessage(
        content="{}",
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    llm, _ = _structured([{"parsed": GOOD, "parsing_error": None, "raw": raw}])
    log = []
    make_model_fn(llm, recorder=log.append)("prompt")
    assert log[0]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "total_tokens": 120,
    }
