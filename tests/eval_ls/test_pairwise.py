"""Unit tests for eval/ls/pairwise.py."""

from __future__ import annotations

import pytest

from eval.ls.pairwise import parse_preference

pytestmark = pytest.mark.unit

FENCE = "`" * 3


@pytest.mark.parametrize(
    "raw, want",
    [
        ('{"winner": "A"}', 0),
        ('{"winner": "b"}', 1),
        ('{"winner": "TIE"}', -1),
        (FENCE + 'json\n{"winner": "A"}\n' + FENCE, 0),
    ],
)
def test_parse(raw, want):
    assert parse_preference(raw) == want


@pytest.mark.parametrize("raw", ['{"winner": "C"}', "{}", "A"])
def test_rejects(raw):
    with pytest.raises(ValueError):
        parse_preference(raw)


@pytest.mark.parametrize(
    "winner,scores",
    [
        ("A", {"run-a": 1, "run-b": 0}),
        ("B", {"run-a": 0, "run-b": 1}),
        ("TIE", {"run-a": 0.5, "run-b": 0.5}),
    ],
)
def test_evaluator_scores_by_run_id(winner, scores):
    from types import SimpleNamespace

    from eval.ls.pairwise import make_pairwise_evaluator

    llm = FakeLLM('{"winner": "' + winner + '"}')
    runs = [
        SimpleNamespace(id=rid, outputs={"answer": answer}, error=None)
        for rid, answer in [("run-a", "Ответ А"), ("run-b", "Ответ Б")]
    ]
    example = SimpleNamespace(
        inputs={"question": "Вопрос"},
        outputs={
            "ground_truth": "Эталон",
            "ground_truth_extra": "Дополнение",
            "forbidden_claims": "Запрет",
            "legal_as_of": "2026-10-16",
        },
    )
    result = make_pairwise_evaluator(llm)(runs, example)
    assert result["key"] == "preference"
    assert result["scores"] == scores
    prompt = "\n".join(m.content for m in llm.messages)
    for value in [
        "Вопрос",
        "Ответ А",
        "Ответ Б",
        "Эталон",
        "Дополнение",
        "Запрет",
        "2026-10-16",
    ]:
        assert value in prompt


class FakeLLM:
    def __init__(self, raw, metadata=None):
        self.raw, self.metadata = raw, metadata or {}

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.messages = messages
        return AIMessage(content=self.raw, response_metadata=self.metadata)


@pytest.mark.parametrize(
    "raw", ["[]", "null", '{"winner": 1}', '{"winner": null}', "oops"]
)
def test_bad_format_is_unscored(raw):
    from types import SimpleNamespace

    from eval.ls.pairwise import make_pairwise_evaluator

    runs = [
        SimpleNamespace(id=rid, outputs={"answer": "a"}, error=None)
        for rid in ("a", "b")
    ]
    example = SimpleNamespace(inputs={"question": "q"}, outputs={})
    result = make_pairwise_evaluator(FakeLLM(raw))(runs, example)
    assert result["scores"] == {}
    assert "judge_format_error" in result["comment"]


def test_truncated_reply_is_unscored():
    from types import SimpleNamespace

    from eval.ls.pairwise import make_pairwise_evaluator

    runs = [
        SimpleNamespace(id=rid, outputs={"answer": "a"}, error=None)
        for rid in ("a", "b")
    ]
    example = SimpleNamespace(inputs={"question": "q"}, outputs={})
    result = make_pairwise_evaluator(
        FakeLLM('{"winner": "A"}', {"finish_reason": "length"})
    )(runs, example)
    assert result["scores"] == {}
    assert "truncated" in result["comment"]


def test_failed_target_skips_judge():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from eval.ls.pairwise import make_pairwise_evaluator

    llm = MagicMock()
    runs = [
        SimpleNamespace(id="a", outputs=None, error="failed"),
        SimpleNamespace(id="b", outputs={"answer": "a"}, error=None),
    ]
    result = make_pairwise_evaluator(llm)(runs, None)
    assert result["scores"] == {}
    assert result["comment"] == "target_error"
    llm.invoke.assert_not_called()
