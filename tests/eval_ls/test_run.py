"""Unit tests for eval/ls/run.py helpers — no network."""

from __future__ import annotations

import pytest

from eval.ls.run import with_current_references

pytestmark = pytest.mark.unit


def test_rejudge_scores_against_current_references():
    # LangSmith pins an existing experiment to its dataset version: a rejudge after
    # editing references saw the old ones (16.10). The wrapper swaps them in.
    seen = []

    def judge(inputs, outputs, reference_outputs):
        seen.append(reference_outputs)
        return {"key": "correctness", "score": 1}

    current = {"Вопрос?": {"ground_truth": "новый", "ground_truth_extra": "доп"}}
    (wrapped,) = with_current_references([judge], current)
    wrapped({"question": " Вопрос? "}, {"answer": "a"}, {"ground_truth": "старый"})
    assert seen == [{"ground_truth": "новый", "ground_truth_extra": "доп"}]


def test_question_missing_from_current_set_fails_loudly():
    (wrapped,) = with_current_references(
        [lambda inputs, outputs, reference_outputs: 1], {}
    )
    with pytest.raises(KeyError):
        wrapped({"question": "нет такого"}, {}, {})
