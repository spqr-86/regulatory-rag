"""Offline checks for the current citation scoring contract."""

import pytest

from eval.judges.citations import validate_and_score

# ANCHOR: Tests quote evidence and status aggregation without model or network calls.
# Inputs are synthetic parser/scoring examples; they do not measure judge quality.
RESPONSE = "Документ 1479, пункт 30."
CONTEXT = "Документ 1479, раздел I, пункт 31."


def make_check(comparison, answer_quote="пункт 30", context_quote="пункт 31"):
    return {
        "field": "provision",
        "answer_quote": answer_quote,
        "context_quote": context_quote,
        "comparison": comparison,
        "reason": "Synthetic scoring fixture",
    }


def score(checks):
    return validate_and_score({"checks": checks}, "Вопрос", RESPONSE, CONTEXT)


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ([make_check("match", "Документ 1479", "Документ 1479")], "pass"),
        ([make_check("mismatch")], "fail"),
        ([make_check("missing_in_answer", None)], "fail"),
        ([make_check("unverifiable", context_quote=None)], "needs_review"),
        ([make_check("unverifiable", None, None)], "needs_review"),
        ([], "needs_review"),
        (
            [make_check("mismatch"), make_check("unverifiable", context_quote=None)],
            "fail",
        ),
        (
            [make_check("unverifiable", context_quote=None), make_check("mismatch")],
            "fail",
        ),
    ],
)
def test_status_aggregation(checks, expected):
    assert score(checks) == expected


@pytest.mark.parametrize(
    "checks",
    [
        [make_check("match", "пункт 99")],
        [make_check("match", context_quote="пункт 99")],
        [make_check("unknown")],
        [make_check("mismatch"), make_check("unknown")],
        [make_check("mismatch"), make_check("match", "пункт 99")],
    ],
)
def test_invalid_evidence_or_comparison_raises(checks):
    with pytest.raises(ValueError):
        score(checks)


@pytest.mark.parametrize("quote", ["", "   ", 9, ["пункт 30"]])
@pytest.mark.parametrize("field", ["answer_quote", "context_quote"])
def test_quote_must_be_a_nonempty_string_or_null(quote, field):
    check = make_check("unverifiable")
    check[field] = quote
    with pytest.raises(ValueError):
        score([check])


@pytest.mark.parametrize("comparison", ["match", "mismatch"])
@pytest.mark.parametrize(
    ("answer_quote", "context_quote"),
    [(None, None), ("пункт 30", None), (None, "пункт 31")],
)
def test_match_and_mismatch_require_both_quotes(
    comparison, answer_quote, context_quote
):
    with pytest.raises(ValueError, match="require both quotes"):
        score([make_check(comparison, answer_quote, context_quote)])


@pytest.mark.parametrize(
    ("answer_quote", "context_quote"),
    [(None, None), ("пункт 30", "пункт 31"), ("пункт 30", None)],
)
def test_missing_in_answer_requires_evidence_of_omission(answer_quote, context_quote):
    with pytest.raises(ValueError, match="requires a null answer_quote"):
        score([make_check("missing_in_answer", answer_quote, context_quote)])
