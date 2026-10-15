"""Offline checks for the current citation scoring contract."""

import copy

import pytest

from eval.judges.citations import (
    JudgeFormatError,
    contract_violation,
    is_complete,
    validate_and_score,
)

# ANCHOR: Tests quote evidence and status aggregation without model or network calls.
# Inputs are synthetic parser/scoring examples; they do not measure judge quality.
RESPONSE = "Документ 1479, пункт 30."
CONTEXT = "Документ 1479, раздел I, пункт 31."


def make_check(
    comparison, answer_quote="пункт 30", context_quote="пункт 31", field="provision"
):
    return {
        "field": field,
        "answer_quote": answer_quote,
        "context_quote": context_quote,
        "comparison": comparison,
        "reason": "Synthetic scoring fixture",
    }


def score(checks):
    return validate_and_score({"checks": checks}, RESPONSE, CONTEXT)


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ([make_check("match", "Документ 1479", "Документ 1479")], "pass"),
        ([make_check("mismatch")], "fail"),
        # Decision 2026-10-15: an unnamed provision is incompleteness, not an
        # error; with nothing matched there is nothing confirmed.
        ([make_check("missing_in_answer", None)], "needs_review"),
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
    with pytest.raises(JudgeFormatError):
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
    with pytest.raises(ValueError, match=f"{comparison} requires both quotes"):
        score([make_check(comparison, answer_quote, context_quote)])


@pytest.mark.parametrize(
    ("answer_quote", "context_quote"),
    [(None, None), ("пункт 30", None)],
)
def test_missing_in_answer_requires_evidence_of_omission(answer_quote, context_quote):
    with pytest.raises(ValueError, match="missing_in_answer requires a context_quote"):
        score([make_check("missing_in_answer", answer_quote, context_quote)])


# Shape of the G001 judge reply (2026-10-12): missing_in_answer with a quote
# copied from the answer. The judge contradicts itself, so a human decides.
def test_missing_in_answer_with_answer_quote_needs_review():
    check = make_check("missing_in_answer", "пункт 30", "раздел I")
    assert score([make_check("match", "Документ 1479", "Документ 1479"), check]) == (
        "needs_review"
    )
    assert contract_violation(check) == "missing_in_answer with answer_quote"


# Decision 2026-10-12: a reference is usable with document and provision;
# a missing section stays in checks but does not change the status (G006).
def test_missing_section_does_not_fail():
    section = make_check("missing_in_answer", None, "раздел I", field="section")
    assert score([make_check("match", "Документ 1479", "Документ 1479"), section]) == (
        "pass"
    )


def test_missing_section_does_not_hide_other_statuses():
    section = make_check("missing_in_answer", None, "раздел I", field="section")
    assert score([section, make_check("mismatch")]) == "fail"
    assert score([section, make_check("unverifiable", context_quote=None)]) == (
        "needs_review"
    )


def test_missing_section_with_answer_quote_is_flagged_but_passes():
    section = make_check("missing_in_answer", "пункт 30", "раздел I", field="section")
    assert score([make_check("match", "Документ 1479", "Документ 1479"), section]) == (
        "pass"
    )
    assert contract_violation(section) == "missing_in_answer with answer_quote"


# Laws: the article finds the norm, a part is optional like a section;
# a wrong part still fails.
LAW_RESPONSE = "по ст. 13 426-ФЗ, часть 4"
LAW_CONTEXT = "426-ФЗ. Статья 13\n3. Работодатель обязан"


def law_score(checks):
    return validate_and_score({"checks": checks}, LAW_RESPONSE, LAW_CONTEXT)


def test_missing_part_does_not_fail():
    article = make_check("match", "ст. 13", "Статья 13")
    part = make_check("missing_in_answer", None, "3. Работодатель", field="part")
    assert law_score([article, part]) == "pass"


def test_missing_part_with_answer_quote_is_flagged_but_passes():
    article = make_check("match", "ст. 13", "Статья 13")
    part = make_check("missing_in_answer", "ст. 13", "3. Работодатель", field="part")
    assert law_score([article, part]) == "pass"
    assert contract_violation(part) == "missing_in_answer with answer_quote"


def test_wrong_part_fails():
    article = make_check("match", "ст. 13", "Статья 13")
    part = make_check("mismatch", "часть 4", "3. Работодатель", field="part")
    assert law_score([article, part]) == "fail"


# Decision 2026-10-15 (human labels on v8-judges-84daf785): the "citations"
# status is accuracy only; an unnamed document or provision goes to is_complete.
def test_missing_provision_does_not_fail_accuracy():
    provision = make_check("missing_in_answer", None, "пункт 31")
    assert score([make_check("match", "Документ 1479", "Документ 1479"), provision]) == (
        "pass"
    )


def test_missing_article_alone_needs_review():
    article = make_check("missing_in_answer", None, "Статья 13")
    assert law_score([article]) == "needs_review"


def test_missing_provision_does_not_hide_mismatch():
    provision = make_check("missing_in_answer", None, "пункт 31")
    assert score([provision, make_check("mismatch")]) == "fail"


@pytest.mark.parametrize(
    ("checks", "expected"),
    [
        ([make_check("match", "Документ 1479", "Документ 1479")], True),
        ([make_check("missing_in_answer", None, "пункт 31")], False),
        (
            [make_check("missing_in_answer", None, "Документ 1479", field="document")],
            False,
        ),
        # Section and law part stay optional for completeness too (decision 12.10).
        ([make_check("missing_in_answer", None, "раздел I", field="section")], True),
        ([make_check("mismatch")], True),
        ([], True),
    ],
)
def test_is_complete(checks, expected):
    assert is_complete({"checks": checks}) is expected


def test_contract_violation_does_not_hide_fail():
    violation = make_check("missing_in_answer", "пункт 30", "раздел I")
    assert score([make_check("mismatch"), violation]) == "fail"
    assert score([violation, make_check("mismatch")]) == "fail"


def _check(answer_quote, context_quote):
    return {
        "checks": [
            {
                "field": "provision",
                "answer_quote": answer_quote,
                "context_quote": context_quote,
                "comparison": "match",
                "reason": "r",
            }
        ]
    }


def test_quote_matches_across_line_break_in_context():
    verdict = _check("осмотров определяется", "осмотров определяется")
    context = "периодических осмотров\nопределяется типами"
    assert validate_and_score(verdict, "осмотров определяется", context) == "pass"


def test_quote_matches_ignoring_case():
    verdict = _check("пункт 9", "пункт 9")
    assert validate_and_score(verdict, "Пункт 9 гласит", "Пункт 9") == "pass"


def test_invented_quote_still_rejected_after_normalization():
    verdict = _check("пункт 10", "пункт 9")
    with pytest.raises(ValueError, match="answer_quote must appear in response"):
        validate_and_score(verdict, "Пункт 9 гласит", "Пункт 9")


def test_scoring_does_not_change_the_verdict():
    """Scoring only reads; the contract_violation mark is added by judge_citations."""
    verdict = {"checks": [make_check("missing_in_answer", "пункт 30", "раздел I")]}
    before = copy.deepcopy(verdict)
    validate_and_score(verdict, RESPONSE, CONTEXT)
    assert verdict == before


def test_no_contract_violation_for_a_consistent_check():
    assert contract_violation(make_check("missing_in_answer", None, "раздел I")) is None
