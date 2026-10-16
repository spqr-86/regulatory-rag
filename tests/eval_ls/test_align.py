"""Unit tests for eval/ls/align.py."""

from __future__ import annotations

import pytest

from eval.ls.align import agreement, split_rows

pytestmark = pytest.mark.unit


def test_confusion_and_rate():
    pairs = [(True, True), (True, True), (False, False), (False, True), (True, False)]
    a = agreement(pairs)
    assert a == {
        "n": 5,
        "agreement": pytest.approx(0.6),
        "tp": 2,
        "tn": 1,
        "fp": 1,
        "fn": 1,
    }


def test_empty_raises():
    with pytest.raises(ValueError):
        agreement([])


def test_split_rows_skips_missing_labels_and_lists_disagreements():
    rows = [
        ("G001", 1, 1),  # agree
        ("G052", 1, 0),  # fn: judge rejected a correct answer
        ("G028", 0, 1),  # fp: judge missed an error
        ("G010", None, 1),  # no human label → skipped
        ("G011", 1, None),  # judge abstained (needs_review, oos) → skipped
    ]
    pairs, skipped, disagreements = split_rows(rows)
    assert pairs == [(True, True), (True, False), (False, True)]
    assert skipped == 2
    assert disagreements == {"fp": ["G028"], "fn": ["G052"]}
