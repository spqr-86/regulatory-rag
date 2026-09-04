"""Unit tests for the synthetic-GT reviewer (eval/review_retrieval_gt.py).

Covers the deterministic, LLM-free core: verdict parsing and normalisation,
the "needs a human" rule, resume log handling, TSV row construction and the
run summary. The LLM call itself is exercised elsewhere.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.review_retrieval_gt import (
    DROP_REASONS,
    Review,
    load_reviewed,
    needs_human,
    parse_review,
    review_row,
    summarize,
)

pytestmark = pytest.mark.unit


class TestParseReview:
    def test_keep_verdict_carries_no_reason(self):
        parsed = parse_review(Review(verdict="keep", reason="none", confidence=0.9))
        assert parsed["verdict"] == "keep"
        assert parsed["reason"] == ""

    def test_drop_keeps_known_reason(self):
        parsed = parse_review(
            Review(verdict="drop", reason="answer_in_question", confidence=0.8)
        )
        assert parsed["verdict"] == "drop"
        assert parsed["reason"] == "answer_in_question"

    def test_unknown_reason_on_drop_becomes_other(self):
        parsed = parse_review(
            Review(verdict="drop", reason="плохой вопрос", confidence=0.7)
        )
        assert parsed["reason"] == "other"

    def test_confidence_clamped_to_unit_range(self):
        assert (
            parse_review(Review(verdict="keep", reason="none", confidence=7.0))[
                "confidence"
            ]
            == 1.0
        )
        assert (
            parse_review(Review(verdict="drop", reason="other", confidence=-1.0))[
                "confidence"
            ]
            == 0.0
        )


class TestNeedsHuman:
    def test_low_confidence_goes_to_human(self):
        assert (
            needs_human({"verdict": "keep", "confidence": 0.4}, threshold=0.75) is True
        )

    def test_confident_keep_does_not(self):
        assert (
            needs_human({"verdict": "keep", "confidence": 0.95}, threshold=0.75)
            is False
        )

    def test_every_drop_goes_to_human_however_confident(self):
        assert (
            needs_human({"verdict": "drop", "confidence": 1.0}, threshold=0.75) is True
        )


class TestLoadReviewed:
    def test_missing_log_is_empty(self, tmp_path):
        assert load_reviewed(tmp_path / "nope.jsonl") == {}

    def test_reads_verdicts_by_question_key(self, tmp_path):
        log = tmp_path / "raw.jsonl"
        log.write_text(
            json.dumps(
                {"n": 1, "verdict": "keep", "confidence": 0.9}, ensure_ascii=False
            )
            + "\n"
            + json.dumps(
                {"n": 2, "verdict": "drop", "confidence": 0.8}, ensure_ascii=False
            )
            + "\n",
            encoding="utf-8",
        )
        reviewed = load_reviewed(log)
        assert set(reviewed) == {1, 2}
        assert reviewed[2]["verdict"] == "drop"

    def test_truncated_last_line_is_skipped(self, tmp_path):
        log = tmp_path / "raw.jsonl"
        log.write_text(
            json.dumps({"n": 1, "verdict": "keep", "confidence": 0.9}) + '\n{"n": 2,',
            encoding="utf-8",
        )
        assert set(load_reviewed(log)) == {1}


class TestReviewRow:
    def test_row_matches_review_tsv_columns(self):
        record = {
            "question": "Сколько огнетушителей?",
            "source": "1479 ппр.pdf",
            "chunk_id": "1479 ппр.pdf#608",
            "chunk_preview": "Заправочный островок ...",
        }
        review = {"verdict": "drop", "reason": "answer_in_question", "confidence": 0.8}
        row = review_row(4, record, review)
        assert row["n"] == 4
        assert row["verdict"] == "drop"
        assert row["note"] == "answer_in_question (0.80)"
        assert row["question"] == record["question"]
        assert row["chunk_id"] == record["chunk_id"]

    def test_keep_note_records_confidence_only(self):
        row = review_row(
            1,
            {"question": "q", "source": "s", "chunk_id": "c", "chunk_preview": "p"},
            {"verdict": "keep", "reason": "", "confidence": 0.93},
        )
        assert row["note"] == "(0.93)"


class TestSummarize:
    def test_counts_verdicts_reasons_and_human_queue(self):
        reviews = [
            {"verdict": "keep", "reason": "", "confidence": 0.95},
            {"verdict": "keep", "reason": "", "confidence": 0.5},
            {"verdict": "drop", "reason": "answer_in_question", "confidence": 0.9},
            {"verdict": "drop", "reason": "not_answerable", "confidence": 0.9},
            {"verdict": "drop", "reason": "answer_in_question", "confidence": 0.9},
        ]
        stats = summarize(reviews, threshold=0.75)
        assert stats["total"] == 5
        assert stats["keep"] == 2
        assert stats["drop"] == 3
        assert stats["needs_human"] == 4
        assert stats["reasons"]["answer_in_question"] == 2

    def test_all_drop_reasons_are_known(self):
        assert "answer_in_question" in DROP_REASONS
        assert "other" in DROP_REASONS


class TestAttachChunkTexts:
    def test_full_text_replaces_the_preview(self):
        from eval.review_retrieval_gt import attach_chunk_texts

        records = [
            {"question": "q", "chunk_id": "ппр.pdf#7", "chunk_preview": "обрезок"}
        ]
        corpus = {"ппр.pdf#7": "полный текст нормы, который длиннее превью"}
        attached = attach_chunk_texts(records, corpus)
        assert attached[0]["chunk_text"] == corpus["ппр.pdf#7"]

    def test_missing_chunk_falls_back_to_preview(self):
        from eval.review_retrieval_gt import attach_chunk_texts

        records = [
            {"question": "q", "chunk_id": "нет.pdf#1", "chunk_preview": "обрезок"}
        ]
        attached = attach_chunk_texts(records, {})
        assert attached[0]["chunk_text"] == "обрезок"
        assert attached[0]["chunk_missing"] is True


class TestValidateAnswerSpan:
    def test_keep_with_a_verbatim_span_survives(self):
        from eval.review_retrieval_gt import validate_answer_span

        chunk = "Заправочный островок оснащается не менее чем 2 огнетушителями."
        review = {
            "verdict": "keep",
            "reason": "",
            "confidence": 0.9,
            "note": "",
            "answer_span": "не менее чем 2 огнетушителями",
        }
        checked = validate_answer_span(review, chunk)
        assert checked["verdict"] == "keep"
        assert checked["confidence"] == 0.9

    def test_span_matched_across_whitespace_differences(self):
        from eval.review_retrieval_gt import validate_answer_span

        chunk = "Покрывала хранятся\n   в водонепроницаемых\nфутлярах."
        review = {
            "verdict": "keep",
            "reason": "",
            "confidence": 0.9,
            "note": "",
            "answer_span": "хранятся в водонепроницаемых футлярах",
        }
        assert validate_answer_span(review, chunk)["confidence"] == 0.9

    def test_invented_span_drops_confidence_and_is_flagged(self):
        from eval.review_retrieval_gt import validate_answer_span

        review = {
            "verdict": "keep",
            "reason": "",
            "confidence": 0.95,
            "note": "",
            "answer_span": "срок хранения 5 лет",
        }
        checked = validate_answer_span(review, "Покрывала хранятся в футлярах.")
        assert checked["confidence"] < 0.75
        assert "span_not_found" in checked["note"]

    def test_drop_verdict_is_left_alone(self):
        from eval.review_retrieval_gt import validate_answer_span

        review = {
            "verdict": "drop",
            "reason": "not_answerable",
            "confidence": 0.9,
            "note": "",
            "answer_span": "",
        }
        assert validate_answer_span(review, "любой текст") == review


class TestMergeArbitration:
    def test_two_drops_confirm_the_reject(self):
        from eval.review_retrieval_gt import merge_arbitration

        merged = merge_arbitration(
            {
                "verdict": "drop",
                "reason": "not_answerable",
                "confidence": 0.9,
                "note": "",
            },
            {
                "verdict": "drop",
                "reason": "not_answerable",
                "confidence": 0.8,
                "note": "",
            },
        )
        assert merged["status"] == "drop_confirmed"
        assert merged["verdict"] == "drop"

    def test_arbiter_keep_makes_it_disputed(self):
        from eval.review_retrieval_gt import merge_arbitration

        merged = merge_arbitration(
            {"verdict": "drop", "reason": "too_generic", "confidence": 0.9, "note": ""},
            {
                "verdict": "keep",
                "reason": "",
                "confidence": 0.9,
                "note": "ответ в тексте",
            },
        )
        assert merged["status"] == "disputed"
        assert merged["note"].startswith("спор:")

    def test_untouched_review_stays_as_is(self):
        from eval.review_retrieval_gt import merge_arbitration

        review = {"verdict": "keep", "reason": "", "confidence": 0.9, "note": ""}
        merged = merge_arbitration(review, None)
        assert merged["status"] == "single_pass"
        assert merged["verdict"] == "keep"


class TestBuildReviewedGt:
    def test_only_confirmed_keeps_survive(self):
        from eval.review_retrieval_gt import build_reviewed_gt

        records = [
            {"question": "a", "chunk_id": "c#1"},
            {"question": "b", "chunk_id": "c#2"},
            {"question": "c", "chunk_id": "c#3"},
        ]
        merged = {
            1: {
                "verdict": "keep",
                "status": "single_pass",
                "confidence": 0.9,
                "reason": "",
            },
            2: {
                "verdict": "drop",
                "status": "drop_confirmed",
                "confidence": 0.9,
                "reason": "not_answerable",
            },
            3: {
                "verdict": "drop",
                "status": "disputed",
                "confidence": 0.9,
                "reason": "too_generic",
            },
        }
        kept = build_reviewed_gt(records, merged)
        assert [r["question"] for r in kept] == ["a"]

    def test_low_confidence_keep_is_excluded(self):
        from eval.review_retrieval_gt import build_reviewed_gt

        records = [{"question": "a", "chunk_id": "c#1"}]
        merged = {
            1: {
                "verdict": "keep",
                "status": "single_pass",
                "confidence": 0.4,
                "reason": "",
            }
        }
        assert build_reviewed_gt(records, merged) == []

    def test_record_keeps_only_gt_fields(self):
        from eval.review_retrieval_gt import build_reviewed_gt

        records = [
            {
                "question": "a",
                "chunk_id": "c#1",
                "source": "s",
                "chunk_preview": "p",
                "chunk_text": "полный",
                "chunk_missing": False,
            }
        ]
        merged = {
            1: {
                "verdict": "keep",
                "status": "single_pass",
                "confidence": 0.9,
                "reason": "",
            }
        }
        kept = build_reviewed_gt(records, merged)
        assert set(kept[0]) == {"question", "chunk_id", "source", "chunk_preview"}
