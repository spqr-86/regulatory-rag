"""Unit tests for the held-out labelling tool (eval/label_golden_retrieval.py).

Covers the LLM-free, Chroma-free core: loading the hand-written golden set,
merging candidate pools from several retrieval paths, prompt rendering, verdict
parsing, quote validation, the two-pass merge, the human queue (disputes plus a
control sample of agreements) and the labelled output the metric runner reads.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.label_golden_retrieval import (
    apply_arbitration,
    build_labeled_gt,
    build_review_rows,
    human_priority,
    format_candidates,
    label_row,
    load_golden_questions,
    load_labeled,
    merge_candidates,
    merge_passes,
    needs_arbitration,
    needs_human,
    parse_labels,
    pick_controls,
    summarize,
    validate_label_spans,
    write_tsv,
)

pytestmark = pytest.mark.unit


def _write_csv(path: Path, rows: list[dict]) -> Path:
    fields = ["question", "ground_truth", "must_not_contain", "oos_type"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})
    return path


def _candidate(cid: str, text: str = "текст нормы", source: str = "a.pdf") -> dict:
    return {"chunk_id": cid, "text": text, "source": source}


class TestLoadGoldenQuestions:
    def test_reads_questions_with_running_numbers(self, tmp_path):
        p = _write_csv(
            tmp_path / "dataset.csv",
            [
                {"question": "Вопрос один?", "ground_truth": "Ответ один"},
                {"question": "Вопрос два?", "ground_truth": "Ответ два"},
            ],
        )
        records = load_golden_questions(p)
        assert [r["n"] for r in records] == [1, 2]
        assert records[0]["question"] == "Вопрос один?"
        assert records[0]["ground_truth"] == "Ответ один"

    def test_out_of_scope_questions_are_marked_not_labelable(self, tmp_path):
        p = _write_csv(
            tmp_path / "dataset.csv",
            [
                {"question": "В корпусе есть ответ?", "ground_truth": "Да"},
                {"question": "Как варить борщ?", "oos_type": "out_of_scope"},
                {
                    "question": "Когда отменили ГОСТ, которого нет?",
                    "oos_type": "false_premise",
                },
            ],
        )
        records = load_golden_questions(p)
        assert [r["in_scope"] for r in records] == [True, False, False]

    def test_numbering_covers_skipped_questions(self, tmp_path):
        """Numbers index the source file, so a filtered run stays comparable."""
        p = _write_csv(
            tmp_path / "dataset.csv",
            [
                {"question": "Как варить борщ?", "oos_type": "out_of_scope"},
                {"question": "Вопрос по существу?", "ground_truth": "Ответ"},
            ],
        )
        records = load_golden_questions(p)
        in_scope = [r for r in records if r["in_scope"]]
        assert [r["n"] for r in in_scope] == [2]

    def test_blank_questions_are_dropped(self, tmp_path):
        p = _write_csv(
            tmp_path / "dataset.csv",
            [
                {"question": "  ", "ground_truth": "нет вопроса"},
                {"question": "Вопрос?", "ground_truth": "Ответ"},
            ],
        )
        assert [r["question"] for r in load_golden_questions(p)] == ["Вопрос?"]


class TestMergeCandidates:
    def test_interleaves_pools_and_deduplicates(self):
        """Round-robin: neither path's ranking is privileged by position."""
        simple = [_candidate("a.pdf#1"), _candidate("a.pdf#2")]
        complex_ = [_candidate("a.pdf#2"), _candidate("b.pdf#7")]
        merged = merge_candidates([simple, complex_])
        assert [c["chunk_id"] for c in merged] == ["a.pdf#1", "a.pdf#2", "b.pdf#7"]

    def test_respects_the_limit(self):
        pools = [[_candidate(f"a.pdf#{i}") for i in range(10)]]
        assert len(merge_candidates(pools, limit=4)) == 4

    def test_empty_pools_give_empty_result(self):
        assert merge_candidates([[], []]) == []

    def test_records_which_pools_supplied_each_candidate(self):
        simple = [_candidate("a.pdf#1")]
        complex_ = [_candidate("a.pdf#1"), _candidate("b.pdf#2")]
        merged = merge_candidates([simple, complex_], pool_names=("simple", "complex"))
        by_id = {c["chunk_id"]: c for c in merged}
        assert by_id["a.pdf#1"]["pools"] == ["simple", "complex"]
        assert by_id["b.pdf#2"]["pools"] == ["complex"]


class TestFormatCandidates:
    def test_numbers_candidates_from_one(self):
        block = format_candidates(
            [_candidate("a.pdf#1", "первый"), _candidate("b.pdf#2", "второй")]
        )
        assert "[1]" in block and "[2]" in block
        assert "первый" in block and "второй" in block

    def test_shows_the_source_document(self):
        block = format_candidates(
            [_candidate("a.pdf#1", "текст", source="1479 ппр.pdf")]
        )
        assert "1479 ппр.pdf" in block


class TestParseLabels:
    def test_keeps_valid_indices_only(self):
        parsed = parse_labels(
            {
                "relevant": [
                    {"index": 2, "quote": "цитата"},
                    {"index": 99, "quote": "x"},
                ]
            },
            n_candidates=3,
        )
        assert [item["index"] for item in parsed["relevant"]] == [2]

    def test_deduplicates_repeated_indices(self):
        parsed = parse_labels(
            {"relevant": [{"index": 1, "quote": "a"}, {"index": 1, "quote": "b"}]},
            n_candidates=3,
        )
        assert [item["index"] for item in parsed["relevant"]] == [1]

    def test_empty_verdict_is_valid(self):
        parsed = parse_labels({"relevant": [], "note": "ответа нет"}, n_candidates=5)
        assert parsed["relevant"] == []
        assert parsed["note"] == "ответа нет"

    def test_garbage_indices_do_not_crash(self):
        parsed = parse_labels(
            {
                "relevant": [
                    {"index": "два", "quote": "a"},
                    {"index": None, "quote": "b"},
                ]
            },
            n_candidates=3,
        )
        assert parsed["relevant"] == []


class TestValidateLabelSpans:
    def test_quote_present_in_the_chunk_is_verified(self):
        candidates = [
            _candidate("a.pdf#1", "Инструктаж проводится раз в шесть месяцев.")
        ]
        checked = validate_label_spans(
            {"relevant": [{"index": 1, "quote": "раз в шесть месяцев"}]}, candidates
        )
        assert checked["relevant"][0]["verified"] is True

    def test_quote_absent_from_the_chunk_is_flagged_not_dropped(self):
        candidates = [
            _candidate("a.pdf#1", "Инструктаж проводится раз в шесть месяцев.")
        ]
        checked = validate_label_spans(
            {"relevant": [{"index": 1, "quote": "раз в год"}]}, candidates
        )
        assert checked["relevant"][0]["verified"] is False
        assert checked["relevant"][0]["index"] == 1

    def test_whitespace_differences_do_not_break_the_match(self):
        candidates = [
            _candidate("a.pdf#1", "Инструктаж   проводится\nраз в шесть месяцев.")
        ]
        checked = validate_label_spans(
            {"relevant": [{"index": 1, "quote": "проводится раз в шесть"}]}, candidates
        )
        assert checked["relevant"][0]["verified"] is True


class TestMergePasses:
    def test_agreement_on_the_same_chunk(self):
        strict = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        lenient = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        candidates = [_candidate("a.pdf#1"), _candidate("a.pdf#2")]
        merged = merge_passes(strict, lenient, candidates)
        assert merged["status"] == "agreed"
        assert merged["gold_chunk_ids"] == ["a.pdf#1"]
        assert merged["disputed_chunk_ids"] == []

    def test_disagreement_goes_to_the_dispute_list(self):
        strict = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        lenient = {
            "relevant": [
                {"index": 1, "quote": "a", "verified": True},
                {"index": 2, "quote": "b", "verified": True},
            ]
        }
        candidates = [_candidate("a.pdf#1"), _candidate("a.pdf#2")]
        merged = merge_passes(strict, lenient, candidates)
        assert merged["status"] == "disputed"
        assert merged["gold_chunk_ids"] == ["a.pdf#1"]
        assert merged["disputed_chunk_ids"] == ["a.pdf#2"]

    def test_both_passes_empty_means_no_gold_chunk(self):
        candidates = [_candidate("a.pdf#1")]
        merged = merge_passes({"relevant": []}, {"relevant": []}, candidates)
        assert merged["status"] == "none_found"
        assert merged["gold_chunk_ids"] == []

    def test_only_lenient_found_something_is_disputed_not_gold(self):
        strict = {"relevant": []}
        lenient = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        candidates = [_candidate("a.pdf#1")]
        merged = merge_passes(strict, lenient, candidates)
        assert merged["status"] == "disputed"
        assert merged["gold_chunk_ids"] == []
        assert merged["disputed_chunk_ids"] == ["a.pdf#1"]

    def test_unverified_quote_in_an_agreement_still_needs_a_human(self):
        strict = {"relevant": [{"index": 1, "quote": "a", "verified": False}]}
        lenient = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        candidates = [_candidate("a.pdf#1")]
        merged = merge_passes(strict, lenient, candidates)
        assert merged["gold_chunk_ids"] == ["a.pdf#1"]
        assert merged["status"] == "unverified_quote"


class TestNeedsHuman:
    def test_dispute_needs_a_human(self):
        assert needs_human({"status": "disputed"}) is True

    def test_nothing_found_needs_a_human(self):
        """A question with no gold chunk drops out of the metric — check it first."""
        assert needs_human({"status": "none_found"}) is True

    def test_unverified_quote_needs_a_human(self):
        assert needs_human({"status": "unverified_quote"}) is True

    def test_clean_agreement_does_not(self):
        assert needs_human({"status": "agreed"}) is False


class TestPickControls:
    def test_picks_the_requested_number_of_agreements(self):
        results = [{"n": i, "status": "agreed"} for i in range(1, 21)]
        controls = pick_controls(results, k=5, seed=1)
        assert len(controls) == 5
        assert all(r["status"] == "agreed" for r in controls)

    def test_is_deterministic_for_a_seed(self):
        results = [{"n": i, "status": "agreed"} for i in range(1, 21)]
        assert pick_controls(results, k=5, seed=1) == pick_controls(
            results, k=5, seed=1
        )

    def test_never_picks_a_case_already_going_to_the_human(self):
        results = [{"n": 1, "status": "disputed"}, {"n": 2, "status": "agreed"}]
        assert [r["n"] for r in pick_controls(results, k=2, seed=1)] == [2]

    def test_fewer_agreements_than_asked_is_not_an_error(self):
        results = [{"n": 1, "status": "agreed"}]
        assert len(pick_controls(results, k=10, seed=1)) == 1


class TestBuildLabeledGt:
    def test_keeps_questions_with_a_gold_chunk(self):
        records = [{"n": 1, "question": "Вопрос?"}]
        merged = {1: {"status": "agreed", "gold_chunk_ids": ["a.pdf#1"]}}
        gt = build_labeled_gt(records, merged)
        assert gt == [
            {
                "question": "Вопрос?",
                "chunk_id": "a.pdf#1",
                "relevant_chunk_ids": ["a.pdf#1"],
                "source": "a.pdf",
            }
        ]

    def test_question_without_a_gold_chunk_is_left_out(self):
        records = [{"n": 1, "question": "Вопрос?"}]
        merged = {1: {"status": "none_found", "gold_chunk_ids": []}}
        assert build_labeled_gt(records, merged) == []

    def test_disputed_chunks_are_not_written_as_gold(self):
        records = [{"n": 1, "question": "Вопрос?"}]
        merged = {
            1: {
                "status": "disputed",
                "gold_chunk_ids": ["a.pdf#1"],
                "disputed_chunk_ids": ["a.pdf#2"],
            }
        }
        gt = build_labeled_gt(records, merged)
        assert gt[0]["relevant_chunk_ids"] == ["a.pdf#1"]

    def test_multiple_gold_chunks_are_all_kept(self):
        records = [{"n": 1, "question": "Вопрос?"}]
        merged = {1: {"status": "agreed", "gold_chunk_ids": ["a.pdf#1", "b.pdf#3"]}}
        gt = build_labeled_gt(records, merged)
        assert gt[0]["relevant_chunk_ids"] == ["a.pdf#1", "b.pdf#3"]
        assert gt[0]["chunk_id"] == "a.pdf#1"


class TestLoadLabeled:
    def test_reads_the_resume_log(self, tmp_path):
        p = tmp_path / "raw.jsonl"
        p.write_text(
            json.dumps({"n": 1, "status": "agreed"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        assert load_labeled(p)[1]["status"] == "agreed"

    def test_missing_log_is_empty(self, tmp_path):
        assert load_labeled(tmp_path / "nope.jsonl") == {}

    def test_torn_last_line_is_skipped(self, tmp_path):
        p = tmp_path / "raw.jsonl"
        p.write_text(
            json.dumps({"n": 1, "status": "agreed"}, ensure_ascii=False) + '\n{"n": 2',
            encoding="utf-8",
        )
        assert list(load_labeled(p)) == [1]


class TestTsv:
    def test_row_carries_question_status_and_candidates(self):
        record = {"n": 3, "question": "Вопрос?"}
        merged = {
            "status": "disputed",
            "gold_chunk_ids": ["a.pdf#1"],
            "disputed_chunk_ids": ["a.pdf#2"],
            "note": "спор",
        }
        row = label_row(record, merged, role="dispute")
        assert row["n"] == 3
        assert row["status"] == "disputed"
        assert row["role"] == "dispute"
        assert "a.pdf#1" in row["gold"]
        assert "a.pdf#2" in row["disputed"]

    def test_written_sheet_has_a_header_and_the_rows(self, tmp_path):
        out = tmp_path / "review.tsv"
        write_tsv(
            [
                label_row(
                    {"n": 1, "question": "Вопрос?"},
                    {
                        "status": "disputed",
                        "gold_chunk_ids": [],
                        "disputed_chunk_ids": ["a.pdf#1"],
                    },
                    role="dispute",
                )
            ],
            out,
        )
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].split("\t")[0] == "n"
        assert len(lines) == 2


class TestSummarize:
    def test_counts_statuses_and_the_human_queue(self):
        results = [
            {"status": "agreed", "gold_chunk_ids": ["a#1"]},
            {"status": "disputed", "gold_chunk_ids": []},
            {"status": "none_found", "gold_chunk_ids": []},
        ]
        summary = summarize(results)
        assert summary["total"] == 3
        assert summary["agreed"] == 1
        assert summary["disputed"] == 1
        assert summary["none_found"] == 1
        assert summary["needs_human"] == 2
        assert summary["with_gold"] == 1


class TestHumanPriority:
    def test_no_gold_chunk_is_blocking(self):
        """Nothing agreed on: the question drops out of the metric until checked."""
        assert (
            human_priority({"status": "none_found", "gold_chunk_ids": []}) == "blocking"
        )
        assert (
            human_priority({"status": "disputed", "gold_chunk_ids": []}) == "blocking"
        )

    def test_dispute_over_an_extra_chunk_is_optional(self):
        """Hit Rate already has a gold chunk; one more only moves recall."""
        assert (
            human_priority(
                {
                    "status": "disputed",
                    "gold_chunk_ids": ["a#1"],
                    "disputed_chunk_ids": ["a#2"],
                }
            )
            == "optional"
        )

    def test_unverified_quote_is_blocking(self):
        assert (
            human_priority({"status": "unverified_quote", "gold_chunk_ids": ["a#1"]})
            == "blocking"
        )

    def test_clean_agreement_is_not_queued(self):
        assert human_priority({"status": "agreed", "gold_chunk_ids": ["a#1"]}) == ""


class TestReviewRowsOrder:
    def test_blocking_rows_come_first_then_optional_then_controls(self):
        records = {
            1: {"n": 1, "question": "Спор с эталоном?"},
            2: {"n": 2, "question": "Ничего не найдено?"},
            3: {"n": 3, "question": "Согласие?"},
        }
        results = [
            {
                "n": 1,
                "status": "disputed",
                "gold_chunk_ids": ["a#1"],
                "disputed_chunk_ids": ["a#2"],
            },
            {"n": 2, "status": "none_found", "gold_chunk_ids": []},
            {"n": 3, "status": "agreed", "gold_chunk_ids": ["a#3"]},
        ]
        rows = build_review_rows(records, results, controls=1, seed=1)
        assert [r["role"] for r in rows] == ["blocking", "optional", "control"]
        assert [r["n"] for r in rows] == [2, 1, 3]


class TestApplyArbitration:
    """Third pass on the stronger judge, called only where the two passes differ."""

    CANDIDATES = [_candidate("a.pdf#1"), _candidate("a.pdf#2"), _candidate("a.pdf#3")]

    def _merged(self, **over):
        base = {
            "status": "disputed",
            "gold_chunk_ids": ["a.pdf#1"],
            "disputed_chunk_ids": ["a.pdf#2"],
            "disputed_indices": [2],
            "quotes": [],
            "note": "",
        }
        base.update(over)
        return base

    def test_arbiter_confirming_a_disputed_chunk_promotes_it_to_gold(self):
        arbitration = {"relevant": [{"index": 2, "quote": "цитата", "verified": True}]}
        out = apply_arbitration(self._merged(), arbitration, self.CANDIDATES)
        assert out["gold_chunk_ids"] == ["a.pdf#1", "a.pdf#2"]
        assert out["disputed_chunk_ids"] == []
        assert out["status"] == "arbitrated"

    def test_arbiter_rejecting_a_disputed_chunk_drops_it(self):
        arbitration = {"relevant": []}
        out = apply_arbitration(self._merged(), arbitration, self.CANDIDATES)
        assert out["gold_chunk_ids"] == ["a.pdf#1"]
        assert out["disputed_chunk_ids"] == []
        assert out["status"] == "arbitrated"

    def test_arbiter_verdict_without_a_verified_quote_is_not_trusted(self):
        """Same rule as for the passes: a quote absent from the chunk is not evidence."""
        arbitration = {
            "relevant": [{"index": 2, "quote": "выдумка", "verified": False}]
        }
        out = apply_arbitration(self._merged(), arbitration, self.CANDIDATES)
        assert out["gold_chunk_ids"] == ["a.pdf#1"]
        assert out["status"] == "unverified_quote"

    def test_arbiter_may_not_add_a_chunk_nobody_disputed(self):
        """The arbiter rules on the disagreement, it does not relabel the pool."""
        arbitration = {"relevant": [{"index": 3, "quote": "цитата", "verified": True}]}
        out = apply_arbitration(self._merged(), arbitration, self.CANDIDATES)
        assert out["gold_chunk_ids"] == ["a.pdf#1"]

    def test_case_with_nothing_found_can_be_rescued_by_the_arbiter(self):
        merged = self._merged(
            status="none_found",
            gold_chunk_ids=[],
            disputed_chunk_ids=[],
            disputed_indices=[],
        )
        arbitration = {"relevant": [{"index": 1, "quote": "цитата", "verified": True}]}
        out = apply_arbitration(merged, arbitration, self.CANDIDATES)
        assert out["gold_chunk_ids"] == ["a.pdf#1"]
        assert out["status"] == "arbitrated"

    def test_no_arbitration_leaves_the_verdict_alone(self):
        merged = self._merged()
        assert apply_arbitration(merged, None, self.CANDIDATES) == merged

    def test_arbitrated_case_with_a_gold_chunk_is_off_the_human_queue(self):
        arbitration = {"relevant": [{"index": 2, "quote": "цитата", "verified": True}]}
        out = apply_arbitration(self._merged(), arbitration, self.CANDIDATES)
        assert needs_human(out) is False

    def test_arbitrated_case_without_a_gold_chunk_still_goes_to_the_human(self):
        merged = self._merged(
            status="none_found",
            gold_chunk_ids=[],
            disputed_chunk_ids=[],
            disputed_indices=[],
        )
        out = apply_arbitration(merged, {"relevant": []}, self.CANDIDATES)
        assert needs_human(out) is True
        assert human_priority(out) == "blocking"


class TestMergePassesCarriesIndices:
    def test_disputed_indices_are_kept_for_the_arbiter(self):
        strict = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        lenient = {
            "relevant": [
                {"index": 1, "quote": "a", "verified": True},
                {"index": 3, "quote": "b", "verified": True},
            ]
        }
        merged = merge_passes(strict, lenient, TestApplyArbitration.CANDIDATES)
        assert merged["disputed_indices"] == [3]

    def test_unverified_agreement_is_offered_to_the_arbiter_too(self):
        strict = {"relevant": [{"index": 1, "quote": "a", "verified": False}]}
        lenient = {"relevant": [{"index": 1, "quote": "a", "verified": True}]}
        merged = merge_passes(strict, lenient, TestApplyArbitration.CANDIDATES)
        assert merged["disputed_indices"] == [1]


class TestNeedsArbitration:
    def test_clean_agreement_needs_none(self):
        assert needs_arbitration({"status": "agreed"}) is False

    def test_dispute_needs_it(self):
        assert needs_arbitration({"status": "disputed"}) is True

    def test_unverified_quote_needs_it(self):
        assert needs_arbitration({"status": "unverified_quote"}) is True

    def test_nothing_found_needs_it(self):
        """The cheap judge finding nothing is exactly where a stronger one earns its price."""
        assert needs_arbitration({"status": "none_found"}) is True


class TestSummarizeArbitration:
    def test_arbitrated_cases_are_counted_and_not_queued(self):
        summary = summarize(
            [
                {"status": "arbitrated", "gold_chunk_ids": ["a#1"]},
                {"status": "arbitrated", "gold_chunk_ids": []},
            ]
        )
        assert summary["arbitrated"] == 2
        assert summary["with_gold"] == 1
        assert summary["needs_human"] == 1
