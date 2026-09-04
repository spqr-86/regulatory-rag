"""Unit tests for the stratified GT sampler (eval/sample_retrieval_gt.py).

The sampler carves a hand-checkable subset (400-600 questions) out of the full
synthetic GT. Two things must hold: the per-document split follows the chosen
weights (corpus shares, not the skewed GT counts), and the draw is reproducible
from a seed — a reviewed sample is worthless if it cannot be rebuilt.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.sample_retrieval_gt import (
    allocate,
    load_records,
    stratified_sample,
    sample_distribution,
)

pytestmark = pytest.mark.unit


def make_records(spec: dict[str, int], questions_per_chunk: int = 3) -> list[dict]:
    """``{"a.pdf": 4}`` -> 4 chunks of that source, N questions each."""
    out = []
    for source, n_chunks in spec.items():
        for c in range(n_chunks):
            for q in range(questions_per_chunk):
                out.append(
                    {
                        "question": f"{source} chunk {c} question {q}?",
                        "chunk_id": f"{source}#{c}",
                        "source": source,
                        "chunk_preview": f"text of {source} chunk {c}",
                    }
                )
    return out


class TestAllocate:
    def test_proportional_split(self):
        assert allocate({"a": 900, "b": 100}, 100) == {"a": 90, "b": 10}

    def test_total_is_exact_with_awkward_remainders(self):
        quotas = allocate({"a": 1, "b": 1, "c": 1}, 100)
        assert sum(quotas.values()) == 100

    def test_largest_remainder_gets_the_odd_seat(self):
        # shares 0.5 / 3.5 / 6.0 of ten seats -> b has the largest remainder
        quotas = allocate({"a": 5, "b": 35, "c": 60}, 10)
        assert quotas == {"a": 1, "b": 3, "c": 6}

    def test_capped_by_availability_and_redistributed(self):
        # b is entitled to 50 but only 5 exist; the rest goes back to a
        quotas = allocate({"a": 50, "b": 50}, 100, available={"a": 95, "b": 5})
        assert quotas == {"a": 95, "b": 5}
        assert sum(quotas.values()) == 100

    def test_source_absent_from_weights_gets_nothing(self):
        assert allocate({"a": 10}, 5, available={"a": 10, "b": 10})["b"] == 0

    def test_asking_for_more_than_exists_returns_everything(self):
        quotas = allocate({"a": 1, "b": 1}, 100, available={"a": 3, "b": 4})
        assert quotas == {"a": 3, "b": 4}


class TestStratifiedSample:
    def test_split_follows_external_weights_not_gt_counts(self):
        # GT is skewed: 'small.pdf' is over-represented relative to the corpus
        records = make_records({"big.pdf": 10, "small.pdf": 90})
        sample = stratified_sample(
            records, n=100, seed=0, weights={"big.pdf": 900, "small.pdf": 100}
        )
        dist = sample_distribution(sample)
        assert dist == {"big.pdf": 30, "small.pdf": 70}  # big capped at its 30 chunks

    def test_defaults_to_gt_counts_when_no_weights(self):
        records = make_records({"a.pdf": 30, "b.pdf": 10})
        dist = sample_distribution(stratified_sample(records, n=40, seed=0))
        assert dist == {"a.pdf": 30, "b.pdf": 10}

    def test_weighted_source_missing_from_gt_does_not_eat_seats(self):
        # 782н has a corpus share but zero GT questions: its seats must go to the
        # documents that can actually fill them, not vanish from the sample
        records = make_records({"a.pdf": 100, "b.pdf": 100})
        sample = stratified_sample(
            records,
            n=100,
            seed=0,
            weights={"a.pdf": 50, "b.pdf": 25, "782н.pdf": 25},
        )
        assert len(sample) == 100
        assert sample_distribution(sample) == {"a.pdf": 67, "b.pdf": 33}

    def test_exact_size(self):
        records = make_records({"a.pdf": 50, "b.pdf": 50})
        assert len(stratified_sample(records, n=137, seed=0)) == 137

    def test_deterministic_for_a_seed(self):
        records = make_records({"a.pdf": 50, "b.pdf": 50})
        first = stratified_sample(records, n=60, seed=7)
        second = stratified_sample(records, n=60, seed=7)
        assert first == second

    def test_different_seed_draws_differently(self):
        records = make_records({"a.pdf": 50, "b.pdf": 50})
        assert stratified_sample(records, n=60, seed=1) != stratified_sample(
            records, n=60, seed=2
        )

    def test_one_question_per_chunk_while_chunks_last(self):
        records = make_records({"a.pdf": 40}, questions_per_chunk=3)
        sample = stratified_sample(records, n=40, seed=0)
        assert len({r["chunk_id"] for r in sample}) == 40

    def test_falls_back_to_extra_questions_when_chunks_run_out(self):
        records = make_records({"a.pdf": 10}, questions_per_chunk=3)
        sample = stratified_sample(records, n=25, seed=0)
        assert len(sample) == 25
        assert len({r["chunk_id"] for r in sample}) == 10
        assert len({r["question"] for r in sample}) == 25

    def test_no_duplicate_questions(self):
        records = make_records({"a.pdf": 20, "b.pdf": 20})
        sample = stratified_sample(records, n=80, seed=3)
        assert len({r["question"] for r in sample}) == len(sample)

    def test_asking_for_more_than_exists_returns_everything(self):
        records = make_records({"a.pdf": 5}, questions_per_chunk=2)
        assert len(stratified_sample(records, n=999, seed=0)) == 10

    def test_records_are_returned_unmodified(self):
        records = make_records({"a.pdf": 5})
        sample = stratified_sample(records, n=5, seed=0)
        assert all(r in records for r in sample)


class TestLoadRecords:
    def test_reads_jsonl(self, tmp_path):
        path = tmp_path / "gt.jsonl"
        records = make_records({"a.pdf": 2})
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        assert load_records(path) == records

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "gt.jsonl"
        path.write_text('\n{"question": "q?", "source": "a.pdf"}\n\n', encoding="utf-8")
        assert load_records(path) == [{"question": "q?", "source": "a.pdf"}]

    def test_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_records(tmp_path / "nope.jsonl")
