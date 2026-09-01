"""Unit tests for the retrieval ground-truth generator (eval/generate_retrieval_gt.py).

Covers the deterministic, LLM-free core: junk-chunk filtering, question
normalisation, exact + near-duplicate dedup, structured-output parsing and
GT-record construction. The LLM call and Chroma iteration are exercised
elsewhere (integration).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.generate_retrieval_gt import (
    Questions,
    is_junk_chunk,
    normalize_question,
    parse_questions,
    dedup_questions,
    build_gt_record,
)

pytestmark = pytest.mark.unit


class TestIsJunkChunk:
    def test_too_short(self):
        assert is_junk_chunk("Статья 5.") is True

    def test_normal_chunk_kept(self):
        text = (
            "Работодатель обязан обеспечить обучение по охране труда и проверку "
            "знания требований охраны труда в порядке, установленном Правительством "
            "Российской Федерации, с учётом мнения профсоюзного органа. " * 2
        )
        assert is_junk_chunk(text) is False

    def test_table_of_contents(self):
        toc = (
            "Содержание\n"
            "1. Общие положения ...... 3\n"
            "2. Требования к обучению ...... 7\n"
            "3. Порядок проверки знаний ...... 12\n"
            "4. Заключительные положения ...... 18\n"
        )
        assert is_junk_chunk(toc) is True


class TestNormalizeQuestion:
    def test_lowercase_and_strip_punctuation(self):
        assert normalize_question("  Кто проходит обучение по программе А?? ") == (
            "кто проходит обучение по программе а"
        )

    def test_collapses_whitespace(self):
        assert normalize_question("что   такое  СОУТ") == "что такое соут"


class TestParseQuestions:
    def test_from_pydantic_model(self):
        q = Questions(questions=["Вопрос один?", "Вопрос два?"])
        assert parse_questions(q) == ["Вопрос один?", "Вопрос два?"]

    def test_from_dict(self):
        assert parse_questions({"questions": ["A?", "B?"]}) == ["A?", "B?"]

    def test_drops_blanks_and_trims(self):
        assert parse_questions({"questions": ["  A?  ", "", "   "]}) == ["A?"]


class TestDedupQuestions:
    def _rec(self, q, cid="src.pdf#1"):
        return {
            "question": q,
            "chunk_id": cid,
            "source": "src.pdf",
            "chunk_preview": "x",
        }

    def test_removes_exact_duplicates_case_insensitive(self):
        recs = [
            self._rec("Кто проходит обучение?"),
            self._rec("кто проходит обучение??"),
            self._rec("Что такое СОУТ?"),
        ]
        kept, removed = dedup_questions(recs, embed_fn=None)
        assert removed == 1
        assert [r["question"] for r in kept] == [
            "Кто проходит обучение?",
            "Что такое СОУТ?",
        ]

    def test_near_duplicate_by_cosine(self):
        recs = [self._rec("A"), self._rec("B"), self._rec("C")]

        # A and B are near-identical (cosine ~1.0), C is orthogonal
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.999, 0.0447],
            "c": [0.0, 1.0],
        }

        def embed_fn(texts):
            return [vectors[t.lower()] for t in texts]

        kept, removed = dedup_questions(
            recs, embed_fn=embed_fn, near_dup_threshold=0.95
        )
        assert removed == 1
        assert {r["question"] for r in kept} == {"A", "C"}


class TestBuildGtRecord:
    def test_schema(self):
        chunk = {
            "chunk_id": 7,
            "text": "Полный текст чанка про обучение по охране труда." * 5,
            "metadata": {"source": "trudkodeks.pdf", "chunk_id": 7},
        }
        rec = build_gt_record("Кто обучается?", chunk)
        assert set(rec) == {"question", "chunk_id", "source", "chunk_preview"}
        assert rec["question"] == "Кто обучается?"
        assert rec["chunk_id"] == "trudkodeks.pdf#7"
        assert rec["source"] == "trudkodeks.pdf"
        assert rec["chunk_preview"].startswith("Полный текст чанка")
        assert len(rec["chunk_preview"]) <= 200
