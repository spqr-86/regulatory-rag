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
    to_passage,
    iter_corpus_chunks,
    _passage_identity,
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

    def test_chunk_id_zero_is_not_treated_as_missing(self):
        """chunk_id numbering is per-source and starts at 0 — a falsy but valid id."""
        passage = {
            "chunk_id": 0,
            "text": "текст",
            "metadata": {"source": "p2464.pdf", "chunk_id": 0},
        }
        assert build_gt_record("Q?", passage)["chunk_id"] == "p2464.pdf#0"


class TestToPassage:
    def test_lifts_chunk_id_to_top_level(self):
        doc = {"text": "t", "metadata": {"source": "a.pdf", "chunk_id": 3}}
        assert to_passage(doc) == {
            "text": "t",
            "metadata": {"source": "a.pdf", "chunk_id": 3},
            "chunk_id": 3,
        }

    def test_no_chunk_id_key_when_metadata_lacks_it(self):
        passage = to_passage({"text": "t", "metadata": {"source": "a.pdf"}})
        assert "chunk_id" not in passage


class TestIterCorpusChunks:
    def test_uses_backend_iter_all_documents(self):
        class FakeBackend:
            def iter_all_documents(self):
                yield {"text": "one", "metadata": {"source": "a.pdf", "chunk_id": 0}}
                yield {"text": "two", "metadata": {"source": "a.pdf", "chunk_id": 1}}

        passages = iter_corpus_chunks(backend=FakeBackend())
        assert [_passage_identity(p) for p in passages] == ["a.pdf#0", "a.pdf#1"]


class TestPassageIdentity:
    """The GT ids must equal the ids the retrieval runners (#6) emit, otherwise
    every Hit Rate silently reads 0."""

    CASES = [
        {"chunk_id": 4, "text": "x", "metadata": {"source": "a.pdf", "page_no": 2}},
        {"chunk_id": 0, "text": "x", "metadata": {"source": "a.pdf"}},
        {"text": "no chunk id here", "metadata": {"source": "a.pdf", "page_no": 7}},
        {"chunk_id": "", "text": "empty id", "metadata": {"source": "b.pdf"}},
        {"text": "no metadata at all", "metadata": None},
    ]

    def test_chunk_id_branch(self):
        assert _passage_identity(self.CASES[0]) == "a.pdf#4"

    def test_content_fallback_branch(self):
        assert _passage_identity(self.CASES[2]) == "a.pdf|7|no chunk id here"

    @pytest.mark.integration
    def test_identity_matches_nlp_core(self):
        """Pins the local mirror to the real passage_identity. Needs the full
        env (nlp_core imports pymorphy3), so it is not part of the unit run."""
        nlp_core = pytest.importorskip("src.v7.nlp_core")
        for case in self.CASES:
            assert _passage_identity(case) == nlp_core.passage_identity(case)


class TestPricing:
    """Cost must be computed for the model actually used.

    Regression: PRICE_PER_1M was a single hard-coded gpt-4o-mini rate applied to
    whatever model the factory happened to return, so both the pre-flight
    estimate and the COST_ABORT_USD guard could be off by the ratio between two
    models' prices.
    """

    def test_known_model_rate(self):
        from eval.generate_retrieval_gt import price_for

        assert price_for("gpt-4o-mini") == {"input": 0.15, "output": 0.60}
        assert price_for("gpt-4o") == {"input": 2.50, "output": 10.00}

    def test_unknown_model_raises(self):
        """Silently pricing an unknown model at some other model's rate is the bug."""
        from eval.generate_retrieval_gt import price_for

        with pytest.raises(ValueError, match="gpt-9-turbo"):
            price_for("gpt-9-turbo")

    def test_total_price_uses_given_model(self):
        from eval.generate_retrieval_gt import calc_total_price

        usages = [{"input": 1_000_000, "output": 1_000_000}]
        assert calc_total_price(usages, model="gpt-4o-mini") == pytest.approx(0.75)
        assert calc_total_price(usages, model="gpt-4o") == pytest.approx(12.50)

    def test_estimate_scales_with_model_price(self):
        from eval.generate_retrieval_gt import estimate_cost

        cheap = estimate_cost(1000, model="gpt-4o-mini")
        dear = estimate_cost(1000, model="gpt-4o")
        assert dear > cheap * 15

    def test_estimate_matches_manual_arithmetic(self):
        from eval.generate_retrieval_gt import estimate_cost

        # 500 input tokens (350 chunk + 150 overhead) + 60 output per chunk.
        expected = (500 / 1_000_000 * 0.15 + 60 / 1_000_000 * 0.60) * 100
        assert estimate_cost(100, model="gpt-4o-mini") == pytest.approx(expected)


class TestGeneratorModel:
    def test_generator_pins_its_own_model(self, monkeypatch):
        """The generator must not inherit the eval judge's model: judging answers
        and inventing questions are different jobs with different price tags."""
        import eval.generate_retrieval_gt as g

        captured = {}

        def fake_get_judge_llm(**kwargs):
            captured.update(kwargs)
            return object()

        import src.infra.llm_factory as lf

        monkeypatch.setattr(lf, "get_judge_llm", fake_get_judge_llm)

        g._make_llm()

        assert captured.get("model_name") == g.GEN_MODEL
        assert g.GEN_MODEL == "gpt-4o-mini"
