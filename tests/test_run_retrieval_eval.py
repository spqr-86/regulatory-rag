"""Unit tests for the retrieval eval runner (eval/run_retrieval_eval.py).

Covers the LLM-free, Chroma-free core: GT loading, chunk_id extraction from
passages, metric aggregation over a stub retrieval_fn, report formatting, and
the parity contract — calling the retrieval node directly must return the same
chunk_ids as the full graph with the same injected engine.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.run_retrieval_eval import (
    DEFAULT_KS,
    evaluate,
    extract_chunk_ids,
    format_report,
    load_gt,
    make_retrieval_fn,
)

pytestmark = pytest.mark.unit


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    return path


class TestLoadGt:
    def test_reads_question_and_chunk_id(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "gt.jsonl",
            [
                {"question": "Вопрос один?", "chunk_id": "a.pdf#1", "source": "a.pdf"},
                {"question": "Вопрос два?", "chunk_id": "a.pdf#2", "source": "a.pdf"},
            ],
        )
        records = load_gt(p)
        assert [r["chunk_id"] for r in records] == ["a.pdf#1", "a.pdf#2"]
        assert records[0]["question"] == "Вопрос один?"

    def test_skips_records_without_question_or_chunk_id(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "gt.jsonl",
            [
                {"question": "Хороший вопрос?", "chunk_id": "a.pdf#1"},
                {"question": "", "chunk_id": "a.pdf#2"},
                {"question": "Без метки?", "chunk_id": ""},
                {"chunk_id": "a.pdf#3"},
            ],
        )
        assert len(load_gt(p)) == 1

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "gt.jsonl"
        p.write_text(
            '{"question": "Вопрос?", "chunk_id": "a.pdf#1"}\n\n', encoding="utf-8"
        )
        assert len(load_gt(p)) == 1

    def test_limit_truncates(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "gt.jsonl",
            [{"question": f"Вопрос {i}?", "chunk_id": f"a.pdf#{i}"} for i in range(10)],
        )
        assert len(load_gt(p, limit=3)) == 3

    def test_source_defaults_to_chunk_id_prefix(self, tmp_path):
        p = _write_jsonl(
            tmp_path / "gt.jsonl", [{"question": "Вопрос?", "chunk_id": "a.pdf#7"}]
        )
        assert load_gt(p)[0]["source"] == "a.pdf"


def _passage(source: str, chunk_id, text: str = "текст фрагмента") -> dict:
    return {
        "text": text,
        "chunk_id": chunk_id,
        "metadata": {"source": source, "chunk_id": chunk_id},
    }


class TestExtractChunkIds:
    def test_builds_source_qualified_identity(self):
        """Index-time chunk_id is a per-source int; GT ids are "source#id"."""
        passages = [_passage("a.pdf", 1), _passage("a.pdf", 2)]
        assert extract_chunk_ids(passages) == ["a.pdf#1", "a.pdf#2"]

    def test_same_chunk_id_in_two_documents_does_not_collide(self):
        passages = [_passage("a.pdf", 5), _passage("b.pdf", 5)]
        assert extract_chunk_ids(passages) == ["a.pdf#5", "b.pdf#5"]

    def test_matches_nlp_core_identity(self):
        from src.v7.nlp_core import passage_identity

        p = _passage("1479 ппр.pdf", 608)
        assert extract_chunk_ids([p]) == [passage_identity(p)]

    def test_preserves_order(self):
        passages = [_passage("b.pdf", 2), _passage("a.pdf", 1), _passage("c.pdf", 3)]
        assert extract_chunk_ids(passages) == ["b.pdf#2", "a.pdf#1", "c.pdf#3"]

    def test_dedups_keeping_first_position(self):
        passages = [_passage("a.pdf", 1), _passage("a.pdf", 1), _passage("a.pdf", 2)]
        assert extract_chunk_ids(passages) == ["a.pdf#1", "a.pdf#2"]

    def test_falls_back_to_content_hash_without_chunk_id(self):
        """Chunks from an un-reindexed store still get a stable, distinct key."""
        passages = [
            {"text": "первый фрагмент", "metadata": {"source": "a.pdf"}},
            {"text": "второй фрагмент", "metadata": {"source": "a.pdf"}},
        ]
        ids = extract_chunk_ids(passages)
        assert len(set(ids)) == 2
        assert all(i.startswith("a.pdf|") for i in ids)

    def test_empty_input(self):
        assert extract_chunk_ids([]) == []


class TestEvaluate:
    def _gt(self):
        return [
            {"question": "q1", "chunk_id": "a#1", "source": "a.pdf"},
            {"question": "q2", "chunk_id": "b#2", "source": "b.pdf"},
        ]

    def test_perfect_retrieval_scores_one(self):
        def fn(query):
            return {"q1": ["a#1"], "q2": ["b#2"]}[query]

        res = evaluate(self._gt(), fn, ks=(5,))
        assert res["metrics"]["hit_rate@5"] == 1.0
        assert res["metrics"]["mrr"] == 1.0
        assert res["n"] == 2

    def test_total_miss_scores_zero(self):
        res = evaluate(self._gt(), lambda q: ["z#9"], ks=(5,))
        assert res["metrics"]["hit_rate@5"] == 0.0
        assert res["metrics"]["mrr"] == 0.0

    def test_hit_rate_respects_k_cutoff(self):
        # relevant chunk sits at rank 3 → hit at k=5, miss at k=2
        def fn(query):
            return ["x#1", "y#2", {"q1": "a#1", "q2": "b#2"}[query]]

        res = evaluate(self._gt(), fn, ks=(2, 5))
        assert res["metrics"]["hit_rate@2"] == 0.0
        assert res["metrics"]["hit_rate@5"] == 1.0

    def test_mrr_uses_rank_of_first_relevant(self):
        def fn(query):
            return ["x#1", {"q1": "a#1", "q2": "b#2"}[query]]

        res = evaluate(self._gt(), fn, ks=(5,))
        assert res["metrics"]["mrr"] == pytest.approx(0.5)

    def test_mrr_is_not_truncated_by_max_k(self):
        """MRR is reported over the full retrieved list, not cut at max(ks)."""

        def fn(query):
            return ["x#1", "y#2", {"q1": "a#1", "q2": "b#2"}[query]]

        res = evaluate(self._gt(), fn, ks=(2,))
        assert res["metrics"]["mrr"] == pytest.approx(1 / 3)

    def test_per_source_breakdown(self):
        def fn(query):
            return ["a#1"] if query == "q1" else ["z#9"]

        res = evaluate(self._gt(), fn, ks=(5,))
        assert res["per_source"]["a.pdf"]["hit_rate@5"] == 1.0
        assert res["per_source"]["b.pdf"]["hit_rate@5"] == 0.0
        assert res["per_source"]["a.pdf"]["n"] == 1

    def test_records_per_question_result(self):
        res = evaluate(self._gt(), lambda q: ["a#1"], ks=(5,))
        by_q = {r["question"]: r for r in res["records"]}
        assert by_q["q1"]["hit"] is True
        assert by_q["q1"]["rank"] == 1
        assert by_q["q2"]["hit"] is False
        assert by_q["q2"]["rank"] is None
        assert by_q["q1"]["retrieved"] == ["a#1"]

    def test_retrieval_failure_counts_as_miss_not_crash(self):
        def fn(query):
            if query == "q2":
                raise RuntimeError("engine exploded")
            return ["a#1"]

        res = evaluate(self._gt(), fn, ks=(5,))
        assert res["metrics"]["hit_rate@5"] == 0.5
        assert res["errors"] == 1

    def test_empty_gt_returns_zero_n_without_crash(self):
        res = evaluate([], lambda q: [], ks=(5,))
        assert res["n"] == 0
        assert res["metrics"]["hit_rate@5"] == 0.0

    def test_default_ks_are_5_10_12(self):
        assert DEFAULT_KS == (5, 10, 12)


class TestFormatReport:
    def test_contains_metrics_and_counts(self):
        res = evaluate(
            [{"question": "q1", "chunk_id": "a#1", "source": "a.pdf"}],
            lambda q: ["a#1"],
            ks=(5,),
        )
        res["path"] = "simple"
        text = format_report(res)
        assert "simple" in text
        assert "hit_rate@5" in text
        assert "mrr" in text
        assert "a.pdf" in text


class TestRetrievalFnParity:
    """The runner calls the retrieval node directly, bypassing the graph.

    Contract: with the same injected engine, that shortcut must return the
    same chunk_ids as the full graph — otherwise the measured number is not
    the number the product serves.
    """

    QUERY = "Требования к ограждениям лестничных клеток в здании"

    @pytest.fixture
    def stub_engine(self):
        from src.v7 import nlp_core
        from src.v7.nodes import rag_complex as rag_complex_mod
        from src.v7.nodes import rag_simple as rag_simple_mod

        corpus = [
            {
                "text": (
                    "Ограждения лестничных клеток в здании выполняются высотой не "
                    "менее 1,2 метра и рассчитываются на нагрузку не менее 0,3 кН/м. "
                    f"Пункт {i}."
                ),
                "metadata": {"chunk_id": f"gost.pdf#{i}", "source": "gost.pdf"},
            }
            for i in range(20)
        ]

        def fake_vector_search(query, filters=None, top_k=12, **kwargs):
            out = []
            for i, doc in enumerate(corpus[:top_k]):
                meta = dict(doc["metadata"])
                score = round(0.60 - i * 0.01, 4)
                out.append(
                    {
                        "text": doc["text"],
                        "metadata": meta,
                        "score": score,
                        "vector_score": score,
                        "chunk_id": meta["chunk_id"],
                        "doc_id": meta["source"],
                    }
                )
            return out

        prev_bm25 = nlp_core._bm25_index
        nlp_core.init_bm25_index(corpus)
        rag_simple_mod.set_vector_search(fake_vector_search)
        rag_complex_mod.set_vector_search(fake_vector_search)
        yield
        # The BM25 index and the injected searches are module globals — leaving
        # them behind reds out unrelated tests in the same process.
        nlp_core._bm25_index = prev_bm25
        rag_simple_mod.set_vector_search(rag_simple_mod._default_vector_search)
        rag_complex_mod.set_vector_search(rag_complex_mod._default_vector_search)

    def _graph_chunk_ids(self, stage: str) -> list[str]:
        from src.v7.graph import build_graph

        result = build_graph().compile().invoke({"query": self.QUERY})
        attempts = [
            a for a in (result.get("retrieval_attempts") or []) if a["stage"] == stage
        ]
        assert attempts, f"graph produced no {stage} retrieval attempt"
        return extract_chunk_ids(attempts[-1]["passages"])

    def test_simple_matches_graph(self, stub_engine):
        expected = self._graph_chunk_ids("simple")
        assert make_retrieval_fn("simple")(self.QUERY) == expected

    def test_complex_matches_graph(self, stub_engine):
        expected = self._graph_chunk_ids("complex")
        assert make_retrieval_fn("complex")(self.QUERY) == expected

    def test_short_query_returns_empty(self, stub_engine):
        assert make_retrieval_fn("simple")("что?") == []

    def test_unknown_path_rejected(self):
        with pytest.raises(ValueError):
            make_retrieval_fn("medium")
