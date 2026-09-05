"""Tests for src/v7/bridge.py — v7 ↔ existing retriever bridge."""

from __future__ import annotations


import pytest
from unittest.mock import MagicMock, patch
from langchain_chroma import Chroma  # noqa: F401 — used as MagicMock spec

from src.v7.bridge import (
    init_v7_from_chroma,
    make_generate_fn,
    make_vector_search_fn,
)


class TestMakeVectorSearchFn:
    @pytest.mark.unit
    def test_returns_callable(self):
        mock_store = MagicMock()
        fn = make_vector_search_fn(mock_store)
        assert callable(fn)

    @pytest.mark.unit
    def test_calls_similarity_search_with_score(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        result = fn(query="test query", top_k=5)
        mock_store.similarity_search_with_score.assert_called_once()
        assert result == []

    @pytest.mark.unit
    def test_converts_documents_to_dicts(self):
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "some text about safety"
        mock_doc.metadata = {"source": "gost.pdf", "page_no": 5}
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.3)]
        fn = make_vector_search_fn(mock_store)
        result = fn(query="safety", top_k=10)
        assert len(result) == 1
        assert result[0]["text"] == "some text about safety"
        assert result[0]["metadata"]["source"] == "gost.pdf"
        # L2 distance 0.3 → similarity = 1/(1+0.3) ≈ 0.7692
        assert result[0]["score"] == pytest.approx(1.0 / 1.3, abs=0.01)

    @pytest.mark.unit
    def test_lifts_chunk_id_to_top_level(self):
        mock_store = MagicMock()
        mock_doc = MagicMock()
        mock_doc.page_content = "text"
        mock_doc.metadata = {"source": "gost.pdf", "chunk_id": 7}
        mock_store.similarity_search_with_score.return_value = [(mock_doc, 0.2)]
        fn = make_vector_search_fn(mock_store)
        result = fn(query="q", top_k=3)
        assert result[0]["chunk_id"] == 7

    @pytest.mark.unit
    def test_respects_top_k(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        fn(query="test", top_k=20)
        call_kwargs = mock_store.similarity_search_with_score.call_args
        assert call_kwargs[1].get("k") == 20 or call_kwargs[0] == ("test",)

    @pytest.mark.unit
    def test_filters_ignored_gracefully(self):
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = []
        fn = make_vector_search_fn(mock_store)
        result = fn(query="test", top_k=5, filters={"doc_type": "gost"})
        assert result == []


class TestMakeGenerateFn:
    @pytest.mark.unit
    def test_returns_callable(self):
        mock_llm = MagicMock()
        fn = make_generate_fn(mock_llm)
        assert callable(fn)

    @pytest.mark.unit
    def test_calls_llm_and_returns_answer(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = (
            "Ответ: высота ограждений не менее 1.2 м."
        )
        fn = make_generate_fn(mock_llm)
        result = fn(
            query="высота ограждений",
            active_query="высота ограждений",
            passages=[{"text": "Ограждения высотой не менее 1.2 м.", "score": 0.8}],
        )
        answer, usage = result
        assert answer == "Ответ: высота ограждений не менее 1.2 м."
        assert usage["node"] == "generate"
        mock_llm.invoke.assert_called_once()

    @pytest.mark.unit
    def test_returns_empty_for_no_passages(self):
        mock_llm = MagicMock()
        fn = make_generate_fn(mock_llm)
        result = fn(query="вопрос", active_query="вопрос", passages=[])
        assert result[0] == ""
        mock_llm.invoke.assert_not_called()

    @pytest.mark.unit
    def test_fallback_on_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM unavailable")
        fn = make_generate_fn(mock_llm)
        passages = [{"text": "текст фрагмента", "score": 0.7}]
        result = fn(query="вопрос", active_query="вопрос", passages=passages)
        assert "текст фрагмента" in result

    @pytest.mark.unit
    def test_handles_gemini_style_content(self):
        """Gemini returns content as list of dicts with 'text' key."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = [{"text": "Синтезированный ответ."}]
        fn = make_generate_fn(mock_llm)
        result = fn(
            query="вопрос",
            active_query="вопрос",
            passages=[{"text": "фрагмент", "score": 0.75}],
        )
        assert result[0] == "Синтезированный ответ."

    @pytest.mark.unit
    def test_includes_low_ranked_passages_in_prompt(self):
        """final_passages is capped at 24 upstream (merge_all_passages). The
        generator must not re-truncate below that — the answer-bearing passage
        can rank low (#22 observed in eval) and must still reach the LLM.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "ответ"
        fn = make_generate_fn(mock_llm)
        passages = [
            {"text": f"нерелевантный фрагмент {i}", "score": 0.5} for i in range(24)
        ]
        passages[21]["text"] = "УНИКАЛЬНЫЙ_МАРКЕР_ОТВЕТА"  # ranks #22

        fn(query="вопрос", active_query="вопрос", passages=passages)

        prompt = mock_llm.invoke.call_args[0][0][0].content
        assert "УНИКАЛЬНЫЙ_МАРКЕР_ОТВЕТА" in prompt


class TestInitV7FromChroma:
    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_injects_vector_search(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock(spec=Chroma)
        mock_store.get.return_value = {
            "documents": ["doc1 text", "doc2 text"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        init_v7_from_chroma(mock_store, llm_provider=None)
        mock_simple.set_vector_search.assert_called_once()
        mock_complex.set_vector_search.assert_called_once()
        mock_bm25.assert_called_once()

    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_bm25_corpus_built_from_chroma(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock(spec=Chroma)
        mock_store.get.return_value = {
            "documents": ["text A", "text B"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        init_v7_from_chroma(mock_store, llm_provider=None)
        corpus = mock_bm25.call_args[0][0]
        assert len(corpus) == 2
        assert corpus[0]["text"] == "text A"
        assert corpus[1]["metadata"]["source"] == "b.pdf"

    @pytest.mark.unit
    @patch("src.v7.bridge.get_simple_llm")
    @patch("src.v7.bridge.get_complex_llm")
    @patch("src.v7.bridge.generate_answer_mod")
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_injects_llm_fns_when_provider_set(
        self,
        mock_complex,
        mock_simple,
        mock_bm25,
        mock_generate,
        mock_get_complex_llm,
        mock_get_simple_llm,
    ):
        mock_store = MagicMock(spec=Chroma)
        mock_store.get.return_value = {
            "documents": ["d"],
            "metadatas": [{"source": "a.pdf"}],
        }
        mock_get_complex_llm.return_value = MagicMock()
        mock_get_simple_llm.return_value = MagicMock()
        init_v7_from_chroma(mock_store, llm_provider="gemini")
        mock_generate.set_generate_fns.assert_called_once()
        # simple generator + expander use get_simple_llm = 2 calls.
        assert mock_get_simple_llm.call_count == 2
        # get_complex_llm: complex generator = 1 call.
        assert mock_get_complex_llm.call_count == 1
        # The two generators must be different LLM instances.
        kwargs = mock_generate.set_generate_fns.call_args.kwargs
        assert "simple" in kwargs and "complex_" in kwargs
        assert kwargs["simple"] is not kwargs["complex_"]

    @pytest.mark.unit
    @patch("src.v7.bridge.get_complex_llm", side_effect=ImportError("no gemini"))
    @patch("src.v7.bridge.generate_answer_mod")
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_falls_back_to_stubs_on_llm_error(
        self,
        mock_complex,
        mock_simple,
        mock_bm25,
        mock_generate,
        mock_get_llm,
    ):
        mock_store = MagicMock(spec=Chroma)
        mock_store.get.return_value = {
            "documents": ["d"],
            "metadatas": [{"source": "a.pdf"}],
        }
        # Should not raise, just log warning
        init_v7_from_chroma(mock_store, llm_provider="gemini")
        mock_generate.set_generate_fns.assert_not_called()

    @pytest.mark.unit
    @patch("src.v7.bridge.init_bm25_index")
    @patch("src.v7.bridge.rag_simple_mod")
    @patch("src.v7.bridge.rag_complex_mod")
    def test_skips_llm_when_provider_none(self, mock_complex, mock_simple, mock_bm25):
        mock_store = MagicMock(spec=Chroma)
        mock_store.get.return_value = {
            "documents": ["d"],
            "metadatas": [{"source": "a.pdf"}],
        }
        # llm_provider=None should skip LLM injection entirely
        init_v7_from_chroma(mock_store, llm_provider=None)
        # No error, search still injected
        mock_simple.set_vector_search.assert_called_once()
