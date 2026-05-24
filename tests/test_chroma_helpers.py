from unittest.mock import MagicMock

from src.indexing.chroma_helpers import query_chunks_by_range, chroma_results_to_documents


class TestChromaResultsToDocuments:
    def test_converts_results_to_documents(self):
        result = {
            "documents": ["text1", "text2"],
            "metadatas": [{"source": "a.pdf"}, {"source": "b.pdf"}],
        }
        docs = chroma_results_to_documents(result)
        assert len(docs) == 2
        assert docs[0].page_content == "text1"
        assert docs[0].metadata == {"source": "a.pdf"}

    def test_empty_results(self):
        assert chroma_results_to_documents(None) == []
        assert chroma_results_to_documents({"documents": None}) == []
        assert chroma_results_to_documents({"documents": []}) == []


class TestQueryChunksByRange:
    def test_queries_chroma_and_returns_sorted_docs(self):
        mock_vs = MagicMock()
        mock_vs.get.return_value = {
            "documents": ["chunk2", "chunk1"],
            "metadatas": [
                {"source": "a.pdf", "chunk_id": 5},
                {"source": "a.pdf", "chunk_id": 3},
            ],
        }
        docs = query_chunks_by_range(mock_vs, "a.pdf", 3, 5)
        assert len(docs) == 2
        assert docs[0].metadata["chunk_id"] == 3  # sorted
        mock_vs.get.assert_called_once_with(
            where={
                "$and": [
                    {"source": "a.pdf"},
                    {"chunk_id": {"$gte": 3}},
                    {"chunk_id": {"$lte": 5}},
                ]
            }
        )

    def test_returns_empty_on_error(self):
        mock_vs = MagicMock()
        mock_vs.get.side_effect = Exception("DB error")
        assert query_chunks_by_range(mock_vs, "a.pdf", 1, 5) == []
