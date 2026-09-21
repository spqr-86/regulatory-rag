"""Explicit store loading against disposable local Chroma collections."""

# ANCHOR: exercise the installed Chroma API with deterministic local embeddings.
# No production indices, model downloads or provider calls.

import pytest


@pytest.mark.integration
def test_two_explicit_chroma_stores_remain_isolated(tmp_path):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings

    from src.backends.chroma_backend import ChromaBackend

    class LocalEmbedding(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, float(len(text)), 0.0] for text in texts]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    embedding = LocalEmbedding()
    for name in ("alpha", "bravo"):
        store = Chroma(
            collection_name=name,
            persist_directory=str(tmp_path / name),
            embedding_function=embedding,
        )
        store.add_texts([name], metadatas=[{"source": name}], ids=[name])
    a = ChromaBackend(
        path=str(tmp_path / "alpha"), collection="alpha", embeddings=embedding
    )
    b = ChromaBackend(
        path=str(tmp_path / "bravo"), collection="bravo", embeddings=embedding
    )
    for store, expected in ((a, "alpha"), (b, "bravo"), (a, "alpha")):
        hits = store.similarity_search_with_score("query", k=1)
        assert hits[0][0].page_content == expected
    with pytest.raises(Exception, match="does not exist"):
        ChromaBackend(
            path=str(tmp_path / "alpha"), collection="missing", embeddings=embedding
        )
