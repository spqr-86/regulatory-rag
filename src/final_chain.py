from __future__ import annotations

from operator import itemgetter

from flashrank import Ranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage

from config.settings import settings

from .applicability_retriever import ApplicabilityRetriever
from .llm_factory import get_llm
from .vector_store import load_vector_store
from .prompt_manager import PromptManager


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_reranked_retriever(
    vector_store, bm25_retriever, llm, query_expansion=True, use_reranker=True
):
    """Build an ApplicabilityRetriever with optional FlashRank reranking.

    Args:
        vector_store: Chroma vector store
        bm25_retriever: BM25Retriever instance
        llm: LLM for query expansion
        query_expansion: Whether to use LLM query expansion (disable for agent mode)
        use_reranker: Whether to apply FlashRank reranking (default True)
    """
    ensemble = ApplicabilityRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        llm=llm,
        search_kwargs={"k": settings.VECTOR_SEARCH_K},
        query_expansion=query_expansion,
    )

    if not use_reranker:
        return ensemble

    flashrank_client = Ranker(
        model_name=settings.RERANKING_MODEL,
        cache_dir=settings.FLASHRANK_CACHE_DIR,
    )
    compressor = FlashrankRerank(client=flashrank_client, top_n=12)
    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble
    )


def create_final_hybrid_chain():
    print("Создание финальной гибридной RAG-цепочки...")
    vector_store = load_vector_store()

    # BM25 is rebuilt on startup — no pickle cache (Option A from CARD-2.1b).
    # Build time is O(n_chunks): ~1-2s for current 1973 chunks. Acceptable.
    # Pickle cache was removed as it allowed RCE if an attacker can write to the dir.
    print("Построение индекса BM25 (может занять время)...")
    all_data = vector_store.get(include=["metadatas", "documents"])
    all_docs_as_objects = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_data["documents"], all_data["metadatas"])
    ]
    keyword_retriever = BM25Retriever.from_documents(all_docs_as_objects)
    keyword_retriever.k = settings.VECTOR_SEARCH_K

    llm = get_llm()

    final_retriever = build_reranked_retriever(
        vector_store, keyword_retriever, llm, query_expansion=True
    )

    prompt_manager = PromptManager()

    def render_prompt(inputs):
        text = prompt_manager.render("final_chain", **inputs)
        return [HumanMessage(content=text)]

    final_chain = (
        {
            "context": itemgetter("question") | final_retriever,
            "question": itemgetter("question"),
        }
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | RunnableLambda(render_prompt)
        | llm
        | StrOutputParser()
    )

    agent_reranker = build_reranked_retriever(
        vector_store, keyword_retriever, llm, query_expansion=False, use_reranker=False
    )

    print("Финальная гибридная цепочка успешно создана.")
    return final_chain, final_retriever, agent_reranker
