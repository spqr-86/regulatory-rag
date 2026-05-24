from __future__ import annotations

from typing import List, Optional, Dict, Any

from langchain_classic.retrievers.multi_query import LineListOutputParser
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from pydantic import PrivateAttr

from .prompt_manager import PromptManager


class ApplicabilityRetriever(BaseRetriever):
    vector_store: VectorStore
    bm25_retriever: BM25Retriever
    llm: Any  # Changed to Any to avoid Pydantic validation issues with RunnableRetry
    search_kwargs: Dict[str, Any] = {"k": 10}
    weights: List[float] = [0.6, 0.4]  # Semantic, Keyword
    query_expansion: bool = (
        True  # Disable for Multi-Agent (agent handles decomposition)
    )
    _expansion_cache: Dict[str, List[str]] = PrivateAttr(default_factory=dict)

    def _generate_queries(self, original_query: str) -> List[str]:
        """Генерирует вариации поискового запроса с помощью LLM.

        Uses LangChain's ``LineListOutputParser`` from MultiQueryRetriever to
        parse line-separated alternatives, instead of hand-rolled splitting.
        Full migration to ``MultiQueryRetriever`` is blocked by this class's
        custom ensemble behaviour (BM25 only on original query, similarity
        score propagation, content-prefix dedup); see CARD-5.2 notes.
        """
        if original_query in self._expansion_cache:
            return self._expansion_cache[original_query]

        prompt_manager = PromptManager()
        try:
            prompt_str = prompt_manager.render(
                "applicability_retriever", question=original_query
            )
            prompt = PromptTemplate.from_template(prompt_str)
            parser = LineListOutputParser()
            chain = prompt | self.llm | parser
            queries = chain.invoke({})

            # Добавляем оригинал, если его нет
            if original_query not in queries:
                queries.insert(0, original_query)

            final_queries = queries[:4]  # Ограничиваем сверху
            self._expansion_cache[original_query] = final_queries
            return final_queries
        except Exception:
            # Fallback если LLM сломалась или промпт не найден
            return [original_query]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        # 1. Генерация вариаций (Multi-Query) — skip if disabled
        queries = self._generate_queries(query) if self.query_expansion else [query]
        # print(f"DEBUG: Generated queries: {queries}")  # Можно раскомментировать для отладки

        all_docs = []

        # 2. Параллельный поиск
        for q in queries:
            # Semantic Search для каждой вариации
            docs_and_scores = self.vector_store.similarity_search_with_score(
                q, **self.search_kwargs
            )
            for doc, score in docs_and_scores:
                # Chroma returns distance. Assuming cosine distance: similarity = 1 - distance
                doc.metadata["similarity_score"] = 1.0 - score
                all_docs.append(doc)

        # BM25 ищем только по оригиналу (ключевые слова важны именно пользовательские)
        # или можно добавить первую (legal) вариацию, если хочется
        docs_keyword = self.bm25_retriever.invoke(query)
        all_docs.extend(docs_keyword)

        # 3. Дедупликация (Reciprocal Rank Fusion не делаем, просто уникальность)
        unique_docs = {}
        for doc in all_docs:
            # Используем хэш контента как ключ
            # (chunk_id в метаданных был бы идеален, но полагаемся на контент)
            key = doc.page_content[
                :200
            ]  # Хэш по началу текста (заголовок + часть тела)
            if key not in unique_docs:
                unique_docs[key] = doc

        return list(unique_docs.values())
