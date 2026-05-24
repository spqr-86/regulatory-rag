"""Упрощённый RAG-pipeline для коллекции ГОСТов ЕРС-Инжиниринг.

Без domain gate, без LangGraph — прямой путь:
    semantic search → FlashRank rerank → DeepSeek generate

Используется из api.py для эндпоинта POST /query/gosts.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any

import structlog
from langchain_chroma import Chroma

logger = structlog.get_logger()

GOSTS_CHROMA_PATH = "./chroma_db_gosts"
GOSTS_COLLECTION_NAME = "wta_gosts"
TOP_K = 15
RERANK_TOP_N = 8


@lru_cache(maxsize=1)
def _load_store() -> Chroma:
    from src.infra.llm_factory import get_embedding_model

    if not os.path.isdir(GOSTS_CHROMA_PATH):
        raise FileNotFoundError(
            f"ChromaDB ГОСТов не найдена: {GOSTS_CHROMA_PATH}. "
            "Запустите python index_gosts.py"
        )
    embeddings = get_embedding_model()
    vs = Chroma(
        collection_name=GOSTS_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=GOSTS_CHROMA_PATH,
    )
    count = vs._collection.count()
    if count == 0:
        raise ValueError("ChromaDB ГОСТов пуста. Запустите python index_gosts.py")
    logger.info("gosts_pipeline: store loaded", chunks=count)
    return vs


def _retrieve(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    vs = _load_store()
    docs_and_scores = vs.similarity_search_with_score(query, k=top_k)
    results = []
    for doc, distance in docs_and_scores:
        similarity = round(1.0 / (1.0 + distance), 4)
        results.append(
            {
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
                "score": similarity,
                "vector_score": similarity,
            }
        )
    return results


def _rerank(
    query: str, passages: list[dict[str, Any]], top_n: int = RERANK_TOP_N
) -> list[dict[str, Any]]:
    try:
        from flashrank import Ranker, RerankRequest

        ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".flashrank_cache"
        )
        rerank_req = RerankRequest(
            query=query,
            passages=[{"id": i, "text": p["text"]} for i, p in enumerate(passages)],
        )
        results = ranker.rerank(rerank_req)
        reranked = []
        for r in results[:top_n]:
            orig = passages[r["id"]]
            reranked.append({**orig, "score": orig["vector_score"]})
        return reranked
    except Exception as exc:
        logger.warning(
            "gosts_pipeline: rerank failed, using vector order", error=str(exc)
        )
        return passages[:top_n]


def _generate(query: str, passages: list[dict[str, Any]]) -> str:
    import os
    from openai import OpenAI

    def _short_source(p: dict[str, Any]) -> str:
        raw = p.get("metadata", {}).get("source", "")
        name = raw.split(" - ")[0].replace(".pdf", "").replace(".docx", "").strip()
        return name or "Неизвестный источник"

    passages_text = "\n\n".join(
        f"[{i + 1}] [Источник: {_short_source(p)}]\n{p.get('text', '')}"
        for i, p in enumerate(passages)
    )
    prompt = (
        "Ты эксперт по техническим нормативам (ГОСТы, СНиП, СП). "
        "Отвечай на вопрос используя ТОЛЬКО приведённые фрагменты. "
        "Указывай источник в квадратных скобках [N]. "
        "Если ответа в фрагментах нет — так и скажи.\n\n"
        f"Вопрос: {query}\n\n"
        f"Фрагменты ({len(passages)} шт.):\n{passages_text}\n\n"
        "Ответ:"
    )
    try:
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("gosts_pipeline: generate failed", error=str(exc))
        return "\n\n".join(p.get("text", "") for p in passages[:5])


def query(question: str) -> dict[str, Any]:
    """Выполнить запрос к коллекции ГОСТов.

    Returns:
        dict с ключами: answer, passages, path, elapsed_sec
    """
    t0 = time.perf_counter()
    passages = _retrieve(question)
    passages = _rerank(question, passages)
    answer = _generate(question, passages)
    elapsed = round(time.perf_counter() - t0, 2)

    logger.info(
        "gosts_pipeline: query done",
        question=question[:80],
        passages=len(passages),
        elapsed_sec=elapsed,
    )
    return {
        "answer": answer,
        "passages": passages,
        "path": "retrieve → rerank → generate",
        "elapsed_sec": elapsed,
    }
