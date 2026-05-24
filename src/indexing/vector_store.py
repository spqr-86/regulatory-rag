from __future__ import annotations

import datetime
import json
import os
from functools import lru_cache
from typing import Any, Iterable, List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from config.settings import settings
from src.infra.llm_factory import get_embedding_model
from utils.logging import logger

try:
    import tiktoken
except Exception:
    tiktoken = None


def _token_len_openai(text: str) -> int:
    if not tiktoken:
        return max(1, len(text) // 4)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Превращает сложные значения в допустимые для Chroma скаляры."""
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (datetime.datetime, datetime.date)):
            out[k] = v.isoformat()
        elif isinstance(v, (list, tuple, dict, set)):
            try:
                out[k] = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                out[k] = str(v)
        else:
            # всё остальное в строку
            out[k] = str(v)
        # опционально: ограничим длину огромных полей
        if isinstance(out[k], str) and len(out[k]) > 2000:
            out[k] = out[k][:2000] + "…"
    return out


def _batches_by_tokens(
    docs: List[Document],
    max_tokens_per_batch: int = 280_000,
    hard_batch_cap: int = 256,
    is_openai: bool = False,
) -> Iterable[List[Document]]:
    batch: List[Document] = []
    cur_tokens = 0
    for d in docs:
        t = _token_len_openai(d.page_content) if is_openai else len(d.page_content) // 4
        if t > max_tokens_per_batch and batch:
            yield batch
            batch, cur_tokens = [], 0
        if (cur_tokens + t > max_tokens_per_batch) or (len(batch) >= hard_batch_cap):
            yield batch
            batch, cur_tokens = [], 0
        batch.append(d)
        cur_tokens += t
    if batch:
        yield batch


def _create_chroma_instance(embeddings) -> Chroma:
    """Create a Chroma instance with standard settings."""
    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_DB_PATH,
    )


def create_vector_store(chunks: List[Document]) -> Chroma:
    logger.info("Создание новой векторной базы данных...")
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)

    embeddings = get_embedding_model()
    is_openai = embeddings.__class__.__name__ in {
        "OpenAIEmbeddings",
        "AzureOpenAIEmbeddings",
    }

    vector_store = _create_chroma_instance(embeddings)

    total, done = len(chunks), 0
    for batch in _batches_by_tokens(
        chunks,
        max_tokens_per_batch=280_000,
        hard_batch_cap=128,
        is_openai=is_openai,
    ):
        texts = [d.page_content for d in batch]
        metas = [_sanitize_metadata(d.metadata or {}) for d in batch]  # ✅ тут
        vector_store.add_texts(texts=texts, metadatas=metas)
        done += len(batch)
        logger.info(f"Chroma add: прогресс {done}/{total}")

    logger.info(f"Векторная БД сохранена: {settings.CHROMA_DB_PATH}")
    return vector_store


@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """Load existing Chroma collection (singleton — cached per process)."""
    if not os.path.isdir(settings.CHROMA_DB_PATH):
        raise FileNotFoundError(f"Chroma DB не найдена: {settings.CHROMA_DB_PATH}")

    embeddings = get_embedding_model()
    vs = _create_chroma_instance(embeddings)

    count = vs._collection.count()
    if count == 0:
        raise ValueError(
            f"Chroma DB пуста (0 документов): {settings.CHROMA_DB_PATH}. "
            "Запустите 'python index.py' для индексации."
        )
    logger.info(f"Chroma DB загружена: {count} документов")
    return vs
