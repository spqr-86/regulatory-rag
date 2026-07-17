"""regulatory-mcp: retrieval-only MCP-сервер поверх retrieval-ядра этого repo.

НЕ трогает api.py и v7-граф. Отдаёт сырые чанки с составным chunk_id — генерации
ответа нет: цитирование и программная верификация цитат — забота вызывающего агента
(prombez-agent). Паттерн — FastMCP lifespan: Chroma-бэкенд (с правильной embedding
function из настроек) и CrossEncoder-реранкер грузятся один раз на процесс, до
первого вызова тула.

Коллекция задаётся окружением (как весь этот repo), по коллекции на процесс:
    CHROMA_DB_PATH=./chroma_db_pb CHROMA_COLLECTION_NAME=fire_safety \
        ./venv/bin/python mcp_server.py

chunk_id в этом repo уникален только внутри документа (per-source), поэтому наружу
отдаётся составной id "source::chunk_id" — глобально уникальный и обратимый.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import Context, FastMCP

OVERFETCH = 4  # кандидатов на реранк на каждый итоговый результат
ID_SEP = "::"


def make_public_id(meta: dict) -> str:
    return f"{meta.get('source', '?')}{ID_SEP}{meta.get('chunk_id', '?')}"


def split_public_id(public_id: str) -> tuple[str, str]:
    source, _, local = public_id.rpartition(ID_SEP)
    return source, local


@dataclass
class AppContext:
    backend: object  # VectorStoreBackend (их протокол)
    rerank: object  # callable(query, passages, top_k) -> passages
    collection_name: str


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Тяжёлая инициализация — ровно один раз на процесс, до первого вызова тула.
    from config.settings import settings
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import make_crossencoder_rerank_fn

    backend = get_vector_store_backend(load_existing=True)
    rerank = make_crossencoder_rerank_fn(settings.CROSSENCODER_MODEL)
    yield AppContext(
        backend=backend, rerank=rerank, collection_name=settings.CHROMA_COLLECTION_NAME
    )


mcp = FastMCP("regulatory-mcp", lifespan=app_lifespan)


@mcp.tool()
def retrieve_chunks(ctx: Context, query: str, top_k: int = 5) -> dict:
    """Поиск по текстам нормативных документов. Возвращает сырые чанки с chunk_id —
    БЕЗ генерации ответа. Цитировать можно только дословный текст поля text,
    привязывая цитату к chunk_id."""
    app = ctx.request_context.lifespan_context
    hits = app.backend.similarity_search_with_score(query, k=top_k * OVERFETCH)
    passages = [
        {
            "chunk_id": make_public_id(doc.metadata),
            "doc_id": doc.metadata.get("source", app.collection_name),
            "text": doc.page_content,
            "score": float(score),
            "section": doc.metadata.get("parent_section")
            or doc.metadata.get("heading_path", ""),
        }
        for doc, score in hits
    ]
    return {
        "collection": app.collection_name,
        "chunks": app.rerank(query, passages, top_k),
    }


@mcp.tool()
def get_chunk(ctx: Context, chunk_id: str) -> dict:
    """Чанк по его chunk_id (формат "source::N") — для контекста и верификации цитат."""
    app = ctx.request_context.lifespan_context
    source, local_id = split_public_id(chunk_id)
    if not source:
        return {"found": False, "error": "ожидается chunk_id формата source::N"}
    for str_or_int in (local_id, _as_int(local_id)):
        if str_or_int is None:
            continue
        docs = app.backend.get_by_filter(
            {"$and": [{"source": source}, {"chunk_id": str_or_int}]}, limit=1
        )
        if docs:
            d = docs[0]
            return {
                "found": True,
                "chunk_id": chunk_id,
                "doc_id": d.metadata.get("source", ""),
                "text": d.page_content,
            }
    return {"found": False}


def _as_int(s: str):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


@mcp.tool()
def collection_info(ctx: Context) -> dict:
    """Какой корпус обслуживает этот сервер и сколько в нём чанков."""
    app = ctx.request_context.lifespan_context
    return {"collection": app.collection_name, "chunks": app.backend.count()}


if __name__ == "__main__":
    mcp.run()
