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

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

# ДО любых импортов из src: настроить structlog на stderr, иначе первый же лог
# уйдёт в stdout и испортит первое JSONRPC-сообщение stdio-транспорта.
from utils.logging import configure_logging

configure_logging()

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
    vsearch: object  # callable(query, top_k) -> list[passage] (v7-формат)
    rerank: object  # callable(query, passages, top_k) -> passages
    collection_name: str


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Тяжёлая инициализация — ровно один раз на процесс, до первого вызова тула.
    # Гибрид vector+BM25 (как в проде v7): BM25 добирает попадания по номерам
    # пунктов и точным терминам, где эмбеддинг табличных строк проваливается
    # (eval: СП 486 «независимо от площади» без BM25 не поднимался).
    from config.settings import settings
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import make_crossencoder_rerank_fn, make_vector_search_fn
    from src.v7.nlp_core import init_bm25_index

    backend = get_vector_store_backend(load_existing=True)
    init_bm25_index(list(backend.iter_all_documents()))  # BM25-корпус на процесс
    vsearch = make_vector_search_fn(backend)
    rerank = make_crossencoder_rerank_fn(settings.CROSSENCODER_MODEL)
    yield AppContext(
        backend=backend,
        vsearch=vsearch,
        rerank=rerank,
        collection_name=settings.CHROMA_COLLECTION_NAME,
    )


mcp = FastMCP("regulatory-mcp", lifespan=app_lifespan)


@mcp.tool()
def retrieve_chunks(ctx: Context, query: str, top_k: int = 5) -> dict:
    """Поиск по текстам нормативных документов. Возвращает сырые чанки с chunk_id —
    БЕЗ генерации ответа. Цитировать можно только дословный текст поля text,
    привязывая цитату к chunk_id."""
    from src.v7.nlp_core import bm25_search, rrf_merge

    app = ctx.request_context.lifespan_context
    n = top_k * OVERFETCH
    # vector + BM25 → RRF-фьюжн → CrossEncoder-реранк (тот же паттерн, что v7-прод).
    fused = rrf_merge(app.vsearch(query, top_k=n), bm25_search(query, top_k=n), top_k=n)
    reranked = app.rerank(query, fused, top_k)
    chunks = [
        {
            "chunk_id": make_public_id(p.get("metadata", {})),
            "doc_id": (p.get("metadata") or {}).get("source", app.collection_name),
            "text": p.get("text", ""),
            "score": p.get("score"),
            "section": (p.get("metadata") or {}).get("parent_section")
            or (p.get("metadata") or {}).get("heading_path", ""),
        }
        for p in reranked
    ]
    return {"collection": app.collection_name, "chunks": chunks}


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


_POINT_PREFIX = re.compile(r"^(?:п(?:ункт)?|ст(?:атья)?)\.?\s*", re.IGNORECASE)
_TABLE = re.compile(
    r"табл(?:ица)?\.?\s*(\d+)\s*,?\s*п(?:ункт)?\.?\s*([\d.]+)", re.IGNORECASE
)


def normalize_point(raw: str) -> str:
    """'п. 218' → '218'; 'табл.1 п.18' → '1.18' (таблица.строка). Детерминированно."""
    raw = raw.strip()
    t = _TABLE.match(raw)
    if t:
        return f"{t.group(1)}.{t.group(2)}"
    return _POINT_PREFIX.sub("", raw).strip().rstrip(".")


def _starts_with_number(text: str, number: str) -> bool:
    """Текст чанка начинается с номера пункта: '54. ...', '18) ...'. Точная граница,
    чтобы «5» не ловил «53. ...»."""
    return bool(re.match(rf"^\s*{re.escape(number)}[.)\s]", text))


def _point_matches(point: str, doc) -> bool:
    text = doc.page_content
    section = (doc.metadata or {}).get("parent_section", "") or ""
    if "." in point:
        # адрес вида таблица.строка ('1.18'): секция называет таблицу, текст — строку.
        table_no, row = point.split(".", 1)
        if f"аблица {table_no}" in section and _starts_with_number(text, row):
            return True
        # обычный многосоставный пункт ('4.4'): текст начинается с него целиком.
    return _starts_with_number(text, point)


def find_norm(backend, source: str, point: str) -> list:
    """Все чанки документа source, чей текст открывает пункт point. Без векторов."""
    docs = backend.get_by_filter({"source": source}, limit=2000)
    return [d for d in docs if _point_matches(point, d)]


@mcp.tool()
def get_norm(ctx: Context, doc_id: str, point: str) -> dict:
    """Точная адресная выборка текста пункта по (doc_id, point) — БЕЗ векторного поиска.

    doc_id — имя документа как в корпусе (source, напр. 'СП 486.1311500.2020.docx').
    point — '218', 'п. 4.4', 'табл.1 п.18'. Надёжна для обычных нумерованных пунктов;
    табличная адресация — best-effort (зависит от разметки таблиц в корпусе).
    found=false = пункт не найден в корпусе — НЕ доказательство, что его нет в документе
    (корпус может быть неполон). Тогда используй retrieve_chunks."""
    app = ctx.request_context.lifespan_context
    norm = normalize_point(point)
    hits = find_norm(app.backend, doc_id, norm)
    return {
        "found": bool(hits),
        "doc_id": doc_id,
        "point": norm,
        "chunks": [
            {
                "chunk_id": make_public_id(d.metadata),
                "text": d.page_content,
                "section": (d.metadata or {}).get("parent_section", ""),
            }
            for d in hits[:5]
        ],
    }


@mcp.tool()
def collection_info(ctx: Context) -> dict:
    """Какой корпус обслуживает этот сервер и сколько в нём чанков."""
    app = ctx.request_context.lifespan_context
    return {"collection": app.collection_name, "chunks": app.backend.count()}


if __name__ == "__main__":
    mcp.run()
