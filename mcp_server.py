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

import structlog
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
    address: object  # AddressIndex | None — адресный слой (точная выборка пунктов)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # Тяжёлая инициализация — ровно один раз на процесс, до первого вызова тула.
    # Гибрид vector+BM25 (как в проде v7): BM25 добирает попадания по номерам
    # пунктов и точным терминам, где эмбеддинг табличных строк проваливается
    # (eval: СП 486 «независимо от площади» без BM25 не поднимался).
    import os

    from config.settings import settings
    from src.address_layer import AddressIndex
    from src.backends.vector_store import get_vector_store_backend
    from src.v7.bridge import make_crossencoder_rerank_fn, make_vector_search_fn
    from src.v7.nlp_core import init_bm25_index

    backend = get_vector_store_backend(load_existing=True)
    init_bm25_index(list(backend.iter_all_documents()))  # BM25-корпус на процесс
    vsearch = make_vector_search_fn(backend)
    rerank = make_crossencoder_rerank_fn(settings.CROSSENCODER_MODEL)
    # Адресный слой: точная выборка пунктов/строк таблиц. Нет файла → None,
    # get_norm деградирует на chunk-scan (не падает).
    address_path = os.environ.get("ADDRESS_INDEX_PATH", "address_index_pb.json")
    address = AddressIndex.load(address_path)
    yield AppContext(
        backend=backend,
        vsearch=vsearch,
        rerank=rerank,
        collection_name=settings.CHROMA_COLLECTION_NAME,
        address=address,
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


def _get_address_chunk(app, source: str, local_id: str) -> dict:
    """Чанк адресного слоя по синтетическому id 'addr:kind:num[.row]'."""
    spec = local_id[len(ADDR_PREFIX) :]
    kind, _, rest = spec.partition(":")
    if kind == "table":
        num, _, row = rest.partition(".")
        recs = (
            app.address.lookup(source, "table", num, row or None) if app.address else []
        )
    else:
        recs = app.address.lookup(source, kind, rest, None) if app.address else []
    if recs:
        return {
            "found": True,
            "chunk_id": f"{source}{ID_SEP}{ADDR_PREFIX}{spec}",
            "doc_id": source,
            "text": recs[0]["text"],
        }
    return {"found": False}


@mcp.tool()
def get_chunk(ctx: Context, chunk_id: str) -> dict:
    """Чанк по его chunk_id ("source::N" из retrieve_chunks или "source::addr:…" из
    get_norm) — для контекста и верификации цитат."""
    app = ctx.request_context.lifespan_context
    source, local_id = split_public_id(chunk_id)
    if not source:
        return {"found": False, "error": "ожидается chunk_id формата source::N"}
    if local_id.startswith(ADDR_PREFIX):
        return _get_address_chunk(app, source, local_id)
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


_POINT_PREFIX = re.compile(r"^п(?:ункт)?\.?\s*", re.IGNORECASE)
_ARTICLE = re.compile(r"^ст(?:атья)?\.?\s*([\d.]+)", re.IGNORECASE)
_TABLE = re.compile(
    r"табл(?:ица)?\.?\s*(\d+)\s*,?\s*(?:п(?:ункт)?\.?\s*([\d.]+))?", re.IGNORECASE
)
# Страница выборки метаданных на документ. Корпус ПБ — 647 чанков суммарно; при росте
# документа за лимит find_norm логирует потерю хвоста (см. ниже), молча не режем.
FIND_NORM_LIMIT = 2000


def normalize_point(raw: str) -> tuple[str, str, str | None]:
    """Разбор адреса в типизированный вид (kind, num, row). Детерминированно, без LLM.

    'п. 218' / '218'   → ('point', '218', None)
    'ст. 83'           → ('article', '83', None)
    'табл.1 п.18'      → ('table', '1', '18')
    'табл.1'           → ('table', '1', None)  — вся таблица
    Табличная ветка включается ТОЛЬКО по явной форме «табл.…» — не угадывается по точке
    в номере: «4.4» — это пункт 4.4, а не строка 4 таблицы 4."""
    raw = raw.strip()
    t = _TABLE.match(raw)
    if t:
        return ("table", t.group(1), t.group(2))
    a = _ARTICLE.match(raw)
    if a:
        return ("article", a.group(1), None)
    return ("point", _POINT_PREFIX.sub("", raw).strip().rstrip("."), None)


def _starts_with_number(text: str, number: str) -> bool:
    """Текст чанка начинается с номера пункта: '54. ...', '18) ...'. Граница точная:
    «5» не ловит «53.» и «5.1», «54» не ловит «54.1» (после точки не должно идти цифры).
    """
    return bool(re.match(rf"^\s*{re.escape(number)}(?:\.(?!\d)|\)|\s)", text))


def _point_matches(kind: str, num: str, row: str | None, doc) -> bool:
    text = doc.page_content
    section = (doc.metadata or {}).get("parent_section", "") or ""
    if kind == "table":
        # секция называет таблицу; row=None — отдаём все строки таблицы.
        if f"аблица {num}" not in section:
            return False
        return True if row is None else _starts_with_number(text, row)
    if kind == "article":
        # статьи начинаются со слова: «Статья 83. …»
        return bool(
            re.match(
                rf"^\s*статья\s+{re.escape(num)}(?:\.(?!\d)|\)|\s)", text, re.IGNORECASE
            )
        )
    return _starts_with_number(text, num)


def find_norm(backend, source: str, address: tuple[str, str, str | None]) -> list:
    """Все чанки документа source по типизированному адресу. Без векторов."""
    kind, num, row = address
    docs = backend.get_by_filter({"source": source}, limit=FIND_NORM_LIMIT)
    if len(docs) >= FIND_NORM_LIMIT:
        structlog.get_logger().warning(
            "find_norm: документ упёрся в лимит выборки, хвост может быть потерян",
            source=source,
            limit=FIND_NORM_LIMIT,
        )
    return [d for d in docs if _point_matches(kind, num, row, d)]


ADDR_PREFIX = "addr:"


def _addr_id(doc_id: str, kind: str, num: str, row: str | None) -> str:
    """Синтетический chunk_id адресной записи: 'doc::addr:kind:num[.row]'.
    Обратимо парсится в get_chunk — цитата из адресного слоя верифицируется как любая.
    """
    tail = f"{kind}:{num}" if row is None else f"{kind}:{num}.{row}"
    return f"{doc_id}{ID_SEP}{ADDR_PREFIX}{tail}"


def _address_hits(address_index, doc_id: str, kind, num, row) -> list[dict]:
    """Записи адресного слоя → чанки в формате get_norm (с синтетическим chunk_id)."""
    recs = address_index.lookup(doc_id, kind, num, row)
    return [
        {
            "chunk_id": _addr_id(doc_id, kind, r.get("num", num), r.get("row")),
            "text": r["text"],
            "section": f"Таблица {r['table']}" if r["kind"] == "table" else "",
        }
        for r in recs
    ]


@mcp.tool()
def get_norm(ctx: Context, doc_id: str, point: str) -> dict:
    """Точная адресная выборка текста пункта по (doc_id, point) — БЕЗ векторного поиска.

    doc_id — имя документа как в корпусе (source, напр. 'СП 486.1311500.2020.docx').
    point — '218', 'п. 4.4', 'ст. 83', 'табл.1 п.18', 'табл.1' (вся таблица).
    Источник — адресный слой (точный разбор docx); при промахе — скан чанков корпуса.
    found=false = пункт не найден в корпусе — НЕ доказательство, что его нет в документе
    (корпус может быть неполон). Тогда используй retrieve_chunks."""
    app = ctx.request_context.lifespan_context
    kind, num, row = normalize_point(point)
    chunks: list[dict] = []
    if app.address is not None:
        chunks = _address_hits(app.address, doc_id, kind, num, row)
    if not chunks:  # адресного слоя нет или промах — деградируем на chunk-scan
        chunks = [
            {
                "chunk_id": make_public_id(d.metadata),
                "text": d.page_content,
                "section": (d.metadata or {}).get("parent_section", ""),
            }
            for d in find_norm(app.backend, doc_id, (kind, num, row))
        ]
    return {
        "found": bool(chunks),
        "doc_id": doc_id,
        "point": num if row is None else f"{num}.{row}",
        "kind": kind,
        "chunks": chunks[:5],
    }


@mcp.tool()
def collection_info(ctx: Context) -> dict:
    """Какой корпус обслуживает этот сервер и сколько в нём чанков."""
    app = ctx.request_context.lifespan_context
    return {"collection": app.collection_name, "chunks": app.backend.count()}


if __name__ == "__main__":
    mcp.run()
