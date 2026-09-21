"""Generic normative-search adapter for the unified Streamlit screen."""

from __future__ import annotations

import os
import threading

# ANCHOR: bind the Generic V7 graph to its explicit Chroma store and expose one
# UI-facing query function; Department settings must never alias this store.
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from src.backends.chroma_backend import ChromaBackend
from src.infra.llm_factory import get_embedding_model
from src.v7.bridge import build_full_v7_runtime
from src.v7.graph import build_graph
from src.v7.runner import run_query


class GenericSearchError(RuntimeError):
    """Safe UI-facing failure for Generic search availability."""


_POOL: ThreadPoolExecutor | None = None
_SLOTS: threading.BoundedSemaphore | None = None
_POOL_LOCK = threading.Lock()


def _shared_pool() -> tuple[ThreadPoolExecutor, threading.BoundedSemaphore]:
    global _POOL, _SLOTS
    with _POOL_LOCK:
        if _POOL is None or _SLOTS is None:
            _POOL = ThreadPoolExecutor(
                max_workers=settings.GENERIC_QUERY_WORKERS,
                thread_name_prefix="generic-query",
            )
            _SLOTS = threading.BoundedSemaphore(
                settings.GENERIC_QUERY_WORKERS + settings.GENERIC_QUERY_PENDING
            )
        return _POOL, _SLOTS


def _validate_store_isolation() -> None:
    generic = (
        os.path.realpath(settings.GENERIC_CHROMA_DB_PATH),
        settings.GENERIC_CHROMA_COLLECTION,
    )
    department = {
        (
            os.path.realpath(settings.DEPARTMENT_V1_CHROMA_DB_PATH),
            settings.DEPARTMENT_V1_COLLECTION,
        ),
        (
            os.path.realpath(settings.DEPARTMENT_V2_CHROMA_DB_PATH),
            settings.DEPARTMENT_V2_COLLECTION,
        ),
    }
    if generic in department:
        raise GenericSearchError(
            "Общая нормативная база настроена на корпус подразделений."
        )


def load_generic_graph():
    """Build the full Generic V7 graph over the normative corpus."""
    _validate_store_isolation()
    try:
        store = ChromaBackend(
            path=settings.GENERIC_CHROMA_DB_PATH,
            collection=settings.GENERIC_CHROMA_COLLECTION,
            embeddings=get_embedding_model(),
        )
        runtime = build_full_v7_runtime(store)
        return build_graph(runtime=runtime).compile()
    except GenericSearchError:
        raise
    except Exception as exc:
        raise GenericSearchError("Общая нормативная база сейчас недоступна.") from exc


def answer_generic_question(graph, question: str, *, writer):
    """Run one normative-only question through the monitored Generic graph."""
    pool, slots = _shared_pool()
    if not slots.acquire(blocking=False):
        raise GenericSearchError("Сервис занят. Повторите запрос позже.")
    try:
        future = pool.submit(run_query, graph, question, source="ui", writer=writer)
    except BaseException:
        slots.release()
        raise
    future.add_done_callback(lambda _future: slots.release())
    try:
        return future.result()
    except Exception as exc:
        raise GenericSearchError("Не удалось выполнить поиск по нормативам.") from exc
