"""V7: pack_context — владелец подготовки контекста до вердикта.

Всё, что может убавить или изменить доказательство, происходит ДО вердикта.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Callable, Dict, List, Optional

from src.v7.config import v7_config
from src.v7.contract import PackResult, PackStatus
from src.v7.gap import build_gap
from src.v7.hard_gates import sanitize_for_llm
from src.v7.nlp_core import passage_identity

# ANCHOR: стабильная версия кандидата для кеша упаковки.
# Input: passages + retrieval plan + active query. Output: SHA-256 key.
# Key invariant: порядок пассажей не влияет, их текст влияет.

_SEP = "|"
# Надбавка на заголовок чанка ("[3] (HIGH) [Источник: …; Раздел: …]"),
# который bridge приписывает каждому пассажу. Без неё бюджет упаковки
# систематически занижает размер промпта.
HEADER_TOKENS_ALLOWANCE = 48

logger = logging.getLogger(__name__)

# ─── DI: expander инжектится один раз при старте (bridge.init_v7_pipeline) ───

_crossref_expander: Optional[Callable[[List[dict], str], List[dict]]] = None


def set_crossref_expander(
    fn: Optional[Callable[[List[dict], str], List[dict]]],
) -> None:
    """Инжект расширителя перекрёстных ссылок.

    Это единственная точка вызова expander во всём пайплайне: решающий код
    не расширяет ничего.
    """
    global _crossref_expander
    _crossref_expander = fn


def candidate_version(passages: List[dict], plan: dict, query: str) -> str:
    """Стабильный ключ версии кандидата.

    Включает идентичность И текст пассажей (enrichment меняет текст при том же
    chunk_id), снимок плана и запрос, с которым работает expander. Не зависит
    от порядка списка: инвариант «expander один раз на версию» иначе непроверяем.
    """
    ids = sorted(
        f"{passage_identity(p)}#{hashlib.sha256((p.get('text') or '').encode('utf-8')).hexdigest()[:16]}"
        for p in passages
    )
    plan_snapshot = json.dumps(plan or {}, sort_keys=True, default=str)
    raw = _SEP.join(ids) + _SEP + plan_snapshot + _SEP + (query or "")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def approx_tokens(text: str) -> int:
    """То же приближение, что использовал bridge: 4 символа на токен."""
    return len(text) // 4


def passage_cost(p: dict) -> int:
    """Стоимость пассажа в промпте: текст плюс заголовок, который допишет bridge."""
    return approx_tokens(p.get("text", "")) + HEADER_TOKENS_ALLOWANCE


def _passage_source(p: dict) -> str:
    return (p.get("metadata") or {}).get("source") or p.get("doc_id", "")


def _merge_new_at_tail(base: List[dict], expanded: List[dict]) -> List[dict]:
    """Вернуть ``base``, затем закрывающие ссылки и остальное расширение.

    Реальный expander вставляет bbox-соседей рядом с родителем; отдавать этот
    порядок дальше — значит вытолкнуть исходный чанк из головы списка (#30).
    Но простое добавление всего нового в хвост тоже ошибочно: лимит чанков мог
    отсечь найденный expander-ом текст пункта раньше нерелевантных соседей.
    """
    seen = {(_passage_source(p), p.get("text", "")) for p in base}
    tail = [p for p in expanded if (_passage_source(p), p.get("text", "")) not in seen]

    # Стабильный greedy-проход: переносим вперёд только пассажи, которые
    # уменьшают множество открытых ссылок из исходного top-5. Порядок base и
    # относительный порядок остальных результатов expander сохраняются.
    prioritized: List[dict] = []
    remaining = list(tail)
    resolve_in = list(base)
    open_refs = set(build_gap(base, resolve_in=resolve_in)["open"])
    while open_refs:
        chosen = None
        chosen_open = open_refs
        for i, passage in enumerate(remaining):
            candidate_open = set(
                build_gap(base, resolve_in=resolve_in + [passage])["open"]
            )
            if len(candidate_open) < len(chosen_open):
                chosen = i
                chosen_open = candidate_open
        if chosen is None:
            break
        passage = remaining.pop(chosen)
        prioritized.append(passage)
        resolve_in.append(passage)
        open_refs = chosen_open

    return list(base) + prioritized + remaining


def pack_context(
    passages: List[dict],
    query: str,
    plan: dict,
    *,
    cache: Optional[Dict[str, PackResult]] = None,
) -> PackResult:
    """Упаковать кандидата в контекст, который увидит генератор.

    Порядок (спек §1): expand → sanitize → обрезка MAX_CHUNKS_FOR_LLM →
    бюджет токенов. degraded (expander упал) запрещает снимать refs_resolved
    (см. validate_context). Вход не мутируется; кеш отдаёт копию.
    """
    if not passages:
        return {"final_context": [], "status": "ok", "dropped": 0}

    key = candidate_version(passages, plan, query)
    if cache is not None and key in cache:
        return copy.deepcopy(cache[key])

    status: PackStatus = "ok"
    working = copy.deepcopy(passages)

    if _crossref_expander is not None:
        try:
            expanded = list(_crossref_expander(copy.deepcopy(working), query))
            if expanded:
                working = _merge_new_at_tail(working, expanded)
        except Exception as exc:  # noqa: BLE001 — живой запрос не должен умирать
            logger.warning("pack_context: expansion failed: %s", exc)
            status = "degraded"
            working = copy.deepcopy(passages)

    packed = [{**p, "text": sanitize_for_llm(p.get("text", ""))} for p in working]

    n_before = len(packed)
    packed = packed[: v7_config.MAX_CHUNKS_FOR_LLM]

    budget = v7_config.PACK_TOKEN_BUDGET
    kept: List[dict] = []
    spent = 0
    for p in packed:
        cost = passage_cost(p)
        if spent + cost > budget:
            break
        kept.append(p)
        spent += cost

    result: PackResult = {
        "final_context": kept,
        "status": status,
        "dropped": n_before - len(kept),
    }
    if cache is not None:
        cache[key] = copy.deepcopy(result)
    return result
