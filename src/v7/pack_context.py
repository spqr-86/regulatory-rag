"""V7: pack_context — владелец подготовки контекста до вердикта.

Всё, что может убавить или изменить доказательство, происходит ДО вердикта.
"""

from __future__ import annotations

import hashlib
import json
from typing import List

from src.v7.nlp_core import passage_identity

# ANCHOR: стабильная версия кандидата для кеша упаковки.
# Input: passages + retrieval plan + active query. Output: SHA-256 key.
# Key invariant: порядок пассажей не влияет, их текст влияет.

_SEP = "|"


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
