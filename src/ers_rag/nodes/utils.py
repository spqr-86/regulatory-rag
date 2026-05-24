"""Shared utilities for v7 nodes."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional


def make_retrieval_id(query: str, filters: Optional[dict] = None) -> str:
    """Детерминированный хеш запроса + фильтров."""
    filters_str = json.dumps(filters or {}, sort_keys=True, ensure_ascii=False)
    raw = (query.strip().lower() + "|" + filters_str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def extract_doc_identifiers(text: str) -> set[str]:
    """Извлечь индексы нормативных документов (СП, ГОСТ, СНиП, ФЗ, НПБ...).

    Используется rewriter-ом для защиты от query drift.
    """
    patterns = [
        r"(?:СП|ГОСТ|СНиП|ФЗ|НПБ|ПБ|ВНТП|ВСН|РД|ППБ)\s*Р?\s*[\d\.\-\*]+",
    ]
    ids: set[str] = set()
    for pat in patterns:
        ids.update(m.strip() for m in re.findall(pat, text, re.IGNORECASE))
    return ids
