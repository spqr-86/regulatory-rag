"""V7: validate_context — чистая проверка кандидата (спек 2026-09-09, §2-3).

Ничего не расширяет, в состояние не пишет, маршрут не выбирает.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Set

from src.v7.contract import OBL_ENUM, OBL_ORIGINAL, OBL_REFS, PackStatus, Verdict
from src.v7.gap import build_gap, has_enumeration_intent
from src.v7.hard_gates import check_full_triage
from src.v7.nlp_core import passage_identity

# ANCHOR: чистая проверка уже упакованного контекста.
# Input: final context, original/active query, plan, required obligations.
# Output: Verdict без выбора маршрута и без изменения входных данных.


def required_obligations(query: str) -> Set[str]:
    """Определить обязательства для запроса.

    refs_resolved и original_query_relevant обязательны всегда: ссылка может
    появиться после расширения или в complex-кандидате, поэтому привязка к
    первичному состоянию оставила бы такой случай непроверенным.
    """
    obligations = {OBL_REFS, OBL_ORIGINAL}
    if has_enumeration_intent(query):
        obligations.add(OBL_ENUM)
    return obligations


def validate_context(
    final_context: List[dict],
    query: str,
    active_query: str,
    plan: dict,
    obligations: Iterable[str],
    *,
    pack_status: PackStatus = "ok",
    prior_context: Optional[List[dict]] = None,
) -> Verdict:
    """Посчитать hard gates и обязательства на упакованном контексте.

    ``prior_context`` — доэскалационный simple-контекст. ``None`` означает,
    что его нет; пустой список является полноценным пустым контекстом.
    """
    required = set(obligations)
    details = check_full_triage(query, active_query, final_context, plan)
    gap = build_gap(final_context)

    unmet: List[str] = []

    if OBL_REFS in required:
        if pack_status == "degraded" or gap["open"] or not final_context:
            unmet.append(OBL_REFS)

    if OBL_ORIGINAL in required:
        if details["keyword_overlap_original"] <= 0.0:
            unmet.append(OBL_ORIGINAL)

    if OBL_ENUM in required:
        baseline = final_context if prior_context is None else prior_context
        prior_ids = {passage_identity(p) for p in baseline}
        cand_ids = {passage_identity(p) for p in final_context}
        # Необходимое, не достаточное условие: без разметки обязательных
        # элементов полноту перечисления подтвердить нечем (спек §2, трек #9).
        if not final_context or cand_ids <= prior_ids:
            unmet.append(OBL_ENUM)

    return {
        "triage": details["triage"],
        "hard_ok": details["sufficient"],
        "details": details,
        "gap": gap,
        "obligations_unmet": unmet,
        "top_score": details["top_score"],
        "pack_status": pack_status,
    }
