"""V7: принятие кандидата и запись терминального вердикта в состояние.

Источник контракта: спек 2026-09-09, §4-5.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.v7.contract import OBL_ENUM, OBL_ORIGINAL, OBL_REFS, RouteDecision, Verdict

# ANCHOR: единая граница решения для simple/complex путей.
# Input: packed context + Verdict. Output: accept bool or complete state update.
# Key invariant: terminal_update always clears every stale contract field.

_EMPTY_GAP = {"kind": "unresolved_ref", "refs": [], "closed": [], "open": []}


def best_effort_allowed(
    verdict: Verdict,
    *,
    on_complex: bool,
    is_last_candidate: bool,
) -> bool:
    """Разрешить одно незакрытое обязательство последнему complex-кандидату.

    ``refs_resolved`` здесь означает, что exhaustive expansion уже не смог
    материализовать внешнюю/второстепенную ссылку. Hard gates и релевантность
    исходному запросу всё равно обязательны и проверяются в ``accept``.
    """
    if not (on_complex and is_last_candidate):
        return False
    unmet = set(verdict["obligations_unmet"])
    if unmet == {OBL_REFS}:
        return verdict["pack_status"] == "ok"
    return unmet == {OBL_ENUM}


def accept(
    final_context: List[dict],
    verdict: Verdict,
    *,
    on_complex: bool = False,
    is_last_candidate: bool = False,
) -> bool:
    """Применить единый предикат принятия на обоих путях."""
    if not final_context:
        return False
    if not verdict["hard_ok"]:
        return False
    if verdict["obligations_unmet"]:
        return best_effort_allowed(
            verdict,
            on_complex=on_complex,
            is_last_candidate=is_last_candidate,
        )
    return True


def _zero_overlap_both(verdict: Verdict) -> bool:
    details = verdict.get("details") or {}
    return (
        details.get("keyword_overlap_active", 1.0) <= 0.0
        and details.get("keyword_overlap_original", 1.0) <= 0.0
    )


def decide_simple(
    final_context: List[dict],
    verdict: Verdict,
    *,
    retrieval_error: bool,
    abstain_on_empty: bool,
) -> Tuple[RouteDecision, str]:
    """Применить таблицу решений simple-ветки; первое совпадение выигрывает.

    ``retrieval_error`` означает именно сбой retrieval. Degraded-упаковка
    блокирует refs_resolved и попадает в строку 5, а не в строку 1.
    """
    if retrieval_error:
        return "abstain", "retrieval_error"

    if not final_context:
        return (
            ("abstain", "empty_pool")
            if abstain_on_empty
            else ("complex", "empty_pool_escalated")
        )

    if _zero_overlap_both(verdict):
        return (
            ("abstain", "zero_overlap_both")
            if abstain_on_empty
            else ("complex", "zero_overlap_both_escalated")
        )

    unmet = set(verdict["obligations_unmet"])

    if OBL_ENUM in unmet:
        return "complex", "enumeration_intent"
    if OBL_REFS in unmet:
        return "complex", "refs_unresolved"
    if OBL_ORIGINAL in unmet:
        return "complex", "zero_overlap_original"

    if verdict["triage"] == "borderline":
        return "complex", "triage_borderline"
    if verdict["triage"] == "clearly_bad":
        return "complex", "triage_clearly_bad"

    if accept(final_context, verdict, on_complex=False, is_last_candidate=True):
        return "generate", "context_sufficient"
    return "complex", "triage_borderline"


def terminal_update(
    *,
    route: RouteDecision,
    reason: str,
    final_context: List[dict],
    candidate_context: List[dict],
    verdict: Optional[Verdict],
    required: Iterable[str],
    technical_failure: bool,
    fallback_snapshot: Optional[dict] = None,
    rejected: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Вернуть полный согласованный вердикт со всеми ключами контракта.

    LangGraph сливает update с прежним состоянием. Поэтому даже сбрасываемые
    поля записываются явно, иначе результат смешает данные разных узлов.
    """
    return {
        "route_decision": route,
        "route_reason": reason,
        "obligations_unmet": list((verdict or {}).get("obligations_unmet", [])),
        "obligations_required": sorted(required),
        "candidate_context": candidate_context,
        "final_context": final_context,
        "technical_failure": technical_failure,
        "fallback_snapshot": fallback_snapshot or {},
        "rejected_candidates": rejected or [],
        "sufficiency_details": (verdict or {}).get("details") or {},
        "triage_gap": (verdict or {}).get("gap") or dict(_EMPTY_GAP),
        # DEPRECATED, производные; потребители переезжают на route_decision.
        "sufficient": route == "generate",
        "final_passages": final_context if route == "generate" else [],
        "final_score": (verdict or {}).get("top_score", 0.0),
    }
