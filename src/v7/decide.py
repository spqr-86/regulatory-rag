"""V7: принятие кандидата и запись терминального вердикта в состояние.

Источник контракта: спек 2026-09-09, §4-5.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.v7.contract import OBL_ENUM, RouteDecision, Verdict

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
    """Разрешить исключение только для enumeration последнего complex-кандидата."""
    if not (on_complex and is_last_candidate):
        return False
    return set(verdict["obligations_unmet"]) <= {OBL_ENUM}


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
    unmet = set(verdict["obligations_unmet"])
    if unmet - {OBL_ENUM}:
        return False
    if OBL_ENUM in unmet:
        return best_effort_allowed(
            verdict,
            on_complex=on_complex,
            is_last_candidate=is_last_candidate,
        )
    return True


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
