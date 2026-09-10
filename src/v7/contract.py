"""V7: терминальный контракт решения (спек 2026-09-09).

Только типы и имена: ни узлов, ни LangGraph, ни логики.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TypedDict

from src.v7.state_types import (
    RetrievalPlan,
    SufficiencyResult,
    TriageCategory,
    TriageGap,
)

RouteDecision = Literal["generate", "complex", "abstain"]
PackStatus = Literal["ok", "degraded"]

# ─── Обязательства: ровно три, других не заводим ───────────────────────────
OBL_REFS = "refs_resolved"
OBL_ORIGINAL = "original_query_relevant"
OBL_ENUM = "enumeration_complete"
ALL_OBLIGATIONS = {OBL_REFS, OBL_ORIGINAL, OBL_ENUM}

# ─── Коды причин. retrieval_error — только simple-ветка ────────────────────
SIMPLE_REASONS = {
    "retrieval_error",
    "empty_pool",
    "zero_overlap_both",
    "empty_pool_escalated",
    "zero_overlap_both_escalated",
    "enumeration_intent",
    "refs_unresolved",
    "zero_overlap_original",
    "triage_borderline",
    "triage_clearly_bad",
    "context_sufficient",
}
COMPLEX_REASONS = {
    "complex_sufficient",
    "complex_fallback_accepted",
    "enumeration_best_effort",
    "complex_exhausted",
}

CandidateOrigin = Literal["simple", "merged", "last_attempt", "fallback_snapshot"]


class Candidate(TypedDict, total=False):
    """Вход конвейера.

    План и active_query едут вместе с пассажами: проверять simple-fallback
    под последним complex-планом нельзя (спек §5).

    packed=True означает, что пассажи УЖЕ прошли enrich и pack на своей
    ветке. Для такого кандидата enrich/pack — тождественная операция:
    повторный прогон расширил бы уже расширенное, нарушив инвариант
    «expander ровно один раз на версию кандидата».
    """

    passages: List[dict]
    plan: RetrievalPlan
    active_query: str
    origin: CandidateOrigin
    packed: bool
    pack_status: PackStatus


class PackResult(TypedDict):
    """Выход pack_context. degraded — сбой расширения, не содержательный результат."""

    final_context: List[dict]
    status: PackStatus
    dropped: int


class Verdict(TypedDict):
    """Выход validate_context. Ничего не решает о маршруте."""

    triage: TriageCategory
    hard_ok: bool
    details: Optional[SufficiencyResult]
    gap: TriageGap
    obligations_unmet: List[str]
    top_score: float
    pack_status: PackStatus


class RejectedCandidate(TypedDict):
    """Диагностика отклонённого кандидата — для телеметрии, не для вердикта."""

    origin: str
    obligations_unmet: List[str]
    triage: str
    passages: int
    pack_status: PackStatus
