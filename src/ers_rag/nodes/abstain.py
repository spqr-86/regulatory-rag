"""V7 node: abstain — honest refusal with diagnostics."""

from __future__ import annotations

from typing import List

from src.ers_rag.state_types import RAGState


def abstain(state: RAGState) -> RAGState:
    """Honest refusal with detailed diagnostics."""
    details = state.get("sufficiency_details")
    query = state.get("query", "")
    active_q = state.get("active_query", query)
    attempts = state.get("retrieval_attempts") or []
    plan = state.get("plan") or {}
    verification = state.get("verification")
    iteration = state.get("verify_iteration", 0)

    reasons: List[str] = []

    if details:
        if not details["above_threshold"]:
            reasons.append(f"лучший score ({details['top_score']:.3f}) ниже порога")
        if not details["enough_evidence"]:
            reasons.append(
                f"найдено {details['passage_count']} фрагментов, требуется больше"
            )
        if not details["keyword_overlap_ok"]:
            min_kw = plan.get("min_keyword_overlap", 0.0)
            reasons.append(
                f"keyword overlap: active={details['keyword_overlap_active']:.0%} "
                f"(порог {min_kw:.0%}), "
                f"original={details['keyword_overlap_original']:.0%}"
            )

    if verification:
        reasons.append(f"LLM-верификатор: {verification.get('reason', '—')}")
        missing = verification.get("missing_aspects", [])
        if missing:
            reasons.append(f"недостающие аспекты: {', '.join(missing)}")

    if iteration > 0:
        reasons.append(f"выполнено {iteration} переформулировок")
        if active_q != query:
            reasons.append(f'последний запрос: "{active_q[:60]}"')

    if not reasons:
        reasons.append("контекст недостаточен для ответа")

    return {
        "abstain_reason": (
            f'Не удалось найти контекст для: "{query[:80]}". '
            f"Причины: {'; '.join(reasons)}. "
            f"Попыток: {len(attempts)}. "
            f"Рекомендация: уточните запрос или укажите номер документа."
        ),
    }
