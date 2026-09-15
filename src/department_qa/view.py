"""Pure view helpers for the department Q&A screen — no Streamlit imports.

# ANCHOR: department answer view
# Role: turn DepartmentResponse into banner kind/text and basis lines (spec §10).
# Titles and locators come from stored evidence, never from model text.
"""

from __future__ import annotations

from typing import Literal

from src.department_qa.contract import Basis, Evidence
from src.department_qa.service import DepartmentResponse

BannerKind = Literal["success", "info", "warning", "error"]

_REASON_TEXT = {
    "possible_mismatch": "Возможное расхождение ЛНА и законодательства — проверьте указанные положения.",
    "internal_evidence_missing": "В ЛНА подразделения подтверждения не найдено.",
    "external_evidence_missing": "В законодательстве подтверждения не найдено.",
    "citation_invalid": "Ответ сослался на фрагмент, которого нет среди найденных, — ссылки не прошли проверку.",
    "retrieval_failed": "Ошибка поиска по документам — это техническая ошибка, а не «документ не найден».",
    "generation_failed": "Модель не вернула ответ в нужном формате.",
    "applicability_unclear": "Уточните вопрос: ответ зависит от условий, которых в нём нет.",
}


def status_banner(response: DepartmentResponse) -> tuple[BannerKind, str]:
    reasons = " ".join(_REASON_TEXT.get(r, r) for r in response.reason_codes)
    if response.status == "answered":
        return "success", "Ответ подтверждён законодательством и ЛНА."
    if response.status == "needs_review":
        return "warning", f"Нужна проверка специалиста. {reasons}".strip()
    if response.status == "needs_context":
        return "info", reasons or "Уточните вопрос."
    if response.status == "out_of_scope":
        return "info", "Вопрос вне поддерживаемой темы (пожарная безопасность)."
    return "error", f"Не удалось получить ответ. {reasons}".strip()


def basis_lines(bases: list[Basis], evidence: list[Evidence]) -> list[str]:
    by_id = {e.id: e for e in evidence}
    lines = []
    for basis in bases:
        refs = []
        for eid in basis.evidence_ids:
            e = by_id.get(eid)
            label = e.title if e else ""
            if e and e.locator:
                label = f"{label}, {e.locator}"
            refs.append(f"[{eid}] {label}".strip())
        lines.append(f"{basis.statement} — {'; '.join(refs)}")
    return lines
