"""Pure view helpers for the department Q&A screen — no Streamlit imports.

# ANCHOR: department answer view
# Role: turn DepartmentResponse into banner kind/text, basis/object-fact/applied
#   lines and the object profile caption (spec §10, object-profile §2.0/§2.4).
# Titles and locators come from stored evidence, never from model text.
"""

from __future__ import annotations

from typing import Literal, Sequence

from src.department_qa.contract import AppliedConclusion, Basis, Evidence, ObjectFact
from src.department_qa.service import DepartmentResponse

BannerKind = Literal["success", "info", "warning", "error"]

# Spec object-profile §2.0: answered means citations were checked, not the meaning.
ANSWERED_BANNER = (
    "Ссылки сверены: законодательство и ЛНА. Смысл ответа не проверен специалистом."
)

_REASON_TEXT = {
    "possible_mismatch": "Возможное расхождение ЛНА и законодательства — проверьте указанные положения.",
    "internal_evidence_missing": "В ЛНА подразделения подтверждения не найдено.",
    "external_evidence_missing": "В законодательстве подтверждения не найдено.",
    "applied_without_norm": "Вывод об объекте сделан без нормы — нужна ссылка на закон или ЛНА.",
    "object_fact_normative": "В сведениях об объекте есть вывод о норме — его нужно проверить.",
    "object_profile_undated": "У листа объекта не указана дата заполнения — сведения могут быть устаревшими.",
    "citation_invalid": "Ответ сослался на фрагмент, которого нет среди найденных, — ссылки не прошли проверку.",
    "retrieval_failed": "Ошибка поиска по документам — это техническая ошибка, а не «документ не найден».",
    "generation_failed": "Модель не вернула ответ в нужном формате.",
    "applicability_unclear": "Уточните вопрос: ответ зависит от условий, которых в нём нет.",
    "profile_mismatch": "Сведения объекта относятся к другому подразделению — ответ не строился.",
}


def status_banner(response: DepartmentResponse) -> tuple[BannerKind, str]:
    reasons = " ".join(_REASON_TEXT.get(r, r) for r in response.reason_codes)
    if response.status == "answered":
        return "success", ANSWERED_BANNER
    if response.status == "needs_review":
        return "warning", f"Нужна проверка специалиста. {reasons}".strip()
    if response.status == "needs_context":
        return "info", reasons or "Уточните вопрос."
    if response.status == "out_of_scope":
        return "info", "Вопрос вне поддерживаемой темы (пожарная безопасность)."
    return "error", f"Не удалось получить ответ. {reasons}".strip()


def basis_lines(
    items: Sequence[Basis | ObjectFact | AppliedConclusion], evidence: list[Evidence]
) -> list[str]:
    by_id = {e.id: e for e in evidence}
    lines = []
    for item in items:
        refs = []
        for eid in item.evidence_ids:
            e = by_id.get(eid)
            label = e.title if e else ""
            if e and e.locator:
                label = f"{label}, {e.locator}"
            refs.append(f"[{eid}] {label}".strip())
        lines.append(f"{item.statement} — {'; '.join(refs)}")
    return lines


def profile_caption(response: DepartmentResponse) -> str:
    if response.profile_sha256 is None:
        return ""
    sheet = f"лист {response.profile_sha256[:8]}"
    if response.profile_as_of_date is None:
        return f"Сведения объекта: дата заполнения не указана · {sheet}"
    return f"Сведения объекта на {response.profile_as_of_date:%d.%m.%Y} · {sheet}"
