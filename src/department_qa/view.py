"""Pure view helpers for the department Q&A screen — no Streamlit imports.

# ANCHOR: department answer view
# Role: turn DepartmentResponse into the portfolio presentation model (spec
#   streamlit-portfolio-demo §5, §9, §14, §15, §18): one UI status, compact
#   profile, reasoning chains, evidence cards and technical rows. Also keeps the
#   earlier banner/basis helpers used by the department page (spec department-qa
#   §10, object-profile §2.0/§2.4).
# Titles and locators come from stored evidence, never from model text; technical
#   reason codes never reach the user-facing title/detail.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from src.department_qa.contract import (
    AppliedConclusion,
    Basis,
    Evidence,
    Level,
    ObjectFact,
    Status,
)
from src.department_qa.object_profile import ObjectProfile
from src.department_qa.service import DepartmentResponse

BannerKind = Literal["success", "info", "warning", "error"]

# UI status shown to the visitor; backend Status stays unchanged (spec §5).
UiStatus = Literal[
    "sufficient",
    "clarification_required",
    "insufficient_evidence",
    "conflict",
    "out_of_scope",
    "failed",
]
Tone = Literal["success", "info", "warning", "error"]

# Spec §5 / §32.5: reason-code priority is fixed here and covered by tests.
_CONFLICT_REASONS = frozenset({"possible_mismatch", "verification_contradiction"})

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
    "applied_on_unknown_field": "Вывод опирается на неизвестное значение сведений об объекте — нужны уточнения.",
    "citation_invalid": "Ответ сослался на фрагмент, которого нет среди найденных, — ссылки не прошли проверку.",
    "retrieval_failed": "Ошибка поиска по документам — это техническая ошибка, а не «документ не найден».",
    "generation_failed": "Модель не вернула ответ в нужном формате.",
    "applicability_unclear": "Уточните вопрос: ответ зависит от условий, которых в нём нет.",
    "profile_mismatch": "Сведения объекта относятся к другому подразделению — ответ не строился.",
}

_STATUS_TEXT: dict[UiStatus, tuple[Tone, str, str]] = {
    "sufficient": (
        "success",
        "Доказательств достаточно",
        "Ответ основан на нормативных источниках, локальных актах и данных подразделения.",
    ),
    "clarification_required": (
        "warning",
        "Нужно уточнение",
        "Для проверки применимости требования не хватает данных об объекте.",
    ),
    "insufficient_evidence": (
        "warning",
        "Недостаточно оснований для надёжного ответа",
        "Система не нашла достаточно подтверждений и не стала формировать уверенный вывод.",
    ),
    "conflict": (
        "error",
        "Обнаружено расхождение",
        "Требуется проверка специалистом.",
    ),
    "out_of_scope": (
        "info",
        "Запрос вне области базы знаний",
        "Доступные источники не покрывают этот вопрос.",
    ),
    "failed": (
        "error",
        "Не удалось получить ответ",
        "Это техническая ошибка, а не отсутствие документа.",
    ),
}


class PresentationStatus(BaseModel):
    """One user-facing evidence status derived from the backend response."""

    code: UiStatus
    tone: Tone
    title: str
    detail: str


class ReasoningChain(BaseModel):
    """Fact -> requirement -> conclusion, built from an applied conclusion."""

    fact: str
    requirement: str
    conclusion: str


class EvidenceCard(BaseModel):
    """One cited source as shown in the sources expander (spec §18)."""

    id: str
    title: str
    source_type: Level
    locator: Optional[str] = None
    excerpt: str = ""
    retrieval_score: Optional[float] = None


class BasisCard(BaseModel):
    """One basis/object fact under "На чём основан ответ" (spec §16).

    The statement is the model's claim; title and locator are resolved from the
    stored evidence it cites, never from model text.
    """

    statement: str
    title: str = ""
    locator: Optional[str] = None


class ClarificationFallback(BaseModel):
    """What the screen shows when the answer needs context (spec §13.2).

    Instead of abstaining silently, the screen names what the norms say
    (``requirements``), what is already known about the object (``facts``) and
    what has to be clarified for a conclusion (``questions``). The model draft
    and its applied conclusions are deliberately absent: they may rest on facts
    the sheet marks unknown.
    """

    requirements: list[BasisCard] = Field(default_factory=list)
    facts: list[BasisCard] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)


def _ui_status(status: Status, reason_codes: Sequence[str]) -> UiStatus:
    if status == "answered":
        return "sufficient"
    if status == "out_of_scope":
        return "out_of_scope"
    if status == "needs_context":
        return "clarification_required"
    if status == "failed":
        return "failed"
    # needs_review: a real conflict outranks a missing source.
    if _CONFLICT_REASONS.intersection(reason_codes):
        return "conflict"
    return "insufficient_evidence"


def _insufficient_detail(reason_codes: Sequence[str]) -> str:
    if "internal_evidence_missing" in reason_codes:
        return "Не найден локальный акт, регулирующий этот вопрос."
    if "external_evidence_missing" in reason_codes:
        return "В законодательстве не найдено подтверждения требования."
    if "applied_without_norm" in reason_codes:
        return "Вывод об объекте сделан без ссылки на норму."
    return _STATUS_TEXT["insufficient_evidence"][2]


def presentation_status(response: DepartmentResponse) -> PresentationStatus:
    """Backend status + reason_codes -> one UI status with user-facing text."""
    code = _ui_status(response.status, response.reason_codes)
    tone, title, detail = _STATUS_TEXT[code]
    if code == "insufficient_evidence":
        detail = _insufficient_detail(response.reason_codes)
    return PresentationStatus(code=code, tone=tone, title=title, detail=detail)


def _plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(int(n)) % 100
    if 11 <= n <= 19:
        return many
    n %= 10
    if n == 1:
        return one
    if 2 <= n <= 4:
        return few
    return many


def _known(value: object) -> bool:
    return value not in ("unknown", "not_applicable")


def unit_name(title: str, unit_id: str = "") -> str:
    """Short human-readable unit name from the object sheet title (spec §8).

    ``"Лист особенностей объекта защиты — офис (демонстрационные данные)"`` ->
    ``"Офис"``. Falls back to ``unit_id`` when the title carries no name.
    """
    name = title.split("—", 1)[1] if "—" in title else title
    name = name.split("(", 1)[0].strip(" .—-")
    if not name:
        return unit_id
    return name[0].upper() + name[1:]


def compact_profile(profile: Optional[ObjectProfile]) -> list[str]:
    """The few typed fields that explain a unit at a glance (spec §9).

    Unknown and not-applicable fields are omitted: they add noise, not context.
    """
    if profile is None:
        return []
    f = profile.typed_fields
    out: list[str] = []

    if _known(f.people_in_object_zone):
        n = f.people_in_object_zone
        out.append(f"{n} {_plural(n, 'сотрудник', 'сотрудника', 'сотрудников')}")
    if _known(f.people_on_floor_total):
        n = f.people_on_floor_total
        out.append(f"на этаже {n} {_plural(n, 'человек', 'человека', 'человек')}")
    if _known(f.people_in_building_total):
        n = f.people_in_building_total
        out.append(f"в здании {n} {_plural(n, 'человек', 'человека', 'человек')}")
    if _known(f.permanent_workplaces_on_floor):
        n = f.permanent_workplaces_on_floor
        out.append(
            f"{n} {_plural(n, 'рабочее место', 'рабочих места', 'рабочих мест')} на этаже"
        )
    if _known(f.evacuation_plan_present):
        out.append(
            "план эвакуации есть"
            if f.evacuation_plan_present
            else "план эвакуации не разработан"
        )
    if _known(f.room_categories):
        out.append("категории помещений: " + "; ".join(f.room_categories))
    if _known(f.aupt_present):
        out.append("АУПТ есть" if f.aupt_present else "АУПТ отсутствует")
    if _known(f.extinguishers_total):
        n = f.extinguishers_total
        out.append(f"{n} {_plural(n, 'огнетушитель', 'огнетушителя', 'огнетушителей')}")
    if _known(f.outside_ladder_last_test_date):
        out.append(f"лестница испытана {f.outside_ladder_last_test_date:%d.%m.%Y}")
    return out


def _statements_by_evidence(
    items: Sequence[Basis | ObjectFact],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        for eid in item.evidence_ids:
            out.setdefault(eid, item.statement)
    return out


def _compose(
    ids: Sequence[str], statements: dict[str, str], by_id: dict[str, Evidence]
) -> str:
    parts: list[str] = []
    for eid in ids:
        text = statements.get(eid)
        if text is None:
            found = by_id.get(eid)
            text = found.text if found else ""
        if text:
            parts.append(text)
    return " · ".join(parts)


def build_reasoning_chains(response: DepartmentResponse) -> list[ReasoningChain]:
    """Applied conclusions -> Fact / Requirement / Conclusion, deterministically.

    Object ids map to the citing object-fact statement (fallback: evidence text,
    e.g. a typed ``obj_f_*`` field); norm ids map to the citing basis statement
    (fallback: evidence text).
    """
    by_id = {e.id: e for e in response.evidence}
    fact_statements = _statements_by_evidence(response.object_facts)
    norm_statements = _statements_by_evidence(
        response.external_basis + response.internal_basis
    )
    chains: list[ReasoningChain] = []
    for applied in response.applied_conclusions:
        obj_ids = [e for e in applied.evidence_ids if e.startswith("obj_")]
        norm_ids = [e for e in applied.evidence_ids if e.startswith(("ext_", "int_"))]
        chains.append(
            ReasoningChain(
                fact=_compose(obj_ids, fact_statements, by_id),
                requirement=_compose(norm_ids, norm_statements, by_id),
                conclusion=applied.statement,
            )
        )
    return chains


def evidence_cards(response: DepartmentResponse) -> list[EvidenceCard]:
    """Cited sources with type, locator, excerpt and retrieval score (spec §18)."""
    return [
        EvidenceCard(
            id=e.id,
            title=e.title or e.source,
            source_type=e.level,
            locator=e.locator,
            excerpt=e.text,
            retrieval_score=e.retrieval_score,
        )
        for e in response.evidence
    ]


def technical_details(
    response: DepartmentResponse,
    *,
    model_name: str = "",
    latency_s: Optional[float] = None,
) -> list[tuple[str, str]]:
    """Label/value rows for the collapsed technical block (spec §14).

    Only values that are actually known are returned; routing and cost are
    deliberately absent (department_qa has no route; cost is P2).
    """
    rows: list[tuple[str, str]] = []
    if model_name:
        rows.append(("LLM", model_name))
    rows.append(("Retrieval", "dense + BM25 → RRF"))
    n = len(response.evidence)
    rows.append(
        ("Used as evidence", f"{n} {_plural(n, 'фрагмент', 'фрагмента', 'фрагментов')}")
    )
    if latency_s is not None:
        rows.append(("Latency", f"{latency_s:.1f} с"))
    rows.append(("Trace ID", response.trace_id))
    return rows


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


def basis_cards(
    items: Sequence[Basis | ObjectFact | AppliedConclusion],
    evidence: Sequence[Evidence],
) -> list[BasisCard]:
    """Basis/fact items -> statement with each cited source's title and locator.

    Titles and locators come from stored evidence, never from model text. An item
    citing several sources yields one card per source; an item citing nothing
    yields a statement-only card. An object fact yields exactly one card: it
    usually cites both a typed field (``obj_f_*``) and its section (``obj_s*``),
    and the typed field wins.
    """
    by_id = {e.id: e for e in evidence}
    cards: list[BasisCard] = []
    for item in items:
        refs = [by_id[eid] for eid in item.evidence_ids if eid in by_id]
        if not refs:
            cards.append(BasisCard(statement=item.statement))
            continue
        if isinstance(item, ObjectFact):
            typed = [ref for ref in refs if ref.id.startswith("obj_f_")]
            refs = (typed or refs)[:1]
        for ref in refs:
            cards.append(
                BasisCard(
                    statement=item.statement,
                    title=ref.title or ref.source,
                    locator=ref.locator,
                )
            )
    return cards


def clarification_fallback(response: DepartmentResponse) -> ClarificationFallback:
    """Norm requirements, known object facts and open questions (spec §13.2).

    Sources for requirements and facts are resolved from stored evidence, as in
    :func:`basis_cards`; ``questions`` are passed through verbatim.
    """
    return ClarificationFallback(
        requirements=basis_cards(
            response.external_basis + response.internal_basis, response.evidence
        ),
        facts=basis_cards(response.object_facts, response.evidence),
        questions=list(response.clarifying_questions),
    )


def profile_caption(response: DepartmentResponse) -> str:
    if response.profile_sha256 is None:
        return ""
    sheet = f"лист {response.profile_sha256[:8]}"
    if response.profile_as_of_date is None:
        return f"Сведения объекта: дата заполнения не указана · {sheet}"
    return f"Сведения объекта на {response.profile_as_of_date:%d.%m.%Y} · {sheet}"
