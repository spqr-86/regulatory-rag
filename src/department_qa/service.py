"""Department Q&A flow: two scoped searches → one prompt → checked structured answer.

# ANCHOR: department answer service
# Role: vertical slice of spec §4–§9 for the Streamlit screen and eval.
# Input: question, unit_id (or None), search_fn(query, filters, top_k) -> passages,
#   model_fn(prompt) -> ModelAnswer (structured output, validated by schema).
# Output: DepartmentResponse (spec §8 contract) with evidence resolved from stored
#   chunks, never from model text.
# Failure modes: retrieval error → failed/retrieval_failed (not "not found");
#   model or schema error → failed/generation_failed; bad citation →
#   failed/citation_invalid with bases hidden (spec П6, no regeneration).
"""

from __future__ import annotations

import uuid
from typing import Callable, Optional

import structlog
from pydantic import BaseModel, Field

from src.department_qa.contract import (
    Basis,
    Evidence,
    ModelAnswer,
    PromptVars,
    Status,
    decide,
)
from src.infra.prompt_manager import PromptManager
from src.v7.scope_filter import build_scope_filters

logger = structlog.get_logger()

SearchFn = Callable[..., list[dict]]
ModelFn = Callable[[str], ModelAnswer]

PROMPT_ID = "department_answer"
TOP_K_PER_LEVEL = 8

_NEXT_STEP: dict[str, str] = {
    "answered": "",
    "needs_context": "Уточните вопрос и повторите запрос.",
    "needs_review": "Передать специалисту по пожарной безопасности для проверки указанных положений.",
    "out_of_scope": "Вопрос вне поддерживаемой темы (пожарная безопасность).",
    "failed": "Повторите запрос позже; при повторе ошибки обратитесь к специалисту.",
}


class DepartmentResponse(BaseModel):
    answer: str = ""
    external_basis: list[Basis] = Field(default_factory=list)
    internal_basis: list[Basis] = Field(default_factory=list)
    status: Status
    reason_codes: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    next_step: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    trace_id: str
    snapshot_id: Optional[str] = None
    unit_id: Optional[str] = None


def _to_evidence(passages: list[dict], prefix: str, level) -> list[Evidence]:
    out = []
    for i, p in enumerate(passages, 1):
        meta = p.get("metadata") or {}
        out.append(
            Evidence(
                id=f"{prefix}_{i:03d}",
                level=level,
                text=p.get("text", ""),
                source=meta.get("source", ""),
                title=meta.get("title") or meta.get("source", ""),
                locator=meta.get("locator") or meta.get("parent_section"),
                document_id=meta.get("document_id"),
                chunk_id=meta.get("chunk_id", p.get("chunk_id")),
            )
        )
    return out


def answer_question(
    question: str,
    unit_id: Optional[str],
    search_fn: SearchFn,
    model_fn: ModelFn,
    snapshot_id: Optional[str] = None,
    prompts: Optional[PromptManager] = None,
) -> DepartmentResponse:
    trace_id = uuid.uuid4().hex
    base = {"trace_id": trace_id, "snapshot_id": snapshot_id, "unit_id": unit_id}

    def _fail(reason: str) -> DepartmentResponse:
        return DepartmentResponse(
            status="failed",
            reason_codes=[reason],
            next_step=_NEXT_STEP["failed"],
            **base,
        )

    ext_filter, int_filter = build_scope_filters(unit_id)
    try:
        external = _to_evidence(
            search_fn(question, filters=ext_filter, top_k=TOP_K_PER_LEVEL),
            "ext",
            "external",
        )
        internal = _to_evidence(
            search_fn(question, filters=int_filter, top_k=TOP_K_PER_LEVEL),
            "int",
            "internal",
        )
    except Exception as exc:
        logger.warning(
            "department_qa.retrieval_failed", trace_id=trace_id, error=str(exc)
        )
        return _fail("retrieval_failed")

    evidence = {e.id: e for e in external + internal}
    prompt_vars = PromptVars(
        question=question,
        unit_label=(
            f"Подразделение: {unit_id}"
            if unit_id
            else "Подразделение не указано: доступны только общекорпоративные документы."
        ),
        external_evidence=external,
        internal_evidence=internal,
    )
    try:
        prompt = (prompts or PromptManager()).render(
            PROMPT_ID, **prompt_vars.model_dump()
        )
        model_answer = model_fn(prompt)
    except Exception as exc:
        logger.warning(
            "department_qa.generation_failed", trace_id=trace_id, error=str(exc)
        )
        return _fail("generation_failed")

    status, reasons = decide(model_answer, evidence)
    logger.info(
        "department_qa.answer",
        trace_id=trace_id,
        snapshot_id=snapshot_id,
        status=status,
        reason_codes=reasons,
        n_external=len(external),
        n_internal=len(internal),
    )
    if reasons == ["citation_invalid"]:
        return _fail("citation_invalid")

    cited = {
        eid
        for basis in model_answer.external_basis + model_answer.internal_basis
        for eid in basis.evidence_ids
    }
    return DepartmentResponse(
        answer=model_answer.answer,
        external_basis=model_answer.external_basis,
        internal_basis=model_answer.internal_basis,
        status=status,
        reason_codes=reasons,
        clarifying_questions=model_answer.clarifying_questions,
        next_step=_NEXT_STEP[status],
        evidence=[e for e in external + internal if e.id in cited]
        or external + internal,
        **base,
    )
