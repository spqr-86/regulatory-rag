"""Department answer contract: model output schema, citation check, status rule.

# ANCHOR: department answer contract
# Role: validate what the model cites and fold it into one status (spec §8, §9).
# Input: ModelAnswer (structured output) + evidence by id (ext_* / int_*).
# Output: (status, reason_codes). Invalid citation → failed/citation_invalid,
#   no regeneration (spec П6).
# Simplified rule for the MVP slice instead of the full §8.1 v7 mapping.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["external", "internal"]
Status = Literal["answered", "needs_context", "needs_review", "out_of_scope", "failed"]

_PREFIX: dict[str, Level] = {"ext_": "external", "int_": "internal"}


class Evidence(BaseModel):
    id: str
    level: Level
    text: str
    source: str
    title: str = ""
    locator: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[int] = None


class Basis(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)


class ObjectSection(BaseModel):
    """One section of the unit object sheet, raw text as written (spec object-profile §2.1)."""

    id: str  # obj_s1 … obj_s9
    number: int
    title: str
    text: str = ""
    presence: Literal["present", "empty", "missing"]


class ModelAnswer(BaseModel):
    """Structured output requested from the model."""

    answer: str
    external_basis: list[Basis] = Field(default_factory=list)
    internal_basis: list[Basis] = Field(default_factory=list)
    possible_mismatch: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    out_of_scope: bool = False


class PromptVars(BaseModel):
    """Only way to render prompts/agents/department_answer_v1.j2."""

    question: str
    unit_label: str
    external_evidence: list[Evidence]
    internal_evidence: list[Evidence]


def check_citations(answer: ModelAnswer, evidence: dict[str, Evidence]) -> list[str]:
    problems: list[str] = []
    for level, bases in (
        ("external", answer.external_basis),
        ("internal", answer.internal_basis),
    ):
        for basis in bases:
            if not basis.evidence_ids:
                problems.append(f"{level} basis without evidence: {basis.statement!r}")
            for eid in basis.evidence_ids:
                found = evidence.get(eid)
                if found is None:
                    problems.append(f"unknown evidence id {eid}")
                elif found.level != level:
                    problems.append(f"level mismatch: {eid} cited as {level}")
    return problems


def decide(
    answer: ModelAnswer, evidence: dict[str, Evidence]
) -> tuple[Status, list[str]]:
    if answer.out_of_scope:
        return "out_of_scope", []
    if check_citations(answer, evidence):
        return "failed", ["citation_invalid"]
    has_ext, has_int = bool(answer.external_basis), bool(answer.internal_basis)
    if answer.clarifying_questions and not (has_ext and has_int):
        return "needs_context", ["applicability_unclear"]

    reasons: list[str] = []
    if not has_ext:
        reasons.append("external_evidence_missing")
    if not has_int:
        reasons.append("internal_evidence_missing")
    if answer.possible_mismatch and has_ext and has_int:
        reasons.append("possible_mismatch")
    return ("needs_review" if reasons else "answered"), reasons
