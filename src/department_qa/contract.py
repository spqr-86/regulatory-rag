"""Department answer contract: model output schema, citation check, status rule.

# ANCHOR: department answer contract
# Role: validate what the model cites and fold it into one status
#   (spec department-qa §8, §9; object-profile §2.3).
# Input: ModelAnswer(V1) (structured output) + evidence by id (ext_* / int_* / obj_s*)
#   + profile fill date.
# Output: (status, reason_codes). Invalid citation → failed/citation_invalid, no
#   regeneration (spec П6). answered = citations checked, not meaning checked (§2.0).
# The schema checks types only; every invariant on evidence_ids lives in decide(),
#   so a bad model answer gets its status instead of generation_failed.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field

Level = Literal["external", "internal", "object"]
Status = Literal["answered", "needs_context", "needs_review", "out_of_scope", "failed"]

_PREFIX: dict[str, Level] = {"ext_": "external", "int_": "internal", "obj_": "object"}
_NORM_LEVELS = {"external", "internal"}

# Closed list of stems (spec object-profile §2.3): a heuristic guard against a
# normative conclusion passed off as an object fact, not a classifier.
NORMATIVE_MARKERS = re.compile(
    r"(?<!\w)(?:требуется|необходим|должен(?!\w)|должн[аоыуе](?!\w)|достаточн"
    r"|допуска|запрещ|обязан|не\s+снижа|не\s+менее|не\s+более)",
    re.IGNORECASE,
)


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


class TypedFieldLine(BaseModel):
    """One canonical object field rendered with its existing section evidence id."""

    section_id: str
    label: str
    value: str


class ObjectFact(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)  # prompt: obj_* only


class AppliedConclusion(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(
        default_factory=list
    )  # prompt: ≥1 obj_* and ≥1 ext_*/int_*


class ModelAnswerV1(BaseModel):
    """Structured output of prompt v1 (no object profile)."""

    answer: str
    external_basis: list[Basis] = Field(default_factory=list)
    internal_basis: list[Basis] = Field(default_factory=list)
    possible_mismatch: bool = False
    clarifying_questions: list[str] = Field(default_factory=list)
    out_of_scope: bool = False


class ModelAnswer(ModelAnswerV1):
    """Structured output of prompt v2: norms, object facts, applied conclusions."""

    object_facts: list[ObjectFact] = Field(default_factory=list)
    applied_conclusions: list[AppliedConclusion] = Field(default_factory=list)


class PromptVars(BaseModel):
    """Only way to render department_answer prompts; v1 ignores object fields."""

    question: str
    unit_label: str
    external_evidence: list[Evidence]
    internal_evidence: list[Evidence]
    object_label: str = ""
    object_sections: list[ObjectSection] = Field(default_factory=list)
    typed_fields: list[TypedFieldLine] = Field(default_factory=list)


def _facts(answer: ModelAnswerV1) -> list[ObjectFact]:
    return getattr(answer, "object_facts", [])


def _applied(answer: ModelAnswerV1) -> list[AppliedConclusion]:
    return getattr(answer, "applied_conclusions", [])


def cited_ids(answer: ModelAnswerV1) -> set[str]:
    items = (
        answer.external_basis
        + answer.internal_basis
        + _facts(answer)
        + _applied(answer)
    )
    return {eid for item in items for eid in item.evidence_ids}


def check_citations(answer: ModelAnswerV1, evidence: dict[str, Evidence]) -> list[str]:
    problems: list[str] = []
    groups = (
        ("external_basis", answer.external_basis, {"external"}),
        ("internal_basis", answer.internal_basis, {"internal"}),
        ("object_facts", _facts(answer), {"object"}),
        ("applied_conclusions", _applied(answer), {"external", "internal", "object"}),
    )
    for name, items, allowed in groups:
        for item in items:
            if not item.evidence_ids:
                problems.append(f"{name} item without evidence: {item.statement!r}")
            for eid in item.evidence_ids:
                found = evidence.get(eid)
                if found is None:
                    problems.append(f"unknown evidence id {eid}")
                elif found.level not in allowed:
                    problems.append(f"level mismatch: {eid} cited in {name}")
    for item in _applied(answer):
        levels = {evidence[e].level for e in item.evidence_ids if e in evidence}
        if item.evidence_ids and "object" not in levels:
            problems.append(
                f"applied conclusion without object evidence: {item.statement!r}"
            )
    return problems


def decide(
    answer: ModelAnswerV1,
    evidence: dict[str, Evidence],
    profile_as_of: Optional[date] = None,
) -> tuple[Status, list[str]]:
    if answer.out_of_scope:
        return "out_of_scope", []
    if check_citations(answer, evidence):
        return "failed", ["citation_invalid"]
    if answer.clarifying_questions:
        return "needs_context", ["applicability_unclear"]

    facts, applied = _facts(answer), _applied(answer)
    applied_levels = [{evidence[e].level for e in c.evidence_ids} for c in applied]
    has_ext = bool(answer.external_basis) or any(
        "external" in lv for lv in applied_levels
    )
    has_int = bool(answer.internal_basis) or any(
        "internal" in lv for lv in applied_levels
    )
    # After check_citations: facts cite only obj_*, every applied cites at least one obj_*.
    cites_object = bool(facts or applied)
    fact_only = bool(facts) and not applied and not has_ext and not has_int

    reasons: list[str] = []
    if any(not (lv & _NORM_LEVELS) for lv in applied_levels):
        reasons.append("applied_without_norm")
    if any(NORMATIVE_MARKERS.search(f.statement) for f in facts):
        reasons.append("object_fact_normative")
    if not fact_only and not has_ext:
        reasons.append("external_evidence_missing")
    if not fact_only and not has_int:
        reasons.append("internal_evidence_missing")
    if cites_object and profile_as_of is None:
        reasons.append("object_profile_undated")
    if answer.possible_mismatch and has_ext and has_int:
        reasons.append("possible_mismatch")
    return ("needs_review" if reasons else "answered"), reasons
