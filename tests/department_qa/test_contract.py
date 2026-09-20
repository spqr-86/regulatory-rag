"""Tests for the department answer contract: citation check and status rule (spec §8, §9)."""

from __future__ import annotations

from datetime import date

import pytest

from src.department_qa.contract import (
    NORMATIVE_MARKERS,
    AppliedConclusion,
    Basis,
    Evidence,
    ModelAnswer,
    ModelAnswerV1,
    ObjectFact,
    RequestContext,
    applied_on_unknown_fields,
    check_citations,
    cited_ids,
    decide,
)

EVIDENCE = {
    "ext_001": Evidence(
        id="ext_001", level="external", text="п. 60 ППР", source="ppr.pdf"
    ),
    "int_001": Evidence(
        id="int_001", level="internal", text="осмотр раз в квартал", source="pril3.md"
    ),
    "obj_s3": Evidence(id="obj_s3", level="object", text="АПС есть", source="list.md"),
    "obj_f_people_in_object_zone": Evidence(
        id="obj_f_people_in_object_zone",
        level="object",
        text="Людей в зоне объекта: 8",
        source="list.md",
        field_state="known",
    ),
    "obj_f_people_in_building_total": Evidence(
        id="obj_f_people_in_building_total",
        level="object",
        text="Людей в здании всего: неизвестно",
        source="list.md",
        field_state="unknown",
    ),
}
DATED = date(2026, 9, 1)


@pytest.mark.unit
def test_request_context_uses_stable_corpus_order():
    context = RequestContext(corpora=("internal", "external"))
    assert context.corpora == ("external", "internal")


@pytest.mark.unit
def test_request_context_rejects_empty_or_duplicate_corpora():
    with pytest.raises(ValueError, match="at least one"):
        RequestContext(corpora=())
    with pytest.raises(ValueError, match="duplicates"):
        RequestContext(corpora=("external", "external"))


@pytest.mark.unit
def test_scoped_decision_requires_basis_for_requested_corpus_even_for_object_facts():
    answer = ModelAnswer(
        answer="На объекте есть АПС.",
        object_facts=[ObjectFact(statement="Есть АПС", evidence_ids=["obj_s3"])],
    )
    assert decide(
        answer,
        EVIDENCE,
        profile_as_of=DATED,
        requested_corpora=("external",),
        include_object_profile=True,
        require_corpus_basis=True,
    ) == ("needs_review", ["external_evidence_missing"])


def _answer(ext=("ext_001",), internal=("int_001",), **kw) -> ModelAnswer:
    return ModelAnswer(
        answer="черновик",
        external_basis=(
            [Basis(statement="закон", evidence_ids=list(ext))] if ext else []
        ),
        internal_basis=(
            [Basis(statement="ЛНА", evidence_ids=list(internal))] if internal else []
        ),
        **kw,
    )


@pytest.mark.unit
def test_valid_citations_pass():
    assert check_citations(_answer(), EVIDENCE) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer, problem",
    [
        (_answer(ext=("ext_404",)), "unknown"),
        (_answer(ext=("int_001",)), "level"),
        (_answer(internal=("ext_001",)), "level"),
    ],
    ids=["unknown-id", "internal-cited-as-law", "law-cited-as-lna"],
)
def test_invalid_citations_reported(answer, problem):
    problems = check_citations(answer, EVIDENCE)
    assert problems and problem in problems[0]


@pytest.mark.unit
def test_basis_without_evidence_ids_is_invalid():
    answer = ModelAnswer(
        answer="x", external_basis=[Basis(statement="закон", evidence_ids=[])]
    )
    assert check_citations(answer, EVIDENCE)


@pytest.mark.unit
def test_both_levels_answered():
    assert decide(_answer(), EVIDENCE) == ("answered", [])


@pytest.mark.unit
def test_invalid_citation_fails_without_retry():
    assert decide(_answer(ext=("ext_404",)), EVIDENCE) == (
        "failed",
        ["citation_invalid"],
    )


@pytest.mark.unit
def test_missing_internal_basis_needs_review():
    assert decide(_answer(internal=()), EVIDENCE) == (
        "needs_review",
        ["internal_evidence_missing"],
    )


@pytest.mark.unit
def test_missing_external_basis_needs_review():
    assert decide(_answer(ext=()), EVIDENCE) == (
        "needs_review",
        ["external_evidence_missing"],
    )


@pytest.mark.unit
def test_mismatch_with_both_sides_needs_review():
    assert decide(_answer(possible_mismatch=True), EVIDENCE) == (
        "needs_review",
        ["possible_mismatch"],
    )


@pytest.mark.unit
def test_mismatch_without_both_sides_is_not_reported():
    # Spec §7: possible_mismatch only with concrete grounds on both sides.
    status, reasons = decide(_answer(internal=(), possible_mismatch=True), EVIDENCE)
    assert "possible_mismatch" not in reasons
    assert status == "needs_review"


@pytest.mark.unit
def test_clarifying_questions_need_context():
    answer = _answer(ext=(), internal=(), clarifying_questions=["Какое подразделение?"])
    assert decide(answer, EVIDENCE) == ("needs_context", ["applicability_unclear"])


@pytest.mark.unit
def test_out_of_scope_wins():
    assert decide(_answer(ext=(), internal=(), out_of_scope=True), EVIDENCE) == (
        "out_of_scope",
        [],
    )


def _v2(facts=(), applied=(), ext=(), internal=(), **kw) -> ModelAnswer:
    return ModelAnswer(
        answer="черновик",
        external_basis=(
            [Basis(statement="закон", evidence_ids=list(ext))] if ext else []
        ),
        internal_basis=(
            [Basis(statement="ЛНА", evidence_ids=list(internal))] if internal else []
        ),
        object_facts=[
            ObjectFact(statement=s, evidence_ids=list(ids)) for s, ids in facts
        ],
        applied_conclusions=[
            AppliedConclusion(statement=s, evidence_ids=list(ids)) for s, ids in applied
        ],
        **kw,
    )


FACT = ("На объекте есть АПС", ("obj_s3",))
APPLIED = ("АПС объекта обслуживается по ППР и ЛНА", ("obj_s3", "ext_001", "int_001"))


@pytest.mark.unit
def test_v1_schema_has_no_object_fields():
    assert "object_facts" not in ModelAnswerV1.model_fields
    assert {"object_facts", "applied_conclusions"} <= set(ModelAnswer.model_fields)


@pytest.mark.unit
@pytest.mark.parametrize(
    "statement, hit",
    [
        ("План эвакуации для офиса не требуется", True),
        ("Огнетушители должны быть на каждом этаже", True),
        ("Количество огнетушителей не снижается", True),
        ("Хранение допускается", True),
        ("Не менее двух выходов", True),
        ("Должна проводиться проверка", True),
        ("Ответственный — офис-менеджер, должность в штате", False),
        ("В зоне офиса 2 огнетушителя ОП-4", False),
        ("Сведений о численности нет", False),
    ],
)
def test_normative_markers(statement, hit):
    assert bool(NORMATIVE_MARKERS.search(statement)) is hit


# --- citation group: any row -> failed / ["citation_invalid"] -----------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer",
    [
        _v2(facts=[("факт", ())]),
        _v2(applied=[("вывод", ())]),
        _v2(facts=[("факт", ("obj_s9",))]),
        _v2(facts=[("факт", ("obj_s2",))]),
        _v2(facts=[("факт", ("ext_001",))]),
        _v2(ext=("obj_s3",)),
        _v2(internal=("obj_s3",)),
        _v2(applied=[("вывод", ("ext_001", "int_001"))]),
    ],
    ids=[
        "fact-without-ids",
        "applied-without-ids",
        "unknown-object-id",
        "empty-section-id",
        "law-cited-as-fact",
        "object-cited-as-law",
        "object-cited-as-lna",
        "applied-without-object",
    ],
)
def test_citation_group_fails(answer):
    assert check_citations(answer, EVIDENCE)
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == (
        "failed",
        ["citation_invalid"],
    )


@pytest.mark.unit
def test_object_citation_without_profile_is_unknown_id():
    no_profile = {k: v for k, v in EVIDENCE.items() if not k.startswith("obj_")}
    assert decide(
        _v2(facts=[FACT], ext=("ext_001",), internal=("int_001",)), no_profile
    ) == (
        "failed",
        ["citation_invalid"],
    )


# --- status rows ---------------------------------------------------------------


@pytest.mark.unit
def test_clarifying_questions_block_even_with_both_levels():
    answer = _answer(clarifying_questions=["Какой объект?"])
    assert decide(answer, EVIDENCE) == ("needs_context", ["applicability_unclear"])


@pytest.mark.unit
def test_applied_with_both_levels_answered_when_dated():
    assert decide(_v2(applied=[APPLIED]), EVIDENCE, profile_as_of=DATED) == (
        "answered",
        [],
    )


# --- unknown typed field: applied conclusion -> needs_context -------------------


@pytest.mark.unit
def test_applied_on_unknown_field_needs_context():
    answer = _v2(
        applied=[
            ("План эвакуации необходим", ("obj_f_people_in_building_total", "ext_001"))
        ],
        ext=("ext_001",),
        internal=("int_001",),
    )
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == (
        "needs_context",
        ["applied_on_unknown_field"],
    )


@pytest.mark.unit
def test_applied_on_known_field_is_answered():
    answer = _v2(
        applied=[("вывод", ("obj_f_people_in_object_zone", "ext_001"))],
        ext=("ext_001",),
        internal=("int_001",),
    )
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == ("answered", [])


@pytest.mark.unit
def test_clarifying_question_wins_over_unknown_field():
    answer = _v2(
        applied=[("вывод", ("obj_f_people_in_building_total", "ext_001"))],
        ext=("ext_001",),
        internal=("int_001",),
        clarifying_questions=["Сколько людей в здании?"],
    )
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == (
        "needs_context",
        ["applicability_unclear"],
    )


@pytest.mark.unit
def test_applied_on_unknown_fields_reports_ids_once():
    answer = _v2(
        applied=[
            ("вывод", ("obj_f_people_in_building_total", "ext_001")),
            ("вывод 2", ("obj_f_people_in_building_total", "int_001")),
        ],
        ext=("ext_001",),
        internal=("int_001",),
    )
    assert applied_on_unknown_fields(answer, EVIDENCE) == [
        "obj_f_people_in_building_total"
    ]


@pytest.mark.unit
def test_applied_without_norm():
    answer = _v2(
        applied=[("вывод", ("obj_s3",))], ext=("ext_001",), internal=("int_001",)
    )
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == (
        "needs_review",
        ["applied_without_norm"],
    )


@pytest.mark.unit
def test_fact_only_answered():
    assert decide(_v2(facts=[FACT]), EVIDENCE, profile_as_of=DATED) == ("answered", [])


@pytest.mark.unit
def test_fact_only_undated_profile():
    assert decide(_v2(facts=[FACT]), EVIDENCE, profile_as_of=None) == (
        "needs_review",
        ["object_profile_undated"],
    )


@pytest.mark.unit
def test_v1_answer_without_object_citations_ignores_profile_date():
    assert decide(_answer(), EVIDENCE, profile_as_of=None) == ("answered", [])


@pytest.mark.unit
def test_one_applied_with_both_levels_closes_both_levels():
    answer = _v2(facts=[FACT], applied=[APPLIED])
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == ("answered", [])


@pytest.mark.unit
def test_cited_ids_collects_all_four_lists():
    answer = _v2(facts=[FACT], applied=[APPLIED])
    assert cited_ids(answer) == {"ext_001", "int_001", "obj_s3"}


@pytest.mark.unit
def test_cited_ids_empty_for_v1_answer_without_citations():
    assert cited_ids(ModelAnswerV1(answer="x")) == set()


# --- intersections (spec §5.1 table) -------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "answer, as_of, expected",
    [
        (
            _answer(clarifying_questions=["Какой объект?"]),
            DATED,
            ("needs_context", ["applicability_unclear"]),
        ),
        (
            _answer(ext=("ext_404",), clarifying_questions=["Какой объект?"]),
            DATED,
            ("failed", ["citation_invalid"]),
        ),
        (
            _answer(ext=("ext_404",), out_of_scope=True),
            DATED,
            ("out_of_scope", []),
        ),
        (
            _v2(applied=[("вывод", ("obj_s3",))]),
            None,
            (
                "needs_review",
                [
                    "applied_without_norm",
                    "external_evidence_missing",
                    "internal_evidence_missing",
                    "object_profile_undated",
                ],
            ),
        ),
        (
            _v2(facts=[FACT], ext=("ext_001",)),
            DATED,
            ("needs_review", ["internal_evidence_missing"]),
        ),
        (
            _v2(),
            DATED,
            (
                "needs_review",
                ["external_evidence_missing", "internal_evidence_missing"],
            ),
        ),
        (
            _v2(
                facts=[("План эвакуации для офиса не требуется", ("obj_s3",))],
                ext=("ext_001",),
                internal=("int_001",),
            ),
            DATED,
            ("needs_review", ["object_fact_normative"]),
        ),
        (
            _v2(facts=[("Дежурного персонала нет", ("obj_s2",))]),
            DATED,
            ("failed", ["citation_invalid"]),
        ),
    ],
    ids=[
        "clarify+both-norms",
        "clarify+bad-citation",
        "out_of_scope+bad-citation",
        "applied-no-norm+undated",
        "fact+ext-no-applied",
        "empty-answer",
        "normative-fact+both+dated",
        "cites-empty-section",
    ],
)
def test_intersections(answer, as_of, expected):
    assert decide(answer, EVIDENCE, profile_as_of=as_of) == expected


@pytest.mark.unit
def test_known_limit_irrelevant_but_existing_citations_pass():
    # Documents the guarantee boundary (spec §2.0 п. 1), not desired behaviour:
    # the runtime does not check that a cited fragment supports the statement.
    answer = _v2(
        applied=[
            (
                "Огнетушители офиса проверяются раз в 10 лет",
                ("obj_s3", "ext_001", "int_001"),
            )
        ]
    )
    assert decide(answer, EVIDENCE, profile_as_of=DATED) == ("answered", [])
