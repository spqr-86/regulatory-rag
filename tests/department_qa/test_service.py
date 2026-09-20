"""Tests for the department Q&A flow: two scoped searches → prompt → checked answer."""

from __future__ import annotations

from datetime import date
import threading
import time

import pytest

from src.department_qa.contract import (
    AppliedConclusion,
    Basis,
    ModelAnswer,
    ObjectFact,
    ObjectSection,
    RequestContext,
    VerificationResult,
)
from src.department_qa.object_profile import ObjectProfile, TypedObjectFields
from src.department_qa.service import (
    ServiceLimits,
    answer_question,
    answer_scoped_question,
)
from src.v7.retrieval import ScopedRetrievalResult
from src.v7.scope_filter import build_scope_filters


def _passage(text, source, **meta):
    return {"text": text, "score": 0.7, "metadata": {"source": source, **meta}}


EXT_HITS = [_passage("Огнетушители осматривают по паспорту.", "ppr.pdf", title="ППР")]
INT_HITS = [
    _passage("Осмотр раз в квартал.", "pril3.md", title="Инструкция", chunk_id=4)
]


class FakeSearch:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, query, filters=None, top_k=8):
        self.calls.append(filters)
        if self.fail:
            raise RuntimeError("chroma down")
        return EXT_HITS if filters["source_type"] == "external" else INT_HITS


def _model(answer: ModelAnswer):
    prompts = []

    def call(prompt: str) -> ModelAnswer:
        prompts.append(prompt)
        return answer

    call.prompts = prompts
    return call


GOOD = ModelAnswer(
    answer="Осмотр ежеквартально.",
    external_basis=[Basis(statement="по паспорту", evidence_ids=["ext_001"])],
    internal_basis=[Basis(statement="раз в квартал", evidence_ids=["int_001"])],
)


@pytest.mark.unit
def test_two_scoped_searches_and_answered_response():
    search, model = FakeSearch(), _model(GOOD)
    result = answer_question("Как часто?", "unit_1", search, model, snapshot_id="s1")

    assert search.calls == list(build_scope_filters("unit_1"))
    assert "[ext_001]" in model.prompts[0] and "[int_001]" in model.prompts[0]
    assert result.status == "answered"
    assert result.snapshot_id == "s1"
    assert [e.id for e in result.evidence] == ["ext_001", "int_001"]
    assert result.evidence[1].text == "Осмотр раз в квартал."
    assert result.evidence[1].chunk_id == 4


@pytest.mark.unit
def test_evidence_keeps_retrieval_score():
    result = answer_question("Как часто?", "unit_1", FakeSearch(), _model(GOOD))
    assert result.evidence[0].retrieval_score == 0.7
    assert result.evidence[1].retrieval_score == 0.7


@pytest.mark.unit
def test_progress_reports_real_stages_in_order():
    stages: list[str] = []
    answer_question(
        "Как часто?", "unit_1", FakeSearch(), _model(GOOD), progress_fn=stages.append
    )
    assert stages == [
        "retrieval_started",
        "retrieval_completed",
        "generation_started",
        "generation_completed",
        "verification_started",
        "verification_completed",
    ]


@pytest.mark.unit
def test_progress_stops_at_retrieval_failure():
    stages: list[str] = []
    result = answer_question(
        "q", "unit_1", FakeSearch(fail=True), _model(GOOD), progress_fn=stages.append
    )
    assert result.status == "failed"
    assert stages == ["retrieval_started"]


@pytest.mark.unit
def test_progress_callback_error_does_not_break_answer():
    def boom(stage: str) -> None:
        raise RuntimeError("ui gone")

    result = answer_question(
        "Как часто?", "unit_1", FakeSearch(), _model(GOOD), progress_fn=boom
    )
    assert result.status == "answered"


@pytest.mark.unit
def test_retrieval_error_is_failed_not_empty():
    model = _model(GOOD)
    result = answer_question("q", "unit_1", FakeSearch(fail=True), model)
    assert (result.status, result.reason_codes) == ("failed", ["retrieval_failed"])
    assert model.prompts == []


@pytest.mark.unit
def test_generation_error_is_failed_without_passages_as_answer():
    def broken(prompt):
        raise ValueError("schema validation failed")

    result = answer_question("q", "unit_1", FakeSearch(), broken)
    assert (result.status, result.reason_codes) == ("failed", ["generation_failed"])
    assert result.answer == ""


@pytest.mark.unit
def test_citation_invalid_hides_basis():
    bad = GOOD.model_copy(
        update={"internal_basis": [Basis(statement="x", evidence_ids=["int_999"])]}
    )
    result = answer_question("q", "unit_1", FakeSearch(), _model(bad))
    assert (result.status, result.reason_codes) == ("failed", ["citation_invalid"])
    assert result.external_basis == [] and result.internal_basis == []


def _profile(unit_id="unit_1", as_of=date(2026, 9, 1)) -> ObjectProfile:
    def section(n, title, text="", presence="present"):
        return ObjectSection(
            id=f"obj_s{n}", number=n, title=title, text=text, presence=presence
        )

    sections = {
        "obj_s3": section(
            3,
            "Системы противопожарной защиты объекта",
            "АПС есть, сигнал принимает пост охраны.",
        ),
        "obj_s4": section(4, "Первичные средства пожаротушения", presence="empty"),
        "obj_s5": section(5, "Дежурный персонал", presence="missing"),
        "obj_s6": section(
            6, "Контактные телефоны объекта", "маркер-телефонов-раздела-6"
        ),
    }
    return ObjectProfile(
        unit_id=unit_id,
        document_id="int_unit_1_list",
        source="unit_1_list.md",
        content_sha256="sha-abc",
        title="Лист особенностей объекта защиты: тест",
        as_of_date=as_of,
        sections=sections,
        typed_fields=TypedObjectFields(
            people_in_object_zone=8,
            people_on_floor_total="unknown",
            people_in_building_total="unknown",
            permanent_workplaces_on_floor="unknown",
            evacuation_plan_present=False,
            room_categories="unknown",
            aupt_present=False,
            extinguishers_total=2,
            outside_ladder_last_test_date="not_applicable",
        ),
    )


V2_GOOD = ModelAnswer(
    answer="Сигнал принимает пост охраны.",
    object_facts=[
        ObjectFact(
            statement="Сигнал АПС принимает пост охраны", evidence_ids=["obj_s3"]
        )
    ],
)


@pytest.mark.unit
def test_profile_sections_become_object_evidence_and_prompt_block():
    search, model = FakeSearch(), _model(V2_GOOD)
    result = answer_question(
        "Кто принимает сигнал?",
        "unit_1",
        search,
        model,
        profile=_profile(),
        prompt_version="v2",
    )

    prompt = model.prompts[0]
    assert "[obj_s3] 3 Системы противопожарной защиты объекта" in prompt
    assert "(не заполнено) — ссылаться нельзя" in prompt
    assert "(раздела нет в листе) — ссылаться нельзя" in prompt
    assert "маркер-телефонов-раздела-6" not in prompt
    assert "Дата заполнения: 01.09.2026" in prompt

    assert (result.status, result.reason_codes) == ("answered", [])
    assert [e.id for e in result.evidence] == ["obj_s3"]
    obj = result.evidence[0]
    assert (obj.level, obj.source, obj.document_id) == (
        "object",
        "unit_1_list.md",
        "int_unit_1_list",
    )
    assert obj.locator == "3 Системы противопожарной защиты объекта"
    assert result.object_facts == V2_GOOD.object_facts
    assert result.profile_as_of_date == date(2026, 9, 1)
    assert result.profile_sha256 == "sha-abc"


@pytest.mark.unit
def test_typed_field_id_is_valid_object_evidence():
    answer = ModelAnswer(
        answer="Число людей в здании неизвестно.",
        object_facts=[
            ObjectFact(
                statement="Число людей в здании неизвестно",
                evidence_ids=["obj_f_people_in_building_total"],
            )
        ],
    )

    result = answer_question(
        "Сколько людей в здании?",
        "unit_1",
        FakeSearch(),
        _model(answer),
        profile=_profile(),
        prompt_version="v4",
    )

    assert (result.status, result.reason_codes) == ("answered", [])
    assert [e.id for e in result.evidence] == ["obj_f_people_in_building_total"]
    assert result.evidence[0].text == "Людей в здании всего: неизвестно"


@pytest.mark.unit
@pytest.mark.parametrize("section_id", ["obj_s4", "obj_s5", "obj_s6"])
def test_empty_missing_and_contacts_sections_are_not_citable(section_id):
    bad = ModelAnswer(
        answer="x",
        object_facts=[ObjectFact(statement="факт", evidence_ids=[section_id])],
    )
    result = answer_question(
        "q",
        "unit_1",
        FakeSearch(),
        _model(bad),
        profile=_profile(),
        prompt_version="v2",
    )
    assert (result.status, result.reason_codes) == ("failed", ["citation_invalid"])
    assert result.object_facts == [] and result.evidence == []


@pytest.mark.unit
@pytest.mark.parametrize("unit_id", ["unit_2", None], ids=["other-unit", "no-unit"])
def test_profile_mismatch_fails_before_search_and_model(unit_id):
    search, model = FakeSearch(), _model(V2_GOOD)
    result = answer_question(
        "q", unit_id, search, model, profile=_profile("unit_1"), prompt_version="v2"
    )
    assert (result.status, result.reason_codes) == ("failed", ["profile_mismatch"])
    assert search.calls == [] and model.prompts == []


@pytest.mark.unit
def test_undated_profile_citation_needs_review():
    result = answer_question(
        "q",
        "unit_1",
        FakeSearch(),
        _model(V2_GOOD),
        profile=_profile(as_of=None),
        prompt_version="v2",
    )
    assert (result.status, result.reason_codes) == (
        "needs_review",
        ["object_profile_undated"],
    )
    assert result.profile_as_of_date is None


@pytest.mark.unit
def test_nothing_cited_means_no_evidence():
    silent = ModelAnswer(answer="Не найдено.")
    result = answer_question("q", "unit_1", FakeSearch(), _model(silent))
    assert result.status == "needs_review"
    assert result.evidence == []


@pytest.mark.unit
def test_out_of_scope_hides_model_content():
    answer = GOOD.model_copy(
        update={"out_of_scope": True, "clarifying_questions": ["?"]}
    )
    result = answer_question(
        "Как оформить отпуск?", "unit_1", FakeSearch(), _model(answer)
    )
    assert (result.status, result.reason_codes) == ("out_of_scope", [])
    assert result.answer == ""
    assert result.external_basis == [] and result.internal_basis == []
    assert result.clarifying_questions == [] and result.evidence == []
    assert result.next_step


@pytest.mark.unit
def test_applied_conclusion_evidence_is_resolved():
    applied = ModelAnswer(
        answer="x",
        applied_conclusions=[
            AppliedConclusion(
                statement="вывод", evidence_ids=["obj_s3", "ext_001", "int_001"]
            )
        ],
    )
    result = answer_question(
        "q",
        "unit_1",
        FakeSearch(),
        _model(applied),
        profile=_profile(),
        prompt_version="v2",
    )
    assert result.status == "answered"
    assert [e.id for e in result.evidence] == ["ext_001", "int_001", "obj_s3"]


@pytest.mark.unit
def test_verifier_missing_fact_gates_answered_to_needs_context():
    applied = ModelAnswer(
        answer="На этаже нужен второй эвакуационный выход.",
        applied_conclusions=[
            AppliedConclusion(
                statement="Порог применён",
                evidence_ids=["obj_s3", "ext_001", "int_001"],
            )
        ],
    )
    verify_prompts = []

    def verify(prompt):
        verify_prompts.append(prompt)
        return VerificationResult(
            verdict="missing",
            missing_fields=["Численность людей на этаже"],
            explanation="В листе указана только другая величина.",
        )

    result = answer_question(
        "Нужен ли второй выход?",
        "unit_1",
        FakeSearch(),
        _model(applied),
        profile=_profile(),
        prompt_version="v2",
        verifier_fn=verify,
    )

    assert result.status == "needs_context"
    assert result.reason_codes == ["verification_missing_facts"]
    assert result.clarifying_questions == ["Численность людей на этаже"]
    assert result.verification.verdict == "missing"
    assert "Численность людей на этаже" not in verify_prompts[0]
    assert "Порог применён" in verify_prompts[0]


@pytest.mark.unit
def test_verifier_failure_fails_closed_without_hiding_answer():
    applied = ModelAnswer(
        answer="Вывод.",
        applied_conclusions=[
            AppliedConclusion(
                statement="вывод", evidence_ids=["obj_s3", "ext_001", "int_001"]
            )
        ],
    )

    def broken(_prompt):
        raise RuntimeError("provider unavailable")

    result = answer_question(
        "q",
        "unit_1",
        FakeSearch(),
        _model(applied),
        profile=_profile(),
        prompt_version="v2",
        verifier_fn=broken,
    )
    assert (result.status, result.reason_codes) == (
        "needs_review",
        ["verification_failed"],
    )
    assert result.answer == "Вывод."


@pytest.mark.unit
@pytest.mark.parametrize(
    "unit_id, phrase",
    [(None, "Объект не выбран"), ("unit_1", "Лист не предоставлен")],
)
def test_prompt_without_profile_says_why(unit_id, phrase):
    model = _model(GOOD)
    answer_question("q", unit_id, FakeSearch(), model, prompt_version="v2")
    assert phrase in model.prompts[0]


def _retrieval(outcome="ready", text="норма", route="simple"):
    context = (
        [_passage(text, f"{text}.md", document_id=text, chunk_id=1)]
        if outcome == "ready"
        else []
    )
    return ScopedRetrievalResult(
        final_context=context,
        outcome=outcome,
        route=route,
        reason=None,
        attempts=[{"route": route}],
        elapsed_ms=1.0,
        technical_failure=False,
    )


@pytest.mark.unit
def test_scoped_invalid_context_performs_no_io():
    calls = []

    def retrieve(*args, **kwargs):
        calls.append("retrieve")

    model = _model(GOOD)
    result = answer_scoped_question(
        "q",
        RequestContext(
            corpora=("external",), unit_id="missing", include_object_profile=False
        ),
        retrieve,
        model,
        known_units={"unit_1"},
    )
    assert (result.status, result.reason_codes) == ("failed", ["invalid_unit"])
    assert calls == [] and model.prompts == []


@pytest.mark.unit
def test_service_limits_reject_unbounded_or_invalid_values():
    with pytest.raises(ValueError, match="max_workers"):
        ServiceLimits(max_workers=0)
    with pytest.raises(ValueError, match="max_pending"):
        ServiceLimits(max_pending=-1)


@pytest.mark.unit
def test_scoped_single_corpus_does_not_start_other_branch():
    calls = []

    def retrieve(question, *, corpus, **kwargs):
        calls.append(corpus)
        return _retrieval(text=corpus)

    external_only = ModelAnswer(
        answer="закон",
        external_basis=[Basis(statement="закон", evidence_ids=["ext_001"])],
    )
    result = answer_scoped_question(
        "q",
        RequestContext(corpora=("external",), include_object_profile=False),
        retrieve,
        _model(external_only),
        known_units=set(),
    )
    assert calls == ["external"]
    assert result.status == "answered"
    assert [item.id for item in result.evidence] == ["ext_001"]
    assert result.retrieval_trace["internal"] == {"outcome": "not_requested"}
    assert result.retrieval_stats["embedding"] == {
        "computations": 0,
        "api_attempts": 0,
        "cache_hits": 0,
    }


@pytest.mark.unit
def test_scoped_dual_branches_overlap_and_keep_stable_evidence_order():
    barrier = threading.Barrier(2)
    release = threading.Event()

    def retrieve(question, *, corpus, **kwargs):
        barrier.wait(timeout=1)
        if corpus == "external":
            release.wait(timeout=1)
        else:
            release.set()
        return _retrieval(
            text=corpus, route="complex" if corpus == "internal" else "simple"
        )

    result = answer_scoped_question(
        "q",
        RequestContext(include_object_profile=False),
        retrieve,
        _model(GOOD),
        known_units=set(),
    )
    assert result.status == "answered"
    assert [item.id for item in result.evidence] == ["ext_001", "int_001"]
    assert result.retrieval_trace["external"]["route"] == "simple"
    assert result.retrieval_trace["internal"]["route"] == "complex"


@pytest.mark.unit
def test_scoped_missing_requested_branch_caps_status_at_needs_review():
    def retrieve(question, *, corpus, **kwargs):
        return _retrieval(text=corpus) if corpus == "external" else _retrieval("empty")

    answer = ModelAnswer(
        answer="ограниченный ответ",
        external_basis=[Basis(statement="закон", evidence_ids=["ext_001"])],
    )
    result = answer_scoped_question(
        "q",
        RequestContext(include_object_profile=False),
        retrieve,
        _model(answer),
        known_units=set(),
    )
    assert result.status == "needs_review"
    assert "internal_evidence_missing" in result.reason_codes


@pytest.mark.unit
def test_scoped_timeout_returns_without_generation_or_waiting_for_worker():
    release = threading.Event()

    def retrieve(*args, **kwargs):
        release.wait(timeout=1)
        return _retrieval()

    model = _model(GOOD)
    started = time.monotonic()
    result = answer_scoped_question(
        "q",
        RequestContext(corpora=("external",), include_object_profile=False),
        retrieve,
        model,
        known_units=set(),
        limits=ServiceLimits(retrieval_timeout_s=0.03),
    )
    elapsed = time.monotonic() - started
    release.set()
    assert elapsed < 0.2
    assert (result.status, result.reason_codes) == ("failed", ["retrieval_timeout"])
    assert model.prompts == []
