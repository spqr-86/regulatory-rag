"""Unit tests for eval/score_object_profile_run.py — deterministic run scorer."""

from __future__ import annotations

import json

import pytest

from eval.score_object_profile_run import (
    DEFAULT_EXPECTATIONS,
    FieldRef,
    applied_on_unknown,
    cited_field_ids,
    field_states,
    forbidden_conclusion_hits,
    load_expectations,
    load_run,
    render_markdown,
    score_question,
    score_run,
    summarize,
)


def _expectation(n=1, typed=None, forbidden=None, expected=None):
    return {
        "n": n,
        "unit_id": "unit_office",
        "object_facts": {
            "typed_fields": (
                typed
                if typed is not None
                else {"people_in_object_zone": 8, "people_in_building_total": "unknown"}
            )
        },
        "forbidden_conclusions": forbidden or [],
        "expected": expected or {"v2": {"status": "answered", "reasons": []}},
    }


def _record(n=1, response=None, mode="v2"):
    return {
        "n": n,
        "mode": mode,
        "unit_id": "unit_office",
        "response": response if response is not None else {"status": "answered"},
    }


# ── load / parse ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_expectations_reads_typed_fields(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "questions:\n"
        "  - n: 1\n"
        "    object_facts:\n"
        "      typed_fields: {people_in_object_zone: 8, people_in_building_total: unknown}\n",
        encoding="utf-8",
    )
    loaded = load_expectations(path)
    assert loaded[1]["object_facts"]["typed_fields"]["people_in_object_zone"] == 8


@pytest.mark.unit
def test_load_expectations_allows_missing_typed_fields(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("questions:\n  - {n: 1, question: q}\n", encoding="utf-8")
    loaded = load_expectations(path)
    assert field_states(loaded[1]) == {}


@pytest.mark.unit
def test_load_expectations_rejects_empty_file(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("questions: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no questions"):
        load_expectations(path)


@pytest.mark.unit
def test_real_expectations_load_and_cover_nine_questions():
    loaded = load_expectations(DEFAULT_EXPECTATIONS)
    assert sorted(loaded) == list(range(1, 10))
    assert loaded[1]["object_facts"]["typed_fields"]["people_in_object_zone"] == 8


@pytest.mark.unit
def test_load_run_keys_by_question_number(tmp_path):
    (tmp_path / "q2.json").write_text(json.dumps(_record(2)), encoding="utf-8")
    (tmp_path / "q1.json").write_text(json.dumps(_record(1)), encoding="utf-8")
    assert sorted(load_run(tmp_path)) == [1, 2]


@pytest.mark.unit
def test_load_run_rejects_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="no q"):
        load_run(tmp_path)


# ── cited fields ───────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_cited_field_ids_keeps_only_obj_f_and_dedups():
    response = {
        "object_facts": [{"evidence_ids": ["obj_f_a", "obj_s4"]}],
        "applied_conclusions": [
            {"evidence_ids": ["obj_f_a", "ext_004"]},
            {"evidence_ids": ["obj_f_b"]},
        ],
    }
    assert cited_field_ids(response) == ["obj_f_a", "obj_f_b"]


@pytest.mark.unit
def test_field_states_classifies_known_unknown_and_not_applicable():
    states = field_states(
        _expectation(
            typed={
                "people_in_object_zone": 8,
                "people_in_building_total": "unknown",
                "outside_ladder_last_test_date": "not_applicable",
            }
        )
    )
    assert states == {
        "people_in_object_zone": "known",
        "people_in_building_total": "unknown",
        "outside_ladder_last_test_date": "not_applicable",
    }


# ── violations ─────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_applied_on_unknown_flags_conclusion_citing_unknown_field():
    response = {
        "applied_conclusions": [
            {
                "statement": "План эвакуации необходим.",
                "evidence_ids": ["obj_f_people_in_building_total", "ext_002"],
            }
        ]
    }
    violations = applied_on_unknown(response, {"people_in_building_total": "unknown"})
    assert [v.code for v in violations] == ["applied_on_unknown"]
    assert "obj_f_people_in_building_total" in violations[0].detail


@pytest.mark.unit
def test_applied_on_unknown_ignores_conditional_answer_asking_for_context():
    response = {
        "clarifying_questions": ["Сколько людей может находиться в здании?"],
        "applied_conclusions": [
            {
                "statement": "План нужен, если в здании 50 и более человек.",
                "evidence_ids": ["obj_f_people_in_building_total", "ext_002"],
            }
        ],
    }
    assert applied_on_unknown(response, {"people_in_building_total": "unknown"}) == []


@pytest.mark.unit
def test_applied_on_unknown_ignores_known_and_unmapped_fields():
    response = {
        "applied_conclusions": [
            {"statement": "s", "evidence_ids": ["obj_f_people_in_object_zone"]},
            {"statement": "s", "evidence_ids": ["obj_f_new_field"]},
        ]
    }
    assert applied_on_unknown(response, {"people_in_object_zone": "known"}) == []


@pytest.mark.unit
def test_forbidden_conclusion_hits_normalizes_whitespace_and_case():
    response = {
        "answer": "Вывод:  План   эвакуации офису НЕ НУЖЕН.",
        "object_facts": [],
        "applied_conclusions": [],
    }
    hits = forbidden_conclusion_hits(response, ["план эвакуации офису не нужен"])
    assert [v.code for v in hits] == ["forbidden_conclusion"]


@pytest.mark.unit
def test_forbidden_conclusion_hits_returns_empty_on_paraphrase():
    response = {"answer": "План эвакуации не требуется.", "object_facts": []}
    assert forbidden_conclusion_hits(response, ["план эвакуации офису не нужен"]) == []


# ── scoring ────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_score_question_marks_contract_primary_match():
    report = score_question(
        _expectation(expected={"v2": {"status": "answered", "reasons": []}}),
        _record(response={"status": "answered", "reason_codes": []}),
    )
    assert report.contract_ok is True


@pytest.mark.unit
def test_score_question_accepts_listed_alternative():
    expectation = _expectation(
        expected={
            "v2": {
                "status": "answered",
                "reasons": [],
                "alternatives": [
                    {"status": "needs_context", "reasons": ["applicability_unclear"]}
                ],
            }
        }
    )
    report = score_question(
        expectation,
        _record(
            response={
                "status": "needs_context",
                "reason_codes": ["applicability_unclear"],
            }
        ),
    )
    assert report.contract_ok is True


@pytest.mark.unit
def test_score_question_rejects_other_reasons():
    report = score_question(
        _expectation(expected={"v2": {"status": "answered", "reasons": []}}),
        _record(
            response={
                "status": "answered",
                "reason_codes": ["external_evidence_missing"],
            }
        ),
    )
    assert report.contract_ok is False


@pytest.mark.unit
def test_score_question_contract_none_for_unknown_mode():
    report = score_question(_expectation(), _record(mode="v9"))
    assert report.contract_ok is None


@pytest.mark.unit
def test_score_question_collects_fields_and_violations():
    expectation = _expectation(forbidden=["план эвакуации необходим"])
    record = _record(
        response={
            "status": "answered",
            "reason_codes": [],
            "answer": "План эвакуации необходим.",
            "applied_conclusions": [
                {
                    "statement": "План эвакуации необходим.",
                    "evidence_ids": ["obj_f_people_in_building_total", "ext_002"],
                }
            ],
        }
    )
    report = score_question(expectation, record)
    assert [f.state for f in report.cited_fields] == ["unknown"]
    assert report.cited_fields[0].label == "Людей в здании всего"
    assert {v.code for v in report.violations} == {
        "applied_on_unknown",
        "forbidden_conclusion",
    }


@pytest.mark.unit
def test_score_question_marks_unmapped_field():
    record = _record(
        response={
            "status": "answered",
            "object_facts": [{"statement": "s", "evidence_ids": ["obj_f_unknown_key"]}],
        }
    )
    report = score_question(_expectation(typed={}), record)
    assert report.cited_fields[0].state == "unmapped"


@pytest.mark.unit
def test_score_run_raises_on_missing_question(tmp_path):
    expectations = {1: _expectation(1), 2: _expectation(2)}
    (tmp_path / "q1.json").write_text(json.dumps(_record(1)), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[2\]"):
        score_run(expectations, tmp_path)


@pytest.mark.unit
def test_score_run_scores_all_in_expectation_order(tmp_path):
    expectations = {1: _expectation(1), 2: _expectation(2)}
    (tmp_path / "q2.json").write_text(json.dumps(_record(2)), encoding="utf-8")
    (tmp_path / "q1.json").write_text(json.dumps(_record(1)), encoding="utf-8")
    reports = score_run(expectations, tmp_path)
    assert [r.n for r in reports] == [1, 2]


@pytest.mark.unit
def test_score_run_only_scores_selected_questions(tmp_path):
    expectations = {1: _expectation(1), 2: _expectation(2)}
    (tmp_path / "q2.json").write_text(json.dumps(_record(2)), encoding="utf-8")
    reports = score_run(expectations, tmp_path, only={2})
    assert [r.n for r in reports] == [2]


# ── summary / render ───────────────────────────────────────────────────────────


@pytest.mark.unit
def test_summarize_counts_reports():
    reports = [
        score_question(
            _expectation(1), _record(1, {"status": "answered", "reason_codes": []})
        ),
        score_question(
            _expectation(2),
            _record(2, {"status": "needs_review", "reason_codes": ["x"]}),
        ),
    ]
    summary = summarize(reports)
    assert summary["questions"] == 2
    assert summary["contract_ok"] == 1


@pytest.mark.unit
def test_render_markdown_lists_violations():
    report = score_question(
        _expectation(1),
        _record(
            1,
            {
                "status": "answered",
                "reason_codes": [],
                "applied_conclusions": [
                    {
                        "statement": "s",
                        "evidence_ids": ["obj_f_people_in_building_total"],
                    }
                ],
            },
        ),
    )
    text = render_markdown([report], summarize([report]))
    assert "applied_on_unknown" in text
    assert "q1" in text


@pytest.mark.unit
def test_field_ref_defaults_label_to_key_for_unknown_field():
    ref = FieldRef(id="obj_f_custom", key="custom", label="custom", state="known")
    assert ref.label == "custom"
