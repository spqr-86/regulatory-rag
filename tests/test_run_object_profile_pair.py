"""Pair run writes prompt, raw model output, evidence and response per call (spec §5.2)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.run_object_profile_pair import (
    check_paid_run,
    load_questions,
    prompt_evidence_ids,
    run_mode,
    select_questions,
)
from src.department_qa.contract import ModelAnswer


def _settings(**overrides):
    base = dict(
        SIMPLE_MODEL_NAME="gpt-4o-mini", TEMPERATURE=0.0, SIMPLE_LLM_PROVIDER="openai"
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _questions(n=9):
    return [{"n": i, "unit_id": None, "question": "q"} for i in range(n)]


@pytest.mark.unit
def test_load_questions_keeps_order_and_units(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "questions:\n"
        '  - {n: 5, unit_id: unit_office, question: "Кто?"}\n'
        '  - {n: 4, unit_id: null, question: "Нужен ли?"}\n',
        encoding="utf-8",
    )
    assert [(q["n"], q["unit_id"]) for q in load_questions(path)] == [
        (5, "unit_office"),
        (4, None),
    ]


@pytest.mark.unit
def test_run_mode_writes_one_record_per_question(tmp_path):
    raw_log = []

    def model_fn(prompt):
        raw_log.append(
            {"attempt": 1, "raw": '{"answer": "вне темы"}', "parsing_error": None}
        )
        return ModelAnswer(answer="вне темы", out_of_scope=True)

    stack = SimpleNamespace(
        config=SimpleNamespace(mode="v2", prompt_version="v2", profiles={}),
        manifest=SimpleNamespace(snapshot_id="snap"),
        search_fn=lambda q, filters=None, top_k=8: [],
        model_fn=model_fn,
    )
    questions = [{"n": 6, "unit_id": None, "question": "Как оформить отпуск?"}]

    records = run_mode(stack, questions, tmp_path, raw_log)

    saved = json.loads((tmp_path / "q6.json").read_text(encoding="utf-8"))
    assert saved == records[0]
    assert saved["mode"] == "v2"
    assert "Как оформить отпуск?" in saved["prompt"]
    assert saved["raw_model_output"] == [
        {"attempt": 1, "raw": '{"answer": "вне темы"}', "parsing_error": None}
    ]
    assert saved["response"]["status"] == "out_of_scope"
    # answer_question searches external+internal before it learns the answer is
    # out_of_scope from the model, so search_calls still records both empty calls
    # (real behaviour of src/department_qa/service.py, not asserted empty by the brief).
    assert [p["hits"] for p in saved["search_calls"]] == [[], []]
    assert saved["evidence_ids"] == []


def _capture_verifier_fn(monkeypatch):
    """Patch answer_question to record the verifier_fn it is called with."""
    from src.department_qa.service import DepartmentResponse

    captured: dict = {}

    def fake_answer_question(*args, verifier_fn=None, **kwargs):
        captured["verifier_fn"] = verifier_fn
        return DepartmentResponse(status="out_of_scope", trace_id="t")

    monkeypatch.setattr(
        "src.department_qa.service.answer_question", fake_answer_question
    )
    return captured


def _verifier_stack():
    def verifier_fn(prompt):
        raise AssertionError("verifier must not be called")

    return SimpleNamespace(
        config=SimpleNamespace(mode="v2", prompt_version="v2", profiles={}),
        manifest=SimpleNamespace(snapshot_id="snap"),
        search_fn=lambda q, filters=None, top_k=8: [],
        model_fn=lambda prompt: ModelAnswer(answer="", out_of_scope=True),
        verifier_fn=verifier_fn,
    )


@pytest.mark.unit
def test_run_mode_passes_verifier_by_default(tmp_path, monkeypatch):
    captured = _capture_verifier_fn(monkeypatch)
    questions = [{"n": 1, "unit_id": None, "question": "q"}]

    run_mode(_verifier_stack(), questions, tmp_path, [])

    assert captured["verifier_fn"] is not None
    assert (tmp_path / "q1.json").exists()


@pytest.mark.unit
def test_run_mode_can_disable_verifier(tmp_path, monkeypatch):
    captured = _capture_verifier_fn(monkeypatch)
    questions = [{"n": 1, "unit_id": None, "question": "q"}]

    run_mode(_verifier_stack(), questions, tmp_path, [], enable_verifier=False)

    assert captured["verifier_fn"] is None
    saved = json.loads((tmp_path / "q1.json").read_text(encoding="utf-8"))
    assert saved["verify_prompt"] == ""
    assert saved["raw_verifier_output"] == []


@pytest.mark.unit
def test_check_paid_run_passes_for_valid_config(tmp_path):
    assert check_paid_run(_settings(), _questions(), tmp_path / "v1") is None


@pytest.mark.unit
def test_check_paid_run_rejects_wrong_model(tmp_path):
    with pytest.raises(ValueError, match="gpt-4o-mini"):
        check_paid_run(
            _settings(SIMPLE_MODEL_NAME="gpt-4o"), _questions(), tmp_path / "v1"
        )


@pytest.mark.unit
def test_check_paid_run_rejects_nonzero_temperature(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        check_paid_run(_settings(TEMPERATURE=0.7), _questions(), tmp_path / "v1")


@pytest.mark.unit
def test_check_paid_run_rejects_non_openai_provider(tmp_path):
    with pytest.raises(ValueError, match="openai"):
        check_paid_run(
            _settings(SIMPLE_LLM_PROVIDER="anthropic"), _questions(), tmp_path / "v1"
        )


@pytest.mark.unit
def test_check_paid_run_accepts_openrouter_model_id(tmp_path):
    settings = _settings(
        SIMPLE_LLM_PROVIDER="openrouter",
        SIMPLE_MODEL_NAME="openai/gpt-4o-mini",
    )
    assert check_paid_run(settings, _questions(), tmp_path / "v2") is None


@pytest.mark.unit
def test_check_paid_run_rejects_wrong_question_count(tmp_path):
    with pytest.raises(ValueError, match="9"):
        check_paid_run(_settings(), _questions(8), tmp_path / "v1")


@pytest.mark.unit
def test_check_paid_run_rejects_existing_results(tmp_path):
    mode_dir = tmp_path / "v1"
    mode_dir.mkdir()
    (mode_dir / "q1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="q1.json"):
        check_paid_run(_settings(), _questions(), mode_dir)


@pytest.mark.unit
def test_check_paid_run_ignores_dry_run_config_only_dir(tmp_path):
    mode_dir = tmp_path / "v1"
    mode_dir.mkdir()
    (mode_dir / "config.json").write_text("{}", encoding="utf-8")
    assert check_paid_run(_settings(), _questions(), mode_dir) is None


@pytest.mark.unit
def test_prompt_evidence_ids_keeps_prompt_order():
    prompt = (
        "- ВНЕШНИЕ (id начинается с ext_) — законодательство\n"
        "[obj_s4] 4 Первичные средства\nтекст\n\n"
        "[ext_001] ПП РФ № 1479, XIX\nтекст [ext_009] внутри строки\n\n"
        "[int_001] Инструкция, 5\nтекст\n"
    )
    assert prompt_evidence_ids(prompt) == ["obj_s4", "ext_001", "int_001"]


@pytest.mark.unit
def test_check_paid_run_accepts_explicitly_allowed_model_and_subset(tmp_path):
    settings = _settings(
        SIMPLE_LLM_PROVIDER="openrouter", SIMPLE_MODEL_NAME="anthropic/claude-sonnet-5"
    )
    assert (
        check_paid_run(
            settings,
            _questions(2),
            tmp_path / "v2",
            allowed_models={"anthropic/claude-sonnet-5"},
            expected_count=2,
        )
        is None
    )


@pytest.mark.unit
def test_check_paid_run_subset_still_checks_count(tmp_path):
    with pytest.raises(ValueError, match="exactly 2"):
        check_paid_run(_settings(), _questions(3), tmp_path / "v1", expected_count=2)


@pytest.mark.unit
def test_select_questions_keeps_only_requested_numbers():
    qs = [{"n": i, "unit_id": None, "question": "q"} for i in range(1, 10)]
    assert [q["n"] for q in select_questions(qs, "2,1")] == [1, 2]


@pytest.mark.unit
def test_select_questions_rejects_unknown_number():
    qs = [{"n": 1, "unit_id": None, "question": "q"}]
    with pytest.raises(ValueError, match="7"):
        select_questions(qs, "1,7")
