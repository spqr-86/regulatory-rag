"""Pair run writes prompt, raw model output, evidence and response per call (spec §5.2)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval.run_object_profile_pair import load_questions, run_mode
from src.department_qa.contract import ModelAnswer


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
    # out_of_scope from the model, so evidence_passed still records both empty calls
    # (real behaviour of src/department_qa/service.py, not asserted empty by the brief).
    assert [p["hits"] for p in saved["evidence_passed"]] == [[], []]
