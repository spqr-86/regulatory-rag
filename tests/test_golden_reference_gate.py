"""A disputed legal reference cannot contribute to the headline score."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from eval.run_v7_eval import (
    DATASET_PATH,
    CorrectnessPromptVariables,
    correctness_prompt_text,
    evaluate_correctness,
    load_dataset,
    scorable_in_scope,
)
from scripts.rejudge_golden import selected_cases


def test_load_dataset_requires_review_status(tmp_path: Path) -> None:
    path = tmp_path / "golden.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "question",
                "ground_truth",
                "reference_status",
                "legal_as_of",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "case_id": "G001",
                "question": "Вопрос?",
                "ground_truth": "Ответ",
                "reference_status": "incorrect",
                "legal_as_of": "2026-09-30",
            }
        )
    assert load_dataset(path)[0]["reference_status"] == "incorrect"


def test_load_dataset_rejects_missing_status(tmp_path: Path) -> None:
    path = tmp_path / "golden.csv"
    path.write_text("question,ground_truth\nВопрос?,Ответ\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reference_status"):
        load_dataset(path)


def test_only_verified_in_scope_answers_are_scorable() -> None:
    results = [
        {
            "reference_status": "verified",
            "oos_type": "",
            "answer": "A",
            "correctness_score": 9,
        },
        {
            "reference_status": "incorrect",
            "oos_type": "",
            "answer": "B",
            "correctness_score": 10,
        },
        {"reference_status": "needs_clarification", "oos_type": "", "answer": "C"},
        {"reference_status": "verified", "oos_type": "out_of_scope", "answer": "D"},
    ]
    assert [r["answer"] for r in scorable_in_scope(results)] == ["A"]


def test_known_false_legal_claims_are_not_in_the_revised_key() -> None:
    rows = {row["case_id"]: row for row in load_dataset(DATASET_PATH)}
    assert len(rows) == 56
    assert "не реже одного раза в год" not in rows["G011"]["ground_truth"]
    assert "256 часов минимум" not in rows["G025"]["ground_truth"]
    assert "более 5%" not in rows["G029"]["ground_truth"]
    assert all(
        rows[case]["reference_status"] != "verified"
        for case in ("G011", "G025", "G029")
    )


def test_judge_records_asserted_claims() -> None:
    judge = RunnableLambda(
        lambda _: (
            '{"score": 3, "reasoning": "Неверный срок", '
            '"forbidden_claims_asserted": ["Ежегодно достаточно"]}'
        )
    )
    result = evaluate_correctness(
        "Как часто?", "Раз в полгода", "Раз в год", judge, "Ежегодно достаточно"
    )
    assert result["correctness_score"] == 3
    assert result["forbidden_claims_asserted"] == ["Ежегодно достаточно"]


def test_malformed_judge_response_does_not_become_zero_score() -> None:
    judge = RunnableLambda(lambda _: "не JSON")
    with pytest.raises(ValueError):
        evaluate_correctness("Вопрос?", "Ответ", "Текст", judge)


def test_json_fence_is_accepted_without_extracting_arbitrary_text() -> None:
    judge = RunnableLambda(
        lambda _: (
            '```json\n{"score": 9, "reasoning": "ok", "forbidden_claims_asserted": []}\n```'
        )
    )
    assert evaluate_correctness("Q", "GT", "A", judge)["correctness_score"] == 9


def test_correctness_prompt_placeholders_match_typed_contract() -> None:
    placeholders = set(
        ChatPromptTemplate.from_template(correctness_prompt_text()).input_variables
    )
    assert placeholders == set(CorrectnessPromptVariables.model_fields)


def test_saved_answers_align_only_with_eligible_current_references() -> None:
    key = [
        {
            "case_id": "G001",
            "question": "Вопрос 1",
            "reference_status": "verified",
            "oos_type": "",
        },
        {
            "case_id": "G002",
            "question": "Вопрос 2",
            "reference_status": "needs_clarification",
            "oos_type": "",
        },
        {
            "case_id": "G003",
            "question": "Вопрос 3",
            "reference_status": "verified",
            "oos_type": "out_of_scope",
        },
    ]
    saved = [
        {"question": "Вопрос 1", "answer": "Ответ 1"},
        {"question": "Вопрос 2", "answer": "Ответ 2"},
        {"question": "Вопрос 3", "answer": "Ответ 3"},
    ]
    assert [case["case_id"] for case, _ in selected_cases(saved, key)] == ["G001"]
    saved[0]["question"] = "Другой вопрос"
    with pytest.raises(ValueError, match="G001"):
        selected_cases(saved, key)
    saved[0]["question"] = "Вопрос 1"
    saved[0]["case_id"] = "G999"
    with pytest.raises(ValueError, match="case_id"):
        selected_cases(saved, key)
