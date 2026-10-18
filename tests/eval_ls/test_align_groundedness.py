"""Unit tests for eval/ls/align_groundedness.py."""

from __future__ import annotations

import hashlib
import json

import pytest

from eval.ls.align_groundedness import judge_rows, load_rows

pytestmark = pytest.mark.unit


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_batch(path, rows, labels):
    path.mkdir()
    (path / "batch.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (path / "human_labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels),
        encoding="utf-8",
    )


def _row(review_id, case_type="regular", answer="ответ", context="контекст"):
    return {
        "review_id": review_id,
        "case_id": "G" + review_id[-3:],
        "case_type": case_type,
        "inputs": {"question": "вопрос"},
        "outputs": {"answer": answer, "context": context},
        "answer_sha256": _sha(answer),
        "context_sha256": _sha(context),
    }


def _label(review_id, score):
    return {"review_id": review_id, "human_groundedness": score}


def test_load_rows_joins_labels_and_maps_case_type(tmp_path):
    _write_batch(
        tmp_path / "b",
        [_row("GR001", "false_premise"), _row("GB002")],
        [_label("GB002", 0), _label("GR001", 1)],
    )
    rows = load_rows([tmp_path / "b"])
    assert [(r["review_id"], r["human"]) for r in rows] == [("GR001", 1), ("GB002", 0)]
    assert rows[0]["reference"] == {"oos_type": "false_premise"}
    assert rows[1]["reference"] == {"oos_type": ""}


def test_load_rows_rejects_edited_batch(tmp_path):
    row = _row("GB001")
    row["outputs"]["context"] = "подменённый контекст"
    _write_batch(tmp_path / "b", [row], [_label("GB001", 0)])
    with pytest.raises(ValueError, match="GB001"):
        load_rows([tmp_path / "b"])


def test_load_rows_rejects_missing_label(tmp_path):
    _write_batch(tmp_path / "b", [_row("GB001"), _row("GB002")], [_label("GB001", 0)])
    with pytest.raises(ValueError, match="GB002"):
        load_rows([tmp_path / "b"])


def test_judge_rows_passes_exact_inputs_and_keeps_comment(tmp_path):
    _write_batch(tmp_path / "b", [_row("GB001")], [_label("GB001", 0)])
    seen = []

    def evaluator(inputs, outputs, reference_outputs):
        seen.append((inputs, outputs, reference_outputs))
        return {"key": "groundedness", "score": 1, "comment": "supported"}

    results = judge_rows(load_rows([tmp_path / "b"]), evaluator)
    assert seen == [
        (
            {"question": "вопрос"},
            {"answer": "ответ", "context": "контекст"},
            {"oos_type": ""},
        )
    ]
    assert results == [
        {
            "review_id": "GB001",
            "case_id": "G001",
            "human": 0,
            "judge": 1,
            "comment": "supported",
        }
    ]


def test_judge_rows_keeps_raw_judge_reply(tmp_path):
    _write_batch(tmp_path / "b", [_row("GB001")], [_label("GB001", 0)])

    def evaluator(inputs, outputs, reference_outputs):
        return {"key": "groundedness", "score": None, "comment": "x", "raw": "{}"}

    [result] = judge_rows(load_rows([tmp_path / "b"]), evaluator)
    assert result["raw"] == "{}"


def test_groundedness_evaluator_returns_raw_reply_on_format_error():
    from langchain_core.messages import AIMessage

    from eval.ls.groundedness import make_groundedness_evaluator

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(content='{"checks": "broken"}')

    evaluator = make_groundedness_evaluator(FakeLLM(), keep_raw=True)
    result = evaluator(
        {"question": "вопрос"},
        {"answer": "ответ", "context": "контекст"},
        {"oos_type": ""},
    )
    assert result["score"] is None
    assert result["comment"].startswith("judge_format_error")
    assert result["raw"] == '{"checks": "broken"}'
    plain = make_groundedness_evaluator(FakeLLM())(
        {"question": "вопрос"},
        {"answer": "ответ", "context": "контекст"},
        {"oos_type": ""},
    )
    assert "raw" not in plain


def test_select_rows_keeps_requested_ids_and_rejects_unknown(tmp_path):
    from eval.ls.align_groundedness import select_rows

    _write_batch(
        tmp_path / "b",
        [_row("GB001"), _row("GB002")],
        [_label("GB001", 0), _label("GB002", 1)],
    )
    rows = load_rows([tmp_path / "b"])
    assert [r["review_id"] for r in select_rows(rows, ["GB002"])] == ["GB002"]
    assert select_rows(rows, None) == rows
    with pytest.raises(ValueError, match="GB404"):
        select_rows(rows, ["GB404"])


def test_truncated_judge_reply_is_unscored_and_keeps_raw():
    from langchain_core.messages import AIMessage

    from eval.ls.groundedness import make_groundedness_evaluator

    class FakeLLM:
        def invoke(self, messages):
            return AIMessage(
                content='{"checks": [', response_metadata={"finish_reason": "length"}
            )

    result = make_groundedness_evaluator(FakeLLM(), keep_raw=True)(
        {"question": "вопрос"},
        {"answer": "ответ", "context": "контекст"},
        {"oos_type": ""},
    )
    assert result["score"] is None
    assert "truncated" in result["comment"]
    assert result["raw"] == '{"checks": ['


def test_judge_rows_saves_each_verdict_before_a_failure(tmp_path):
    _write_batch(
        tmp_path / "b",
        [_row("GB001"), _row("GB002")],
        [_label("GB001", 0), _label("GB002", 1)],
    )
    calls = []

    def evaluator(inputs, outputs, reference_outputs):
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError("judge timeout")
        return {"key": "groundedness", "score": 0, "comment": "x"}

    saved = []
    with pytest.raises(TimeoutError):
        judge_rows(load_rows([tmp_path / "b"]), evaluator, on_result=saved.append)
    assert [r["review_id"] for r in saved] == ["GB001"]


def test_acceptance_requires_every_row_scored():
    from eval.ls.align_groundedness import acceptance

    scored = [{"review_id": f"R{i}", "human": 1, "judge": 1} for i in range(11)]
    unscored = [{"review_id": f"U{i}", "human": 1, "judge": None} for i in range(8)]
    assert acceptance(scored)["accepted"] is True
    result = acceptance(scored + unscored)
    assert result["accepted"] is False
    assert result["skipped"] == 8
