"""Pure contracts and metrics for issue #9 triage calibration."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from eval.triage_calibration import (
    CalibrationQuestion,
    binary_hit,
    load_calibration_gt,
    load_calibration_split,
    make_audit_record,
    score_context,
    summarize_audit,
)


def _passage(source: str, chunk_id: int) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": "нормативный текст",
        "metadata": {"source": source},
    }


def _question(**overrides) -> CalibrationQuestion:
    data = {
        "question": "Какие требования применяются?",
        "chunk_id": "doc.pdf#1",
        "relevant_chunk_ids": ["doc.pdf#1", "doc.pdf#2"],
        "source": "doc.pdf",
        "required_elements": [
            {
                "id": "rule",
                "label": "основное требование",
                "acceptable_chunk_ids": ["doc.pdf#1", "doc.pdf#2"],
                "critical": True,
            }
        ],
    }
    data.update(overrides)
    return CalibrationQuestion.model_validate(data)


@pytest.mark.unit
def test_alternative_chunks_cover_one_element():
    question = _question()

    score = score_context([_passage("doc.pdf", 2)], question.required_elements)

    assert score == {
        "covered": 1,
        "required": 1,
        "coverage": 1.0,
        "all_required_covered": True,
        "critical_misses": [],
        "covered_element_ids": ["rule"],
        "missing_element_ids": [],
    }


@pytest.mark.unit
def test_elements_are_cumulative_and_critical_misses_are_named():
    question = _question(
        required_elements=[
            {
                "id": "duty",
                "label": "обязанность",
                "acceptable_chunk_ids": ["doc.pdf#1"],
                "critical": True,
            },
            {
                "id": "exception",
                "label": "исключение",
                "acceptable_chunk_ids": ["doc.pdf#3"],
                "critical": True,
            },
            {
                "id": "note",
                "label": "пояснение",
                "acceptable_chunk_ids": ["doc.pdf#4"],
                "critical": False,
            },
        ]
    )

    score = score_context([_passage("doc.pdf", 1)], question.required_elements)

    assert score["coverage"] == pytest.approx(1 / 3)
    assert score["all_required_covered"] is False
    assert score["covered_element_ids"] == ["duty"]
    assert score["missing_element_ids"] == ["exception", "note"]
    assert score["critical_misses"] == ["exception"]


@pytest.mark.unit
def test_missing_required_elements_is_rejected():
    with pytest.raises(ValidationError):
        _question(required_elements=[])


@pytest.mark.unit
def test_duplicate_element_ids_are_rejected():
    element = {
        "id": "rule",
        "label": "правило",
        "acceptable_chunk_ids": ["doc.pdf#1"],
        "critical": True,
    }
    with pytest.raises(ValidationError):
        _question(required_elements=[element, element])


@pytest.mark.unit
def test_loader_rejects_unannotated_record(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Какое правило применяется?",
                "chunk_id": "doc.pdf#1",
                "relevant_chunk_ids": ["doc.pdf#1"],
                "source": "doc.pdf",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_elements"):
        load_calibration_gt(path)


@pytest.mark.unit
def test_binary_hit_preserves_or_semantics_of_relevant_chunks():
    assert binary_hit([_passage("doc.pdf", 2)], ["doc.pdf#1", "doc.pdf#2"])
    assert not binary_hit([_passage("doc.pdf", 3)], ["doc.pdf#1", "doc.pdf#2"])


@pytest.mark.unit
def test_audit_summary_has_all_eight_cells_and_risk_numerators():
    records = [
        {
            "question": "q1",
            "escalated": False,
            "simple_hit": True,
            "complex_hit": True,
            "simple_coverage": 1.0,
            "complex_coverage": 1.0,
            "final_coverage": 1.0,
            "terminal_route": "generate",
            "terminal_reason": "context_sufficient",
            "final_critical_misses": [],
            "simple_latency_ms": 10.0,
            "complex_latency_ms": 100.0,
        },
        {
            "question": "q2",
            "escalated": False,
            "simple_hit": True,
            "complex_hit": False,
            "simple_coverage": 0.5,
            "complex_coverage": 0.0,
            "final_coverage": 0.5,
            "terminal_route": "generate",
            "terminal_reason": "context_sufficient",
            "final_critical_misses": ["exception"],
            "simple_latency_ms": 20.0,
            "complex_latency_ms": 200.0,
        },
        {
            "question": "q3",
            "escalated": True,
            "simple_hit": False,
            "complex_hit": True,
            "simple_coverage": 0.0,
            "complex_coverage": 1.0,
            "final_coverage": 1.0,
            "terminal_route": "generate",
            "terminal_reason": "complex_sufficient",
            "final_critical_misses": [],
            "simple_latency_ms": 30.0,
            "complex_latency_ms": 300.0,
        },
    ]

    summary = summarize_audit(records)

    assert len(summary["matrix"]) == 8
    assert summary["matrix"]["not_escalated|simple_hit|complex_hit"] == 1
    assert summary["matrix"]["not_escalated|simple_hit|complex_miss"] == 1
    assert summary["matrix"]["escalated|simple_miss|complex_hit"] == 1
    assert summary["unsafe_generate"] == {"count": 1, "rate": pytest.approx(1 / 3)}
    assert summary["critical_miss_generate"] == {
        "count": 1,
        "rate": pytest.approx(1 / 3),
    }
    assert summary["routes"] == {"generate": 3}
    assert summary["reasons"] == {
        "complex_sufficient": 1,
        "context_sufficient": 2,
    }
    assert summary["latency_ms"]["simple"]["p95"] == 30.0
    assert summary["latency_ms"]["complex"]["p50"] == 200.0


@pytest.mark.unit
def test_audit_record_uses_actual_terminal_context_after_escalation():
    question = _question(
        required_elements=[
            {
                "id": "rule",
                "label": "правило",
                "acceptable_chunk_ids": ["doc.pdf#1"],
                "critical": True,
            },
            {
                "id": "exception",
                "label": "исключение",
                "acceptable_chunk_ids": ["doc.pdf#3"],
                "critical": True,
            },
        ]
    )
    simple = {
        "route_decision": "complex",
        "route_reason": "enumeration_intent",
        "final_context": [_passage("doc.pdf", 1)],
    }
    complex_ = {
        "route_decision": "generate",
        "route_reason": "complex_sufficient",
        "final_context": [_passage("doc.pdf", 1), _passage("doc.pdf", 3)],
    }

    record = make_audit_record(
        question,
        simple,
        complex_,
        simple_latency_ms=10.0,
        complex_latency_ms=100.0,
    )

    assert record["escalated"] is True
    assert record["simple_coverage"] == 0.5
    assert record["complex_coverage"] == 1.0
    assert record["final_coverage"] == 1.0
    assert record["terminal_route"] == "generate"
    assert record["final_critical_misses"] == []


@pytest.mark.unit
def test_split_loader_joins_explicit_factoid_and_multi_element_annotations(tmp_path):
    base = tmp_path / "base.jsonl"
    rows = [
        {
            "question": "Фактоидный вопрос?",
            "chunk_id": "doc.pdf#1",
            "relevant_chunk_ids": ["doc.pdf#1", "doc.pdf#2"],
            "source": "doc.pdf",
        },
        {
            "question": "Вопрос с условием и исключением?",
            "chunk_id": "doc.pdf#3",
            "relevant_chunk_ids": ["doc.pdf#3", "doc.pdf#4"],
            "source": "doc.pdf",
        },
    ]
    base.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    import hashlib

    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps(
            {
                "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "status": "reviewed",
                "count": 2,
                "factoid_indices": [1],
                "elements_by_index": {
                    "2": [
                        {
                            "id": "condition",
                            "label": "условие",
                            "acceptable_chunk_ids": ["doc.pdf#3"],
                            "critical": True,
                        },
                        {
                            "id": "exception",
                            "label": "исключение",
                            "acceptable_chunk_ids": ["doc.pdf#4"],
                            "critical": True,
                        },
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    questions = load_calibration_split(base, manifest)

    assert len(questions) == 2
    assert questions[0].required_elements[0].acceptable_chunk_ids == [
        "doc.pdf#1",
        "doc.pdf#2",
    ]
    assert [element.id for element in questions[1].required_elements] == [
        "condition",
        "exception",
    ]


@pytest.mark.unit
def test_split_loader_rejects_stale_or_incomplete_manifest(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text(
        json.dumps(
            {
                "question": "Вопрос?",
                "chunk_id": "doc.pdf#1",
                "relevant_chunk_ids": ["doc.pdf#1"],
                "source": "doc.pdf",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps(
            {
                "base_sha256": "0" * 64,
                "status": "reviewed",
                "count": 1,
                "factoid_indices": [1],
                "elements_by_index": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        load_calibration_split(base, manifest)


@pytest.mark.unit
def test_split_loader_rejects_draft_without_explicit_diagnostic_override(tmp_path):
    base = tmp_path / "base.jsonl"
    base.write_text(
        json.dumps(
            {
                "question": "Вопрос?",
                "chunk_id": "doc.pdf#1",
                "relevant_chunk_ids": ["doc.pdf#1"],
                "source": "doc.pdf",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    import hashlib

    manifest = tmp_path / "annotations.json"
    manifest.write_text(
        json.dumps(
            {
                "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "status": "draft",
                "count": 1,
                "factoid_indices": [1],
                "elements_by_index": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest is draft"):
        load_calibration_split(base, manifest)
    assert len(load_calibration_split(base, manifest, allow_draft=True)) == 1
