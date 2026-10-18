"""Behavioral regressions for the unified evaluation contract, without providers."""

import json
from pathlib import Path

import pytest

from eval.ls.dataset import build_examples
from eval.ls.evaluators import oos_handled, retrieval_hit
from eval.ls.judge import make_correctness_evaluator
from eval.ls.target import state_to_outputs

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_false_premise_is_not_a_keyword_refusal():
    ref = {"oos_type": "false_premise", "ground_truth": "Минимум 3 человека."}
    answer = {"answer": "Нет, комиссия должна состоять минимум из 5 человек."}
    result = oos_handled(answer, ref)
    assert result["score"] is None
    assert result["comment"] == "false_premise_scored_by_correctness"


def test_false_premise_reaches_fact_judge_and_can_fail():
    from langchain_core.messages import AIMessage

    class Judge:
        def invoke(self, messages):
            return AIMessage(
                content=json.dumps(
                    {
                        "facts": [
                            {
                                "fact": "Минимум 3 человека",
                                "status": "contradicted",
                                "answer_quote": "минимум 5 человек",
                            }
                        ],
                        "errors": [],
                        "reasoning": "wrong number",
                        "forbidden_claims_asserted": [],
                    }
                )
            )

    results = make_correctness_evaluator(Judge())(
        {"question": "Минимум 5 человек?"},
        {"answer": "Нет, минимум 5 человек."},
        {"oos_type": "false_premise", "ground_truth": "Минимум 3 человека."},
    )
    assert results[0]["score"] == 0


def test_failed_generation_is_not_an_oos_refusal():
    assert (
        oos_handled(
            {"answer": "", "generation_failed": True}, {"oos_type": "out_of_scope"}
        )["score"]
        is None
    )


def test_unlabelled_false_premise_is_not_a_retrieval_miss():
    result = retrieval_hit(
        {"answer": "Исправление", "chunk_ids": []},
        {"oos_type": "false_premise", "relevant_chunk_ids": []},
    )
    assert result["score"] is None and result["comment"] == "no_labels"


def test_real_false_premises_report_missing_retrieval_labels():
    examples = build_examples(
        ROOT / "tests/dataset.csv",
        ROOT / "eval/runs/gt_label_j7ceu4r4_2026-10-18/golden_retrieval_labeled.jsonl",
    )
    by_id = {e["metadata"]["case_id"]: e for e in examples}
    # G071 awaits Petr's decision (eval/runs/false_premise_labels_2026-10-18/).
    assert by_id["G071"]["outputs"]["relevant_chunk_ids"] == []
    assert by_id["G071"]["metadata"]["retrieval_labelled"] is False
    assert by_id["G071"]["metadata"]["case_type"] == "false_premise"
    assert by_id["G071"]["metadata"]["labels_sha256"]
    assert by_id["G043"]["outputs"]["relevant_chunk_ids"] == ["2464.md#209"]
    assert by_id["G043"]["metadata"]["retrieval_labelled"] is True


def test_false_premise_preserves_supplied_labels(tmp_path):
    import csv

    rows = list(csv.DictReader((ROOT / "tests/dataset.csv").open()))
    row = next(r for r in rows if r["case_id"] == "G043")
    csv_path = tmp_path / "cases.csv"
    with csv_path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    labels = tmp_path / "labels.jsonl"
    labels.write_text(
        json.dumps({"question": row["question"], "relevant_chunk_ids": ["fixture#1"]})
    )
    assert build_examples(csv_path, labels)[0]["outputs"]["relevant_chunk_ids"] == [
        "fixture#1"
    ]


def test_summary_percentiles_use_percent_units():
    from eval.ls.artifacts import summarize_records

    rows = [{"outputs": {"answer": "ok", "elapsed_sec": x}} for x in range(1, 101)]
    summary = summarize_records(rows, assigned=100)
    assert summary["latency_sec"]["p50"] == pytest.approx(50.5)
    assert summary["latency_sec"]["p95"] == pytest.approx(95.05)


def test_target_retains_accounting_and_failure_details():
    usage = [
        {
            "model": "gpt-4o",
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "usage_unknown": True,
        }
    ]
    outputs = state_to_outputs(
        {
            "answer": "",
            "generation_failed": True,
            "technical_failure": True,
            "generation_error": {"error_type": "Timeout"},
            "retrieval_attempts": [{"stage": "complex"}],
            "llm_usage": usage,
        }
    )
    assert outputs["generation_failed"] and outputs["technical_failure"]
    assert outputs["usage_unknown"] and outputs["cost_incomplete"]
    assert outputs["path"] == "complex" and outputs["prompt_tokens"] == 4
    assert outputs["usage"] == usage


def test_groundedness_validates_evidence_and_distinguishes_unknown():
    from eval.ls.groundedness import parse_groundedness

    answer, context = (
        "Минимум 3 человека.",
        "Комиссия состоит не менее чем из 3 человек.",
    )
    payload = {
        "checks": [
            {
                "answer_quote": answer,
                "context_quote": context,
                "status": "supported",
                "reason": "same number",
            }
        ]
    }
    assert parse_groundedness(json.dumps(payload), answer, context)["score"] == 1
    payload["checks"][0]["context_quote"] = "Выдуманная цитата."
    with pytest.raises(ValueError, match="context_quote"):
        parse_groundedness(json.dumps(payload), answer, context)
    payload["checks"][0].update(context_quote=None, status="unverifiable")
    assert parse_groundedness(json.dumps(payload), answer, context)["score"] is None


def test_summary_keeps_failures_and_judge_abstentions_visible():
    from eval.ls.artifacts import summarize_records

    rows = [
        {
            "case_id": "A",
            "outputs": {"answer": "ok"},
            "scores": {"correctness": {"score": 1, "comment": "pass"}},
        },
        {
            "case_id": "B",
            "outputs": {},
            "error": "Timeout",
            "scores": {"correctness": {"score": None, "comment": "target_error"}},
        },
        {
            "case_id": "C",
            "outputs": {"answer": "x"},
            "scores": {"correctness": {"score": None, "comment": "judge_format_error"}},
        },
    ]
    summary = summarize_records(rows, assigned=3)
    assert summary["assigned"] == 3 and summary["target_failed"] == 1
    metric = summary["metrics"]["correctness"]
    assert metric["eligible"] == 3 and metric["scored"] == 1
    assert metric["unresolved"] == 2 and metric["coverage"] == pytest.approx(1 / 3)
    assert summarize_records([], assigned=2)["completed"] == 0


def test_checkpoint_survives_interruption_and_refuses_overwrite(tmp_path):
    from eval.ls.artifacts import Artifacts

    artifacts = Artifacts(tmp_path / "run", {"run_id": "test", "references": []})
    artifacts.append("results", {"case_id": "A", "outputs": {"answer": "ok"}})
    assert json.loads((tmp_path / "run/results.jsonl").read_text())["case_id"] == "A"
    with pytest.raises(FileExistsError):
        Artifacts(tmp_path / "run", {"run_id": "second"})


def test_artifacts_refuse_orphaned_existing_results(tmp_path):
    from eval.ls.artifacts import Artifacts

    original = '{"case_id":"old"}\n'
    (tmp_path / "results.jsonl").write_text(original)
    with pytest.raises(FileExistsError):
        Artifacts(tmp_path, {"run_id": "new"})
    assert (tmp_path / "results.jsonl").read_text() == original


def test_legacy_run_delegates_without_loading_a_second_graph(monkeypatch):
    import eval.ls.run as unified
    import eval.run_v7_eval as legacy

    seen = []
    monkeypatch.setattr(
        unified, "run_experiment", lambda *a, **kw: seen.append((a, kw))
    )
    # Fail before a real graph/embedding client can be built while migration is WIP.
    monkeypatch.setattr(
        legacy,
        "get_vector_store_backend",
        lambda **kw: pytest.fail("Legacy run must delegate before building a graph"),
        raising=False,
    )
    legacy.run(limit=2, skip_judge=True)
    assert len(seen) == 1 and seen[0][1]["limit"] == 2
