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


def test_real_false_premises_are_retrieval_labelled():
    examples = build_examples(
        ROOT / "tests/dataset.csv",
        ROOT / "eval/runs/gt_label_j7ceu4r4_2026-10-18/golden_retrieval_labeled.jsonl",
    )
    by_id = {e["metadata"]["case_id"]: e for e in examples}
    assert by_id["G043"]["outputs"]["relevant_chunk_ids"] == ["2464.md#209"]
    assert by_id["G043"]["metadata"]["retrieval_labelled"] is True
    assert by_id["G043"]["metadata"]["case_type"] == "false_premise"
    assert by_id["G043"]["metadata"]["labels_sha256"]
    assert by_id["G071"]["metadata"]["retrieval_labelled"] is True


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
    # No support in the generator context means the claim is not grounded.
    payload["checks"][0].update(context_quote=None, status="unverifiable")
    assert parse_groundedness(json.dumps(payload), answer, context)["score"] == 0
    payload["checks"] = []
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


@pytest.mark.parametrize("argv", [[], ["--prefix", "x"], ["--rejudge", "exp"]])
def test_cli_refuses_paid_run_without_explicit_flag(argv, monkeypatch):
    import eval.ls.run as run

    def forbidden(*args, **kwargs):
        raise AssertionError("paid path reached")

    monkeypatch.setattr(run, "run_experiment", forbidden)
    monkeypatch.setattr(run, "rejudge", forbidden)
    with pytest.raises(SystemExit):
        run.main(argv)


def test_cli_paid_flag_reaches_run(monkeypatch):
    import eval.ls.run as run

    seen = []
    monkeypatch.setattr(run, "run_experiment", lambda *a, **k: seen.append(k))
    run.main(["--paid", "--prefix", "x"])
    run.main(["--dry-run"])
    assert [k["dry_run"] for k in seen] == [False, True]


def test_labels_snapshot_comes_from_labels_manifest(tmp_path):
    from eval.ls.dataset import DEFAULT_LABELS, labels_index_snapshot

    assert labels_index_snapshot(ROOT / DEFAULT_LABELS) == "build-j7ceu4r4"
    labels = tmp_path / "labels.jsonl"
    labels.write_text("")
    assert labels_index_snapshot(labels) is None
    (tmp_path / "labels_manifest.json").write_text('{"index_snapshot": "build-old"}')
    assert labels_index_snapshot(labels) == "build-old"


def test_paid_run_refuses_labels_from_another_snapshot(tmp_path, monkeypatch):
    import shutil

    import eval.ls.run as run
    from config.settings import settings
    from eval.ls.dataset import DEFAULT_LABELS

    snapshot = tmp_path / run.LABELS_SNAPSHOT
    snapshot.mkdir()
    (snapshot / "snapshot-report.json").write_text("{}")
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(snapshot))
    old = tmp_path / "old" / "labels.jsonl"
    old.parent.mkdir()
    shutil.copy(ROOT / DEFAULT_LABELS, old)
    with pytest.raises(ValueError, match="labels_manifest"):
        run.run_experiment(cases=["G001"], labels_path=old)


def test_groundedness_quote_ignores_markdown_bold_in_answer():
    from eval.ls.groundedness import parse_groundedness

    answer = "Срок составляет **«не менее 16 часов»** [Фрагмент 1]."
    context = "продолжительностью не менее 16 часов"
    payload = {
        "checks": [
            {
                "answer_quote": "Срок составляет «не менее 16 часов»",
                "context_quote": "не менее 16 часов",
                "status": "supported",
                "reason": "same number",
            }
        ]
    }
    assert parse_groundedness(json.dumps(payload), answer, context)["score"] == 1
