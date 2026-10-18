"""Offline integration of selection, checkpoints, audit and frozen rejudging."""

import json
from types import SimpleNamespace as NS

import pytest

from eval.ls import run
from eval.ls.artifacts import Artifacts
from eval.ls.audit import audited

pytestmark = pytest.mark.unit


def example(cid="A", *, question="Question", truth="Original", oos=""):
    return NS(
        id=cid,
        inputs={"question": question},
        outputs={"ground_truth": truth, "oos_type": oos},
        metadata={"case_id": cid},
        modified_at="v1",
    )


def test_dry_run_never_builds_clients_or_models(tmp_path, monkeypatch):
    import langsmith

    monkeypatch.setattr(langsmith, "Client", lambda: pytest.fail("network client"))
    monkeypatch.setattr(run, "make_target", lambda **kw: pytest.fail("graph"))
    monkeypatch.setattr(run, "judge_evaluators", lambda: pytest.fail("judge"))
    assert (
        run.run_experiment(
            "verified", "dry", False, 2, dry_run=True, output_dir=tmp_path / "dry"
        )
        == "dry-run"
    )
    manifest = json.loads((tmp_path / "dry/manifest.json").read_text())
    assert len(manifest["references"]) == 2
    assert manifest["dry_run"] is True
    assert "API_KEY" not in json.dumps(manifest)


def test_cases_selected_before_limit_and_unknown_ids_rejected():
    examples = [example("A"), example("B"), example("C")]
    assert [e.id for e in run.select_examples(examples, ["B", "C"], 1)] == ["B"]
    with pytest.raises(ValueError, match="Unknown"):
        run.select_examples(examples, ["missing"], None)
    with pytest.raises(ValueError, match="positive"):
        run.select_examples(examples, None, 0)


def test_audit_calls_code_evaluator_with_its_named_arguments(tmp_path):
    from eval.ls.evaluators import retrieval_hit

    sink = Artifacts(tmp_path, {})
    result = audited(retrieval_hit, ("retrieval_hit",), sink, {})(
        NS(id="r", inputs={}, outputs={"answer": "ok", "chunk_ids": ["x"]}),
        NS(id="a", metadata={}, outputs={"relevant_chunk_ids": ["x"]}),
    )
    assert result["results"][0]["score"] == 1


def test_audit_provider_error_is_unresolved_and_premise_alias_preserved(tmp_path):
    def correctness(inputs, outputs, reference_outputs):
        raise TimeoutError("provider")

    result = audited(
        correctness, ("correctness", "forbidden_claim"), Artifacts(tmp_path, {}), {}
    )(NS(id="r", inputs={}, outputs={"answer": "no"}), example(oos="false_premise"))
    scores = {s["key"]: s for s in result["results"]}
    assert scores["correctness"]["score"] is None
    assert scores["premise_corrected"]["score"] is None


def test_frozen_references_join_by_id_even_when_question_changes():
    seen = []

    def evaluator(inputs, outputs, reference_outputs):
        seen.append(reference_outputs)
        return {"key": "correctness", "score": 1}

    wrapped = run.with_frozen_references(evaluator, {"A": example(truth="Frozen")})
    wrapped(
        NS(inputs={"question": "Edited"}, outputs={"answer": "ok"}),
        example(truth="New"),
    )
    assert seen == [{"ground_truth": "Frozen", "oos_type": ""}]
    with pytest.raises(ValueError, match="reference"):
        wrapped(NS(inputs={}, outputs={}), example("unknown"))


def test_streamed_row_is_durable_before_later_interruption(tmp_path):
    from eval.ls.run import consume_results

    def rows():
        yield {
            "run": NS(
                id="r",
                outputs={"answer": "ok", "cost_usd": 0, "elapsed_sec": 1},
                error=None,
            ),
            "example": example(),
            "evaluation_results": {
                "results": [NS(key="correctness", score=1, comment="pass")]
            },
        }
        raise KeyboardInterrupt

    sink = Artifacts(tmp_path, {})
    with pytest.raises(KeyboardInterrupt):
        consume_results(rows(), sink, [example(), example("B")], {})
    assert json.loads((tmp_path / "results.jsonl").read_text())["case_id"] == "A"
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["completed"] == 1 and summary["pending"] == 1


def test_unlabelled_retrieval_is_unresolved_coverage():
    from eval.ls.artifacts import summarize_records

    result = summarize_records(
        [
            {
                "outputs": {"answer": "ok"},
                "scores": {"retrieval_hit": {"score": None, "comment": "no_labels"}},
            }
        ],
        assigned=1,
    )
    assert result["metrics"]["retrieval_hit"]["eligible"] == 1
    assert result["metrics"]["retrieval_hit"]["unresolved"] == 1


def test_manifest_never_serializes_database_credentials(tmp_path, monkeypatch):
    from src.v7.config import v7_config

    monkeypatch.setattr(v7_config, "TELEMETRY_PG_DSN", "postgres://user:SECRET@host/db")
    run.run_experiment(dry_run=True, output_dir=tmp_path)
    raw = (tmp_path / "manifest.json").read_text()
    assert "SECRET" not in raw and "TELEMETRY_PG_DSN" not in raw
    manifest = json.loads(raw)
    assert "prompts/agents/generate_answer_v7.j2" in manifest["prompt_sha256"]


@pytest.mark.parametrize("current", [False, True])
def test_rejudge_uses_versioned_references_without_building_target(
    tmp_path, monkeypatch, current
):
    import langsmith

    calls = []
    original = example(truth="Original")
    updated = example(question="Changed question", truth="Updated")
    source_run = NS(
        id="run-a",
        reference_example_id="A",
        inputs={"question": "Question"},
        outputs={
            "answer": "ok",
            "context": "evidence",
            "cost_usd": 0.5,
            "elapsed_sec": 1,
        },
        error=None,
    )

    def correctness(inputs, outputs, reference_outputs):
        calls.append(reference_outputs["ground_truth"])
        return [{"key": "correctness", "score": 1, "comment": "pass"}]

    class Results:
        def __iter__(self):
            scores = []
            for e in self.evaluators:
                scores += [NS(**s) for s in e(source_run, original)["results"]]
            yield {
                "run": source_run,
                "example": original,
                "evaluation_results": {"results": scores},
            }

    class Client:
        def read_project(self, **kw):
            return NS(
                metadata={"dataset_version": "v1", "index_snapshot": "source-snapshot"},
                reference_dataset_id="dataset",
            )

        def list_runs(self, **kw):
            return [source_run]

        def list_examples(self, **kw):
            return [original if kw.get("as_of") == "v1" else updated]

        def evaluate(self, experiment, **kw):
            assert experiment == "source"
            result = Results()
            result.evaluators = kw["evaluators"]
            return result

    monkeypatch.setattr(langsmith, "Client", Client)
    monkeypatch.setattr(run, "judge_evaluators", lambda: [correctness])
    monkeypatch.setattr(
        run, "make_target", lambda **kw: pytest.fail("RAG must not run")
    )
    run.rejudge("source", output_dir=tmp_path, current_references=current)
    assert calls == ["Updated" if current else "Original"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["reference_mode"] == ("current" if current else "original")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["reused_pipeline_cost"]["total_usd"] == 0.5
    assert summary["total_cost_usd"] == 0


def test_legacy_output_fails_before_any_execution(monkeypatch, tmp_path):
    from eval.run_v7_eval import run as legacy

    monkeypatch.setattr(
        run, "run_experiment", lambda **kw: pytest.fail("must fail before execution")
    )
    with pytest.raises(ValueError, match="output-dir"):
        legacy(output=tmp_path / "old.jsonl")


def test_empty_answer_never_gets_oos_credit_even_with_abstain_route():
    from eval.ls.evaluators import oos_handled

    assert (
        oos_handled(
            {"answer": "", "route_decision": "abstain"}, {"oos_type": "out_of_scope"}
        )["score"]
        is None
    )
