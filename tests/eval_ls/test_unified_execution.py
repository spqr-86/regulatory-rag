"""Exercise the real runner/SDK evaluator adapter using entirely local providers."""

# ANCHOR: fake SDK streaming + actual target callbacks → inspectable real artifacts.

import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from langsmith.schemas import Example

from eval.ls import run
from eval.ls.dataset import DEFAULT_CSV, DEFAULT_LABELS, build_examples

pytestmark = pytest.mark.unit


def test_runner_streams_accounting_and_failure_ids(tmp_path, monkeypatch):
    import langsmith
    from langsmith.evaluation.evaluator import _normalize_evaluator_func

    from config.settings import settings

    now = datetime.now(timezone.utc)
    examples = [
        Example(
            **e,
            created_at=now,
            modified_at=now,
            dataset_id="00000000-0000-0000-0000-000000000001",
        )
        for e in build_examples(DEFAULT_CSV, DEFAULT_LABELS)[:2]
    ]
    snapshot = tmp_path / run.LABELS_SNAPSHOT
    snapshot.mkdir()
    (snapshot / "snapshot-report.json").write_text("{}")
    monkeypatch.setattr(settings, "CHROMA_DB_PATH", str(snapshot))
    target_calls = []

    def target_factory(*, run_id, checkpoint):
        def target(inputs):
            target_calls.append(inputs["case_id"])
            if len(target_calls) == 2:
                checkpoint(
                    {
                        "inputs": inputs,
                        "query_id": "failed-query",
                        "run_id": run_id,
                        "target_error": "TimeoutError",
                        "cost_incomplete": True,
                        "elapsed_sec": 5,
                    }
                )
                raise TimeoutError("fake provider")
            outputs = {
                "answer": "ok",
                "cost_usd": 0.1,
                "elapsed_sec": 2,
                "query_id": "query-a",
                "run_id": run_id,
                "chunk_ids": examples[0].outputs["relevant_chunk_ids"],
            }
            checkpoint({"inputs": inputs, "outputs": outputs})
            return outputs

        return target

    class Results:
        experiment_name = "fake-sdk"
        url = "offline://fake-sdk"

        def __init__(self, target, data, evaluators):
            self.target, self.data, self.evaluators = target, data, evaluators

        def __iter__(self):
            for i, ex in enumerate(self.data):
                try:
                    outputs, error = self.target(ex.inputs), None
                except TimeoutError:
                    outputs, error = {}, "TimeoutError"
                actual = NS(id=str(i), inputs=ex.inputs, outputs=outputs, error=error)
                scores = []
                for evaluator in self.evaluators:
                    # Exercise installed SDK signature normalization without trace upload.
                    normalized, _ = _normalize_evaluator_func(evaluator)
                    scores += [NS(**s) for s in normalized(actual, ex)["results"]]
                yield {
                    "run": actual,
                    "example": ex,
                    "evaluation_results": {"results": scores},
                }

    class Client:
        def list_examples(self, **kw):
            return examples

        def evaluate(self, target, **kw):
            assert kw["blocking"] is False and kw["max_concurrency"] == 1
            return Results(target, kw["data"], kw["evaluators"])

    monkeypatch.setattr(langsmith, "Client", Client)
    monkeypatch.setattr(run, "make_target", target_factory)
    directory = tmp_path / "artifacts"
    assert (
        run.run_experiment(skip_judge=True, limit=2, output_dir=directory) == "fake-sdk"
    )
    assert target_calls == [e.metadata["case_id"] for e in examples]
    records = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text().splitlines()
    ]
    summary = json.loads((directory / "summary.json").read_text())
    assert records[0]["scores"]["retrieval_hit"]["score"] == 1
    assert records[1]["outputs"]["query_id"] == "failed-query"
    assert records[1]["outputs"]["run_id"] == records[0]["outputs"]["run_id"]
    assert summary["assigned"] == summary["completed"] == 2
    assert summary["target_failed"] == 1 and summary["cost_incomplete"]
    assert summary["metrics"]["correctness"]["eligible"] == 2
    assert summary["metrics"]["correctness"]["scored"] == 0
    assert summary["latency_sec"]["p50"] == 3.5


def test_make_target_uses_single_telemetry_call_and_shared_ids(monkeypatch):
    import src.backends.vector_store as backend
    import src.v7.bridge as bridge
    import src.v7.graph as graph_module
    import src.v7.runner as telemetry
    from eval.ls.target import make_target

    graph = Mock()
    monkeypatch.setattr(backend, "get_vector_store_backend", lambda **kw: None)
    monkeypatch.setattr(bridge, "build_full_v7_runtime", lambda _: None)
    monkeypatch.setattr(
        graph_module, "build_graph", lambda **kw: NS(compile=lambda: graph)
    )
    monkeypatch.setattr(telemetry, "default_writer", lambda: None)
    seen = []

    def run_query(pipeline, question, **kw):
        seen.append(kw)
        return {"answer": "ok"}, kw["query_id"]

    monkeypatch.setattr(telemetry, "run_query", run_query)
    checkpoints = []
    result = make_target("batch-a", checkpoints.append)(
        {"question": "q", "case_id": "A", "example_id": "example-a"}
    )
    assert len(seen) == 1 and seen[0]["source"] == "eval"
    assert result["query_id"] == seen[0]["query_id"]
    assert result["run_id"] == seen[0]["run_id"] == "batch-a"
    assert checkpoints[0]["outputs"] == result


def test_groundedness_prompt_contract_and_provider_usage(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.prompts import ChatPromptTemplate

    from eval.ls.artifacts import Artifacts, summarize_records
    from eval.ls.audit import AuditedModel, audited
    from eval.ls.groundedness import PROMPT_PATH, Variables, make_groundedness_evaluator

    assert set(
        ChatPromptTemplate.from_template(PROMPT_PATH.read_text()).input_variables
    ) == set(Variables.model_fields)
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        # Proposed v2 shape for the same parser-only usage fixture, not alignment data.
        content='{"contract_version":2,"checks":[{"answer_quote":"ok","evidence":[{"fragment_id":1,"quote":"ok"}],"status":"supported","reason":"same"}]}',
        usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20},
    )
    usages = {}
    evaluator = audited(
        make_groundedness_evaluator(AuditedModel(llm, "gpt-4o")),
        ("groundedness",),
        Artifacts(tmp_path, {"run_id": "assess-a"}),
        usages,
    )
    result = evaluator(
        NS(id="r", inputs={"question": "q"}, outputs={"answer": "ok", "context": "ok"}),
        NS(id="e", outputs={}, metadata={}),
    )
    assert result["results"][0]["score"] == 1
    assert result["results"][0]["evaluator_info"]["assessment_id"] == "assess-a"
    summary = summarize_records(
        [{"outputs": {"answer": "ok", "cost_usd": 0.1}, "judge_usage": usages["r"]}],
        assigned=1,
    )
    assert summary["judge_cost"]["cost_usd"] > 0
    assert summary["total_cost_usd"] > summary["pipeline_cost"]["total_usd"]
