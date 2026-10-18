"""Offline regression cases from golden56 error analysis (2026-10-16)."""

from types import SimpleNamespace
from unittest.mock import Mock

from src.v7.bridge import make_generate_fn
from src.v7.nodes.generate_answer import generate_answer, set_generate_fn


def test_empty_generation_keeps_usage_and_failure_through_node():
    llm = Mock(model_name="mock-model")
    llm.invoke.return_value = SimpleNamespace(
        content="",
        usage_metadata={"input_tokens": 123, "output_tokens": 456},
        response_metadata={"finish_reason": "length"},
    )
    set_generate_fn(make_generate_fn(llm))
    try:
        result = generate_answer(
            {"query": "G026", "final_context": [{"text": "raw norm"}]}
        )
    finally:
        set_generate_fn(None)
    assert result["generation_failed"] is True
    assert "raw norm" not in result["answer"]
    assert result["llm_usage"][0]["completion_tokens"] == 456
    assert result["generation_error"]["finish_reason"] == "length"


def test_provider_exception_does_not_return_raw_context():
    llm = Mock(model_name="mock-model")
    llm.invoke.side_effect = TimeoutError("provider timeout")
    answer, usage = make_generate_fn(llm)("G026", "G026", [{"text": "raw norm"}])
    assert "raw norm" not in answer
    assert usage["generation_failed"]
    assert usage["usage_unknown"]
    assert usage["error_type"] == "TimeoutError"


def test_empty_and_failed_eval_records_keep_accounting(tmp_path):
    import json
    from types import SimpleNamespace as NS

    from eval.ls.artifacts import Artifacts
    from eval.ls.run import consume_results

    references = [
        NS(id=str(i), outputs={}, metadata={"case_id": str(i)}) for i in range(2)
    ]
    outputs = [
        {"answer": "", "elapsed_sec": 4, "cost_usd": 0, "abstain_reason": "no context"},
        {
            "answer": "technical failure",
            "elapsed_sec": 4,
            "cost_usd": 0,
            "generation_failed": True,
            "usage_unknown": True,
            "cost_incomplete": True,
            "generation_error": {"error_type": "ValueError"},
        },
    ]
    rows = [
        {
            "run": NS(id=str(i), outputs=o, error=None),
            "example": references[i],
            "evaluation_results": {"results": []},
        }
        for i, o in enumerate(outputs)
    ]
    summary = consume_results(
        rows, Artifacts(tmp_path, {}), references, {}, skip_judge=True
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["outputs"]["elapsed_sec"] == 4
    assert records[0]["outputs"]["abstain_reason"] == "no context"
    assert records[1]["outputs"]["generation_error"]["error_type"] == "ValueError"
    assert summary["target_failed"] == 2
    assert summary["pipeline_cost"]["incomplete_examples"] == 1
    assert summary["cost_incomplete"] is True


def test_telemetry_records_generation_failure():
    from src.v7.telemetry import build_event

    event = build_event(
        {"generation_failed": True, "generation_error": {"error_type": "TimeoutError"}},
        source="eval",
        latency_ms=3,
    )
    assert event["error"] == "generation_failed: TimeoutError"


def test_compiled_graph_keeps_generation_failure_and_usage():
    from src.v7.graph import build_graph
    from src.v7.runtime import V7Runtime

    usage = {
        "generation_failed": True,
        "error_type": "TimeoutError",
        "usage_unknown": True,
    }
    graph = build_graph(
        overrides={
            "intent_gate": lambda _: {"intent": "domain"},
            "router": lambda _: {},
            "rag_simple": lambda _: {},
            "evaluate_triage": lambda _: {
                "route_decision": "generate",
                "final_context": [{"text": "raw norm"}],
            },
        },
        runtime=V7Runtime(generate_simple=lambda *args: ("Сбой генерации", usage)),
    ).compile()
    result = graph.invoke({"query": "G026"})
    assert result["generation_failed"]
    assert result["generation_error"]["error_type"] == "TimeoutError"
    assert result["llm_usage"][0]["usage_unknown"]
