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


def test_empty_and_failed_eval_records_keep_accounting(tmp_path, monkeypatch):
    import json

    import eval.run_v7_eval as runner

    rows = runner.load_dataset(runner.DATASET_PATH)[:2]
    monkeypatch.setattr(runner, "load_dataset", lambda _: rows)
    monkeypatch.setattr(runner, "get_vector_store_backend", lambda **_: None)
    monkeypatch.setattr(runner, "build_full_v7_runtime", lambda _: None)
    monkeypatch.setattr(runner, "build_graph", lambda **_: Mock())
    monkeypatch.setattr(runner, "default_writer", lambda: None)
    base = runner.run_query(Mock(invoke=lambda _: {"llm_usage": []}), "q")
    base.update(
        answer="",
        elapsed_sec=4,
        route_decision="abstain",
        route_reason="complex_exhausted",
        abstain_reason="no context",
    )
    failed = {
        **base,
        "answer": "technical failure",
        "generation_failed": True,
        "usage_unknown": True,
        "generation_error": {"error_type": "ValueError"},
    }
    monkeypatch.setattr(runner, "run_query", Mock(side_effect=[base, failed]))
    output = tmp_path / "report.json"
    runner.run(output=output, skip_judge=True)
    result = json.loads(output.read_text())
    assert len(result["results"]) == 2
    assert result["results"][0]["elapsed_sec"] == 4
    assert result["results"][0]["abstain_reason"] == "no context"
    assert result["aggregate"]["failed"] == 1
    assert result["aggregate"]["cost"]["queries"] == 2
    assert result["valid_results"] == 0
    assert result["aggregate"]["cost"]["cost_incomplete"] is True
    assert result["aggregate"]["cost"]["usage_unknown_queries"] == 1


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
