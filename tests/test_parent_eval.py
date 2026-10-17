"""Offline tests: parent diagnostics survive eval serialization and gate output."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from eval import run_v7_eval as evaluator
from src.v7.config import v7_config


@pytest.mark.parametrize("enabled,replaced", [(True, 0), (False, 0), (True, 2)])
def test_eval_gate_before_output(tmp_path, monkeypatch, enabled, replaced):
    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", enabled)
    dataset = evaluator.load_dataset(evaluator.DATASET_PATH)[:1]
    monkeypatch.setattr(evaluator, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(
        evaluator,
        "run_query",
        lambda *args, **kwargs: {
            "answer": "ok",
            "context": "",
            "path": "simple",
            "elapsed_sec": 0,
            "cost_usd": 0,
            "parent_context_stats": [{"replaced": replaced}],
        },
    )
    monkeypatch.setattr(evaluator, "get_vector_store_backend", Mock())
    monkeypatch.setattr(evaluator, "build_full_v7_runtime", Mock())
    monkeypatch.setattr(evaluator, "build_graph", Mock())
    monkeypatch.setattr(evaluator, "default_writer", lambda: None)
    output = tmp_path / "result.json"
    if enabled and replaced == 0:
        with pytest.raises(RuntimeError, match="parent_replaced_total.*0"):
            evaluator.run(output=output, skip_judge=True)
        assert not output.exists()
    else:
        evaluator.run(output=output, skip_judge=True)
        assert output.exists()
        if enabled:
            import json

            assert (
                json.loads(output.read_text())["aggregate"]["parent_replaced_total"]
                == replaced
            )


def test_eval_carries_and_sums_parent_debug(monkeypatch):
    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", True)
    stats = {
        "replaced": 2,
        "children": 1,
        "deduplicated": 3,
        "truncated": 1,
        "slots_lost": 4,
        "unavailable_reason": None,
    }
    graph = SimpleNamespace(
        invoke=lambda _: {"parent_context_stats": [stats], "answer": "ok"}
    )
    row = evaluator.record_fields(evaluator.run_query(graph, "q"))
    assert row["parent_context_stats"] == [stats]
    assert row["parent_truncated"] == 1
    assert row["parent_slots_lost"] == 4
    total = evaluator.summarize_parents([row, row])
    assert total["parent_replaced_total"] == 4
    assert total["parent_children_total"] == 2
    assert total["parent_deduplicated_total"] == 6
    assert total["parent_truncated_total"] == 2
    assert total["parent_slots_lost_total"] == 8
