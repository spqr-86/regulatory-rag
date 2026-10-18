"""Offline tests: parent diagnostics survive eval serialization and gate output."""

from types import SimpleNamespace

import pytest

from eval import run_v7_eval as evaluator
from src.v7.config import v7_config


@pytest.mark.parametrize("enabled,replaced", [(True, 0), (False, 0), (True, 2)])
def test_eval_gate_before_output(tmp_path, monkeypatch, enabled, replaced):
    monkeypatch.setattr(v7_config, "PARENT_CONTEXT_ENABLE", enabled)
    import json

    from eval.ls.artifacts import Artifacts
    from eval.ls.run import consume_results

    example = SimpleNamespace(id="a", outputs={}, metadata={"case_id": "A"})
    rows = [
        {
            "run": SimpleNamespace(
                id="run-a",
                outputs={"answer": "ok", "parent_replaced": replaced},
                error=None,
            ),
            "example": example,
            "evaluation_results": {"results": []},
        }
    ]
    sink = Artifacts(tmp_path, {})
    if enabled and replaced == 0:
        with pytest.raises(RuntimeError, match="parent_replaced_total.*0"):
            consume_results(rows, sink, [example], {}, require_parent_context=enabled)
        assert json.loads((tmp_path / "summary.json").read_text())["valid"] is False
    else:
        summary = consume_results(
            rows, sink, [example], {}, require_parent_context=enabled
        )
        assert summary["valid"] is True
        assert summary["parent_context"]["parent_replaced_total"] == replaced


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
