"""Synthetic offline checks for route evidence and grouped denominators."""

# ANCHOR: synthetic saved runs and outcomes → evidence selection and group metrics.
from eval.chunking_ab.norms import Chunk
from eval.chunking_ab.routes import Evidence, choose_route, scan_file
from eval.chunking_ab.step1 import Outcome
from eval.chunking_ab.step3 import count_context, grouped_metrics


def test_latest_observation_and_conflict_are_explicit():
    old = Evidence("simple", "old.json", 2, "2026-09-23", "path")
    new = Evidence("complex", "new.json", 4, "2026-10-16", "path")
    result = choose_route("Вопрос?", [old, new])
    assert result.route == "complex"
    assert result.conflicting_history
    assert result.selected == new


def test_equal_time_conflict_unknown_and_missing_unknown():
    a = Evidence("simple", "a", 1, "2026-10-16", "path")
    b = Evidence("complex", "b", 1, "2026-10-16", "path")
    assert choose_route("Q", [a, b]).route == "unknown"
    assert choose_route("Q", []).source == "c"


def test_scan_does_not_inherit_retrieval_only_path(tmp_path):
    p = tmp_path / "saved.json"
    p.write_text('{"path":"simple","results":[{"question":"Q","hit":true}]}')
    assert scan_file(p) == []
    p.write_text(
        '{"timestamp":"2026-10-16","results":[\n{"question":"Q","path":"complex"}]}'
    )
    rows = scan_file(p)
    assert rows[0][0] == "Q"
    assert rows[0][1].line == 2


def test_abstain_can_still_have_complex_attempt(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"question":"Q","path":"abstain","models":[{"stage":"complex"}]}\n')
    assert scan_file(p)[0][1].route == "complex"


def test_group_denominators_include_unlabelled_questions():
    rows = [
        Outcome((True, False), (False, False), (1.0, 0.0), 2, (0,)),
        Outcome((), (), (), 0, ()),
        Outcome((True,), (True,), (1.0,), 3, (1,)),
    ]
    m = grouped_metrics(rows, ["simple", "simple", "complex"])
    assert m["simple"]["questions"] == 2
    assert m["simple"]["norms"] == 2
    assert m["simple"]["q_found"] == 1
    assert m["unknown"]["questions"] == 0


def test_count_baseline_has_no_character_budget():
    chunks = [Chunk("doc", "x" * 10000, "body", None)] * 25
    ctx = count_context(list(range(25)), chunks, 24)
    assert len(ctx) == 24
    assert sum(len(c.text) for c in ctx) == 240000


def test_aggregate_usage_dict_is_not_attempt_list(tmp_path):
    p = tmp_path / "summary.json"
    p.write_text('{"question":"Q","usage":{"total":4},"path":"simple"}')
    assert scan_file(p)[0][1].route == "simple"
