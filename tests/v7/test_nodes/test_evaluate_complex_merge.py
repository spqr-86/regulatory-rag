"""Merged complex candidate: rerank order + anchors (spec 2026-10-12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v7.nlp_core import passage_identity
from src.v7.nodes.evaluate_complex import _candidates

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "merge_candidates"

# Targets the reranker ranked high but the old merge dropped (trace reviews
# docs/evaluation/experiments/langsmith-learning-2026-10-09/).
RECORDED = [
    ("g001_program_a.json", ["2464.pdf#105", "2464.pdf#109"]),
    ("g020_drivers_medical.json", ["29н.pdf#319"]),
]


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.xfail(strict=True, reason="merge ignores rerank order; fixed in Task 2")
@pytest.mark.parametrize("name,targets", RECORDED)
def test_recorded_targets_reach_merged_candidate(name, targets):
    merged = _candidates(_load(name))[0]["passages"]
    ids = {passage_identity(p) for p in merged}
    assert [t for t in targets if t not in ids] == []
