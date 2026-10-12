"""Merged complex candidate: rerank order + anchors (spec 2026-10-12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v7 import pack_context as pc
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


@pytest.mark.parametrize("name,targets", RECORDED)
def test_recorded_targets_reach_merged_candidate(name, targets):
    merged = _candidates(_load(name))[0]["passages"]
    ids = {passage_identity(p) for p in merged}
    assert [t for t in targets if t not in ids] == []


# ─── assembly rule ─────────────────────────────────────────────────────────

PLAN = {"mmr_lambda": 0.7}


def _p(cid, rerank=None, vector=0.5, source="doc.pdf"):
    p = {
        "chunk_id": cid,
        "doc_id": source,
        "vector_score": vector,
        "metadata": {"source": source},
    }
    if rerank is not None:
        p["rerank_score"] = rerank
    return p


def _state(complex_passages, anchors=None, stages=("simple", "complex")):
    attempts = [{"stage": s, "passages": [], "attempt_plan": PLAN} for s in stages]
    if "complex" in stages:
        attempts[-1]["passages"] = complex_passages
    state = {
        "query": "q",
        "active_query": "q",
        "plan": PLAN,
        "retrieval_attempts": attempts,
    }
    if anchors is not None:
        state["fallback_snapshot"] = {
            "passages": anchors,
            "plan": PLAN,
            "active_query": "q",
            "origin": "fallback_snapshot",
            "packed": True,
            "pack_status": "ok",
        }
    return state


def _ids(state):
    return [p["chunk_id"] for p in _candidates(state)[0]["passages"]]


@pytest.fixture
def small_limits(monkeypatch):
    monkeypatch.setattr(pc.v7_config, "SIMPLE_TOP_K", 2)
    monkeypatch.setattr(pc.v7_config, "FINAL_MERGE_TOP_K", 4)


def test_anchors_lead_in_snapshot_order(small_limits):
    anchors = [_p("a1", vector=0.1), _p("a2", vector=0.9), _p("a3")]
    pool = [_p(f"c{i}", rerank=0.9 - i / 100) for i in range(5)]
    ids = _ids(_state(pool, anchors))
    assert ids[:2] == ["a1", "a2"]
    assert len(ids) == 4


def test_pool_chunk_equal_to_anchor_not_duplicated(small_limits):
    anchors = [_p("a1"), _p("a2")]
    pool = [_p("a1", rerank=0.99), _p("c1", rerank=0.5), _p("c2", rerank=0.4)]
    assert _ids(_state(pool, anchors)) == ["a1", "a2", "c1", "c2"]


def test_no_snapshot_fills_all_slots_from_pool(small_limits):
    pool = [_p(f"c{i}", rerank=0.9 - i / 100, source=f"d{i}.pdf") for i in range(6)]
    assert _ids(_state(pool)) == ["c0", "c1", "c2", "c3"]


def test_short_snapshot_leaves_more_room_for_pool(small_limits):
    pool = [_p(f"c{i}", rerank=0.9 - i / 100, source=f"d{i}.pdf") for i in range(6)]
    assert _ids(_state(pool, [_p("a1")])) == ["a1", "c0", "c1", "c2"]


def test_empty_complex_attempt_leaves_merged_empty(small_limits):
    """Anchors alone would re-pack the fallback snapshot; let it answer."""
    assert _ids(_state([], [_p("a1"), _p("a2")])) == []


def test_pool_of_anchor_duplicates_leaves_merged_empty(small_limits):
    assert _ids(_state([_p("a1", rerank=0.9)], [_p("a1"), _p("a2")])) == []


def test_without_complex_attempt_pool_is_empty(small_limits):
    state = _state([], [_p("a1")], stages=("simple",))
    state["retrieval_attempts"][0]["passages"] = [_p("s1", vector=0.99)]
    assert _ids(state) == []


def test_simple_attempt_not_in_pool(small_limits):
    state = _state([_p("c1", rerank=0.5)], [_p("a1")])
    state["retrieval_attempts"][0]["passages"] = [_p("s1", vector=0.99)]
    assert "s1" not in _ids(state)


def test_high_rerank_low_vector_beats_inverse(small_limits):
    pool = [
        _p("hi_vec", rerank=0.10, vector=0.95, source="x.pdf"),
        _p("hi_rr", rerank=0.95, vector=0.10, source="y.pdf"),
        _p("mid", rerank=0.50, vector=0.50, source="z.pdf"),
    ]
    ids = _ids(_state(pool, [_p("a1"), _p("a2")]))
    assert ids == ["a1", "a2", "hi_rr", "mid"]


def test_without_rerank_falls_back_to_vector_order(small_limits):
    pool = [_p(f"c{i}", vector=0.1 * i, source=f"d{i}.pdf") for i in range(5)]
    assert _ids(_state(pool, [_p("a1"), _p("a2")])) == ["a1", "a2", "c4", "c3"]


def test_duplicate_in_pool_counted_once(small_limits):
    pool = [_p("c1", rerank=0.9), _p("c1", rerank=0.9), _p("c2", rerank=0.5)]
    assert _ids(_state(pool, [_p("a1"), _p("a2")])) == ["a1", "a2", "c1", "c2"]
