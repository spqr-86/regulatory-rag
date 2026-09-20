"""Fixed synthetic backend characterization captured before shared retrieval.

These fixtures test routing and evidence preservation, not model quality.
"""

# ANCHOR: execute the original graph against deterministic backend responses.
# Inputs: checked-in cases and config. Output: stable terminal evidence signature.

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.v7 import nlp_core, pack_context
from src.v7.config import v7_config
from src.v7.graph import build_graph
from src.v7.nodes import generate_answer, rag_complex, rag_simple, visual_enrichment

FIXTURE = Path(__file__).parent / "fixtures/shared_retrieval_baseline.json"
pytestmark = pytest.mark.unit


def run_case(case, monkeypatch):
    for key, value in case["config"].items():
        monkeypatch.setattr(v7_config, key, value)

    def search(query, filters=None, top_k=12):
        if case.get("error"):
            raise RuntimeError("recorded backend failure")
        stage = "simple" if top_k == v7_config.SIMPLE_TOP_K else "complex"
        return copy.deepcopy(case[stage])

    monkeypatch.setattr(rag_simple, "_vector_search", search)
    monkeypatch.setattr(rag_complex, "_vector_search", search)
    monkeypatch.setattr(nlp_core, "_bm25_index", None)
    monkeypatch.setattr(rag_simple, "_expand_fn", None)
    monkeypatch.setattr(rag_complex, "_rerank_fn", None)
    monkeypatch.setattr(rag_complex, "_section_fetch_fn", None)
    monkeypatch.setattr(pack_context, "_crossref_expander", None)
    monkeypatch.setattr(visual_enrichment, "_visual_proof_fn", None)
    seen = []

    def generate(query, active_query, passages):
        seen.append(copy.deepcopy(passages))
        return "offline generation spy"

    monkeypatch.setattr(generate_answer, "_generate_fn_simple", generate)
    monkeypatch.setattr(generate_answer, "_generate_fn_complex", generate)
    state = build_graph().compile().invoke({"query": case["query"]})
    return signature(state), seen


def signature(state):
    return {
        "route": state.get("route_decision"),
        "reason": state.get("route_reason"),
        "clarification": bool(state.get("clarify_message")),
        "technical_failure": state.get("technical_failure", False),
        "stages": [a["stage"] for a in state.get("retrieval_attempts", [])],
        "final_context": state.get("final_context", []),
        "rejected_origins": [c["origin"] for c in state.get("rejected_candidates", [])],
    }


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text())["cases"], ids=lambda c: c["name"]
)
def test_generic_characterization(case, monkeypatch):
    actual, seen = run_case(case, monkeypatch)
    assert actual == case["expected"]
    assert seen == ([actual["final_context"]] if actual["route"] == "generate" else [])
