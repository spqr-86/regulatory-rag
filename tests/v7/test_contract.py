from src.v7.contract import (
    ALL_OBLIGATIONS,
    COMPLEX_REASONS,
    OBL_ENUM,
    OBL_ORIGINAL,
    OBL_REFS,
    SIMPLE_REASONS,
)
from src.v7.state_types import RAGState


def test_obligation_names_are_exactly_three():
    assert ALL_OBLIGATIONS == {OBL_REFS, OBL_ORIGINAL, OBL_ENUM}
    assert OBL_REFS == "refs_resolved"
    assert OBL_ORIGINAL == "original_query_relevant"
    assert OBL_ENUM == "enumeration_complete"


def test_reason_codes_match_the_spec():
    assert SIMPLE_REASONS == {
        "retrieval_error",
        "empty_pool",
        "zero_overlap_both",
        "empty_pool_escalated",
        "zero_overlap_both_escalated",
        "enumeration_intent",
        "refs_unresolved",
        "zero_overlap_original",
        "triage_borderline",
        "triage_clearly_bad",
        "context_sufficient",
    }
    assert COMPLEX_REASONS == {
        "complex_sufficient",
        "complex_fallback_accepted",
        "enumeration_best_effort",
        "complex_exhausted",
    }
    assert "retrieval_error" not in COMPLEX_REASONS


def test_state_carries_terminal_contract_keys():
    keys = RAGState.__annotations__
    for key in (
        "route_decision",
        "route_reason",
        "obligations_unmet",
        "obligations_required",
        "final_context",
        "candidate_context",
        "technical_failure",
        "fallback_snapshot",
        "rejected_candidates",
    ):
        assert key in keys, key
