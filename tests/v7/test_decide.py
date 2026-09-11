from src.v7.contract import OBL_ENUM, OBL_ORIGINAL, OBL_REFS
from src.v7.decide import accept, best_effort_allowed, decide_simple, terminal_update

CTX = [{"chunk_id": 1, "text": "t", "metadata": {"source": "d"}}]


def _v(unmet, hard_ok=True, triage="sufficient", active=0.5, original=0.5):
    return {
        "triage": triage,
        "hard_ok": hard_ok,
        "details": {
            "keyword_overlap_active": active,
            "keyword_overlap_original": original,
            "triage": triage,
        },
        "gap": {"kind": "unresolved_ref", "refs": [], "closed": [], "open": []},
        "obligations_unmet": list(unmet),
        "top_score": 0.7,
        "pack_status": "ok",
    }


def test_accept_when_everything_clear():
    assert accept(CTX, _v([])) is True


def test_reject_empty_context():
    assert accept([], _v([])) is False


def test_reject_when_hard_gates_fail():
    assert accept(CTX, _v([], hard_ok=False, triage="borderline")) is False


def test_reject_when_refs_unmet():
    assert accept(CTX, _v([OBL_REFS])) is False


def test_refs_best_effort_allowed_only_on_last_complex_candidate():
    verdict = _v([OBL_REFS])
    assert accept(CTX, verdict, on_complex=True, is_last_candidate=False) is False
    assert accept(CTX, verdict, on_complex=True, is_last_candidate=True) is True


def test_refs_best_effort_rejects_degraded_pack():
    verdict = _v([OBL_REFS])
    verdict["pack_status"] = "degraded"

    assert accept(CTX, verdict, on_complex=True, is_last_candidate=True) is False


def test_enumeration_alone_blocks_on_simple():
    assert (
        accept(CTX, _v([OBL_ENUM]), on_complex=False, is_last_candidate=True) is False
    )


def test_enumeration_allowed_on_last_complex_candidate():
    assert accept(CTX, _v([OBL_ENUM]), on_complex=True, is_last_candidate=True) is True


def test_enumeration_not_allowed_on_non_last_candidate():
    assert (
        accept(CTX, _v([OBL_ENUM]), on_complex=True, is_last_candidate=False) is False
    )


def test_best_effort_requires_all_other_obligations_met():
    v = _v([OBL_ENUM, OBL_REFS])
    assert best_effort_allowed(v, on_complex=True, is_last_candidate=True) is False
    assert accept(CTX, v, on_complex=True, is_last_candidate=True) is False


def test_terminal_update_writes_every_key_and_clears_stale_state():
    stale = {
        "route_decision": "generate",
        "route_reason": "context_sufficient",
        "obligations_unmet": [],
        "obligations_required": ["refs_resolved"],
        "final_context": [{"text": "старое"}],
        "candidate_context": [{"text": "старое"}],
        "final_passages": [{"text": "старое"}],
        "final_score": 0.9,
        "sufficient": True,
        "technical_failure": False,
        "triage_gap": {
            "kind": "unresolved_ref",
            "refs": [],
            "closed": [],
            "open": ["clause:1"],
        },
        "sufficiency_details": {"triage": "sufficient"},
        "fallback_snapshot": {"passages": [{"text": "старое"}]},
        "rejected_candidates": [{"origin": "merged"}],
    }
    update = terminal_update(
        route="abstain",
        reason="empty_pool",
        final_context=[],
        candidate_context=[],
        verdict=_v(
            [OBL_REFS, OBL_ORIGINAL],
            hard_ok=False,
            triage="clearly_bad",
            original=0.0,
        ),
        required={OBL_REFS, OBL_ORIGINAL},
        technical_failure=False,
    )
    merged = {**stale, **update}
    assert set(stale).issubset(set(update)), "update обязан переписать каждый ключ"
    assert merged["route_decision"] == "abstain"
    assert merged["final_context"] == []
    assert merged["final_passages"] == []
    assert merged["fallback_snapshot"] == {}
    assert merged["rejected_candidates"] == []
    assert merged["sufficient"] is False
    assert merged["triage_gap"]["open"] == []
    assert sorted(merged["obligations_required"]) == [OBL_ORIGINAL, OBL_REFS]


def test_terminal_update_sufficient_is_derived():
    update = terminal_update(
        route="generate",
        reason="context_sufficient",
        final_context=CTX,
        candidate_context=CTX,
        verdict=_v([]),
        required={OBL_REFS},
        technical_failure=False,
    )
    assert update["sufficient"] is True
    assert update["final_passages"] == CTX


def _d(
    ctx,
    unmet,
    hard_ok=True,
    triage="sufficient",
    err=False,
    abstain=True,
    active=0.5,
    original=0.5,
):
    return decide_simple(
        ctx,
        _v(
            unmet,
            hard_ok=hard_ok,
            triage=triage,
            active=active,
            original=original,
        ),
        retrieval_error=err,
        abstain_on_empty=abstain,
    )


def test_row1_retrieval_error_wins_over_everything():
    assert _d(CTX, [], err=True) == ("abstain", "retrieval_error")


def test_row2_empty_pool():
    assert _d([], []) == ("abstain", "empty_pool")


def test_row2_empty_pool_escalates_when_policy_off():
    assert _d([], [], abstain=False) == ("complex", "empty_pool_escalated")


def test_row3_zero_overlap_both():
    assert _d(
        CTX,
        [OBL_ORIGINAL],
        hard_ok=False,
        triage="clearly_bad",
        active=0.0,
        original=0.0,
    ) == ("abstain", "zero_overlap_both")


def test_row3_escalates_when_policy_off():
    assert _d(
        CTX,
        [OBL_ORIGINAL],
        hard_ok=False,
        triage="clearly_bad",
        active=0.0,
        original=0.0,
        abstain=False,
    ) == ("complex", "zero_overlap_both_escalated")


def test_row4_enumeration_intent_escalates():
    assert _d(CTX, [OBL_ENUM]) == ("complex", "enumeration_intent")


def test_row5_refs_unresolved_escalates():
    assert _d(CTX, [OBL_REFS]) == ("complex", "refs_unresolved")


def test_row5_covers_degraded_pack_without_calling_it_a_retrieval_error():
    v = _v([OBL_REFS])
    v["pack_status"] = "degraded"
    assert decide_simple(
        CTX,
        v,
        retrieval_error=False,
        abstain_on_empty=True,
    ) == ("complex", "refs_unresolved")


def test_row4_wins_over_row5():
    assert _d(CTX, [OBL_ENUM, OBL_REFS]) == ("complex", "enumeration_intent")


def test_row6_zero_overlap_original():
    assert _d(CTX, [OBL_ORIGINAL], active=0.4, original=0.0) == (
        "complex",
        "zero_overlap_original",
    )


def test_row7_borderline():
    assert _d(CTX, [], hard_ok=False, triage="borderline") == (
        "complex",
        "triage_borderline",
    )


def test_row8_clearly_bad():
    assert _d(
        CTX,
        [],
        hard_ok=False,
        triage="clearly_bad",
        active=0.3,
        original=0.3,
    ) == ("complex", "triage_clearly_bad")


def test_row9_generate():
    assert _d(CTX, []) == ("generate", "context_sufficient")


def test_every_reason_is_a_declared_simple_code():
    from src.v7.contract import SIMPLE_REASONS

    cases = [
        _d(CTX, [], err=True),
        _d([], []),
        _d([], [], abstain=False),
        _d(CTX, [OBL_ENUM]),
        _d(CTX, [OBL_REFS]),
        _d(CTX, [OBL_ORIGINAL], active=0.4, original=0.0),
        _d(CTX, [], hard_ok=False, triage="borderline"),
        _d(CTX, []),
    ]
    for route, reason in cases:
        assert reason in SIMPLE_REASONS, reason
