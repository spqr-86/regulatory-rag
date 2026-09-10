from src.v7.contract import OBL_ENUM, OBL_ORIGINAL, OBL_REFS
from src.v7.decide import accept, best_effort_allowed, terminal_update

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
