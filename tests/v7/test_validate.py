from src.v7.contract import OBL_ENUM, OBL_ORIGINAL, OBL_REFS
from src.v7.validate import required_obligations, validate_context

PLAN = {
    "threshold": 0.1,
    "min_passages": 1,
    "min_keyword_overlap": 0.0,
    "borderline_threshold": 0.05,
    "max_single_doc_ratio": 1.0,
}


def _p(chunk_id, text, source="doc.pdf", score=0.9):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "vector_score": score,
        "metadata": {"source": source},
    }


def test_refs_and_original_are_always_required():
    obl = required_obligations("любой запрос")
    assert OBL_REFS in obl and OBL_ORIGINAL in obl


def test_enumeration_obligation_only_for_enumeration_queries():
    assert OBL_ENUM in required_obligations("кто проходит медосмотр")
    assert OBL_ENUM not in required_obligations("что такое медосмотр")


def test_open_ref_keeps_refs_obligation_unmet():
    ctx = [_p(1, "медосмотр проводится согласно пункт 15 настоящего порядка")]
    v = validate_context(ctx, "медосмотр", "медосмотр", PLAN, {OBL_REFS})
    assert OBL_REFS in v["obligations_unmet"]


def test_resolved_ref_clears_obligation():
    ctx = [
        _p(1, "медосмотр проводится согласно пункт 15 настоящего порядка"),
        _p(2, "15. Медосмотр проводится ежегодно."),
    ]
    v = validate_context(ctx, "медосмотр", "медосмотр", PLAN, {OBL_REFS})
    assert OBL_REFS not in v["obligations_unmet"]


def test_degraded_pack_blocks_refs_obligation():
    ctx = [_p(1, "текст без ссылок вообще")]
    v = validate_context(
        ctx, "медосмотр", "медосмотр", PLAN, {OBL_REFS}, pack_status="degraded"
    )
    assert OBL_REFS in v["obligations_unmet"]
    assert v["pack_status"] == "degraded"


def test_zero_original_overlap_keeps_original_obligation():
    ctx = [_p(1, "совершенно посторонний текст")]
    v = validate_context(ctx, "медосмотр водителей", "прочее", PLAN, {OBL_ORIGINAL})
    assert OBL_ORIGINAL in v["obligations_unmet"]


def test_positive_original_overlap_below_plan_floor_keeps_obligation(monkeypatch):
    plan = {**PLAN, "min_keyword_overlap_original": 0.10}
    monkeypatch.setattr(
        "src.v7.validate.check_full_triage",
        lambda *args, **kwargs: {
            "sufficient": True,
            "triage": "sufficient",
            "top_score": 0.9,
            "keyword_overlap_active": 0.5,
            "keyword_overlap_original": 0.05,
        },
    )

    verdict = validate_context(
        [_p(1, "медосмотр")],
        "медосмотр водителей",
        "медосмотр",
        plan,
        {OBL_ORIGINAL},
    )

    assert OBL_ORIGINAL in verdict["obligations_unmet"]


def test_enumeration_cleared_only_when_not_subset_of_prior():
    prior = [_p(1, "а) водители"), _p(2, "б) машинисты")]
    same = [_p(1, "а) водители"), _p(2, "б) машинисты")]
    wider = same + [_p(3, "в) крановщики")]
    q = "кто проходит медосмотр"
    assert (
        OBL_ENUM
        in validate_context(same, q, q, PLAN, {OBL_ENUM}, prior_context=prior)[
            "obligations_unmet"
        ]
    )
    assert (
        OBL_ENUM
        not in validate_context(wider, q, q, PLAN, {OBL_ENUM}, prior_context=prior)[
            "obligations_unmet"
        ]
    )


def test_empty_prior_is_not_the_same_as_no_prior():
    """[] — реальный пустой доэскалационный контекст; None — его отсутствие."""
    q = "кто проходит медосмотр"
    cand = [_p(1, "а) водители"), _p(2, "б) машинисты")]
    with_empty = validate_context(cand, q, q, PLAN, {OBL_ENUM}, prior_context=[])
    without = validate_context(cand, q, q, PLAN, {OBL_ENUM}, prior_context=None)
    assert OBL_ENUM not in with_empty["obligations_unmet"]
    assert OBL_ENUM in without["obligations_unmet"]


def test_empty_context_keeps_every_obligation():
    q = "кто проходит медосмотр"
    v = validate_context([], q, q, PLAN, required_obligations(q))
    assert set(v["obligations_unmet"]) == set(required_obligations(q))
    assert v["hard_ok"] is False


def test_validate_does_not_mutate_context():
    import copy

    ctx = [_p(1, "текст")]
    snapshot = copy.deepcopy(ctx)
    validate_context(ctx, "q", "q", PLAN, {OBL_REFS})
    assert ctx == snapshot


def test_validate_module_does_not_import_nodes():
    import src.v7.validate as v

    assert "src.v7.nodes" not in open(v.__file__, encoding="utf-8").read()
