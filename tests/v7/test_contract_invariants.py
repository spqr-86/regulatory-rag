"""Инварианты контракта решения (спек 2026-09-09, раздел «Тестирование»).

Проверяются на сквозном пути. При расхождении чинится реализация — но если
тест противоречит спеку, чинится тест: формулировка спека главнее.
"""

# ANCHOR: End-to-end executable invariants for the terminal triage contract.
# Inputs traverse evaluator, packer, validator, and real prompt assembly; outputs
# assert context identity, obligation safety, and complete terminal state writes.

import copy

import pytest

from src.v7 import bridge, pack_context as pc
from src.v7.contract import OBL_ENUM, OBL_REFS
from src.v7.decide import accept
from src.v7.nodes.evaluate_complex import evaluate_complex
from src.v7.nodes.evaluate_triage import evaluate_triage
from src.v7.nodes.generate_answer import generate_answer, set_generate_fn
from src.v7.pack_context import candidate_version, pack_context
from src.v7.validate import required_obligations, validate_context

PLAN = {
    "threshold": 0.4,
    "min_passages": 2,
    "min_keyword_overlap": 0.1,
    "borderline_threshold": 0.3,
    "max_single_doc_ratio": 1.0,
}


def _p(i, text, score=0.8, source="doc.pdf"):
    return {
        "chunk_id": i,
        "text": text,
        "vector_score": score,
        "metadata": {"source": source},
    }


GOOD = [
    _p(1, "медосмотр водителей проводится ежегодно"),
    _p(2, "медосмотр водителей обязателен"),
]


@pytest.fixture(autouse=True)
def _no_expander():
    pc.set_crossref_expander(None)
    yield
    pc.set_crossref_expander(None)


def _simple_state(passages, query="медосмотр водителей"):
    return {
        "query": query,
        "active_query": query,
        "plan": PLAN,
        "retrieval_attempts": [
            {
                "stage": "simple",
                "passages": passages,
                "attempt_plan": PLAN,
                "retrieval_error": False,
            }
        ],
    }


def test_invariant_1_prompt_carries_exactly_the_validated_context():
    """Сквозь настоящий сборщик промпта: что валидировано, то и в промпте."""
    seen = {}

    class _FakeLLM:
        def invoke(self, messages):
            seen["prompt"] = messages[0].content

            class R:
                content = "ответ"
                response_metadata = {}
                usage_metadata = {}

            return R()

    out = evaluate_triage(_simple_state(GOOD))
    assert out["route_decision"] == "generate"

    set_generate_fn(bridge.make_generate_fn(_FakeLLM(), backend=None))
    try:
        generate_answer({**_simple_state(GOOD), **out})
    finally:
        set_generate_fn(None)

    for p in out["final_context"]:
        assert p["text"] in seen["prompt"]
    # ничего сверх проверенного в промпт не попало
    assert seen["prompt"].count("[Источник:") == len(out["final_context"])


def test_invariant_2_expander_runs_once_across_simple_and_complex():
    """Полный жизненный цикл: simple упаковал → эскалация → complex взял снимок."""
    calls = []

    def _expander(ps, q):
        calls.append(q)
        return list(ps)

    pc.set_crossref_expander(_expander)
    q = "кто проходит медосмотр"
    simple_ps = [
        _p(1, "кто проходит медосмотр: а) водители"),
        _p(2, "б) машинисты медосмотр"),
    ]
    simple_out = evaluate_triage(_simple_state(simple_ps, q))
    assert simple_out["route_decision"] == "complex"
    calls_after_simple = len(calls)

    complex_state = {
        **_simple_state(simple_ps, q),
        "fallback_snapshot": simple_out["fallback_snapshot"],
    }
    complex_state["retrieval_attempts"] = [
        complex_state["retrieval_attempts"][0],
        {
            "stage": "complex",
            "passages": simple_ps,
            "attempt_plan": PLAN,
            "retrieval_error": False,
        },
    ]
    evaluate_complex(complex_state)
    # снимок уже упакован: повторного расширения того же кандидата нет
    assert len(calls) <= calls_after_simple + 1, calls


def test_invariant_2b_version_is_order_independent():
    ps = [_p(1, "a"), _p(2, "b")]
    assert candidate_version(ps, PLAN, "q") == candidate_version(
        list(reversed(ps)), PLAN, "q"
    )


def test_simple_evidence_survives_escalation_to_complex():
    """A direct norm at simple rank 12 must still reach the generator."""
    q = "кто проходит медосмотр"
    direct = _p(12, "медосмотр проходят водители и машинисты")
    simple_ps = [_p(i, f"медосмотр работников правило {i}") for i in range(1, 12)] + [
        direct
    ]

    simple_state = _simple_state(simple_ps, q)
    simple_out = evaluate_triage(simple_state)
    assert simple_out["route_decision"] == "complex"

    complex_state = {**simple_state, **simple_out}
    complex_state["retrieval_attempts"] = [
        simple_state["retrieval_attempts"][0],
        {
            "stage": "complex",
            "passages": [],
            "attempt_plan": PLAN,
            "retrieval_error": False,
        },
    ]
    out = evaluate_complex(complex_state)

    assert out["route_decision"] == "generate"
    assert any(p["chunk_id"] == direct["chunk_id"] for p in out["final_context"])


def test_invariant_3_obligation_cleared_only_on_packed_context(monkeypatch):
    """Ссылку закрывает чанк, срезанный ТОКЕННЫМ БЮДЖЕТОМ — обязательство стоит."""
    monkeypatch.setattr(pc.v7_config, "MAX_CHUNKS_FOR_LLM", 10)
    first = _p(1, "медосмотр проводится согласно пункт 15")
    second = _p(2, "15. Медосмотр проводится ежегодно.")
    budget = pc.passage_cost(first)  # хватает ровно на первый
    monkeypatch.setattr(pc.v7_config, "PACK_TOKEN_BUDGET", budget)
    packed = pack_context([first, second], "медосмотр", PLAN)
    assert len(packed["final_context"]) == 1
    v = validate_context(
        packed["final_context"],
        "медосмотр",
        "медосмотр",
        PLAN,
        {OBL_REFS},
    )
    assert OBL_REFS in v["obligations_unmet"]


def test_invariant_4_subset_candidate_does_not_clear_enumeration():
    q = "кто проходит медосмотр"
    prior = [_p(1, "а) водители"), _p(2, "б) машинисты")]
    v = validate_context(list(prior), q, q, PLAN, {OBL_ENUM}, prior_context=prior)
    assert OBL_ENUM in v["obligations_unmet"]


def test_invariant_5_best_effort_blocked_by_other_obligations():
    verdict = {
        "triage": "sufficient",
        "hard_ok": True,
        "details": {"keyword_overlap_active": 0.5, "keyword_overlap_original": 0.5},
        "gap": {"kind": "unresolved_ref", "refs": [], "closed": [], "open": []},
        "obligations_unmet": [OBL_ENUM, OBL_REFS],
        "top_score": 0.8,
        "pack_status": "ok",
    }
    assert accept(GOOD, verdict, on_complex=True, is_last_candidate=True) is False


@pytest.mark.parametrize(
    "node,state",
    [
        ("triage", "empty"),
        ("triage", "good"),
        ("triage", "enumeration"),
        ("complex", "empty"),
        ("complex", "good"),
    ],
)
def test_invariant_6_terminal_branches_leave_no_stale_state(node, state):
    """Каждая терминальная ветка обоих узлов переписывает контракт целиком."""
    stale = {
        "route_decision": "generate",
        "route_reason": "context_sufficient",
        "obligations_unmet": ["stale"],
        "obligations_required": ["stale"],
        "final_context": [_p(99, "остаток")],
        "candidate_context": [_p(99, "остаток")],
        "final_passages": [_p(99, "остаток")],
        "final_score": 0.99,
        "sufficient": True,
        "technical_failure": False,
        "triage_gap": {
            "kind": "unresolved_ref",
            "refs": [],
            "closed": [],
            "open": ["clause:99"],
        },
        "sufficiency_details": {"triage": "sufficient", "passage_count": 99},
        "fallback_snapshot": {
            "passages": [_p(99, "остаток")],
            "plan": PLAN,
            "active_query": "старый запрос",
            "origin": "fallback_snapshot",
            "packed": True,
            "pack_status": "ok",
        },
        "rejected_candidates": [{"origin": "stale"}],
    }
    q = "кто проходит медосмотр" if state == "enumeration" else "медосмотр водителей"
    passages = {
        "empty": [],
        "good": GOOD,
        "enumeration": [
            _p(1, "кто проходит медосмотр: а) водители"),
            _p(2, "б) машинисты медосмотр"),
        ],
    }[state]
    base = {**stale, **_simple_state(passages, q)}
    if node == "triage":
        out = evaluate_triage(base)
    else:
        base["retrieval_attempts"].append(
            {
                "stage": "complex",
                "passages": passages,
                "attempt_plan": PLAN,
                "retrieval_error": False,
            }
        )
        out = evaluate_complex(base)

    merged = {**stale, **out}
    for key in stale:
        assert key in out, f"{key} не переписан — в состоянии останется старое"
    assert merged["sufficient"] == (merged["route_decision"] == "generate")
    assert merged["final_passages"] == (
        merged["final_context"] if merged["route_decision"] == "generate" else []
    )
    assert "stale" not in merged["obligations_unmet"]
    assert merged["triage_gap"]["open"] != ["clause:99"]
    if merged["route_decision"] != "complex":
        assert merged["fallback_snapshot"] == {}


def test_invariant_7_degraded_pack_does_not_clear_refs():
    def _boom(ps, q):
        raise RuntimeError("expander down")

    pc.set_crossref_expander(_boom)
    packed = pack_context(GOOD, "медосмотр водителей", PLAN)
    assert packed["status"] == "degraded"
    v = validate_context(
        packed["final_context"],
        "медосмотр водителей",
        "медосмотр водителей",
        PLAN,
        required_obligations("медосмотр водителей"),
        pack_status=packed["status"],
    )
    assert OBL_REFS in v["obligations_unmet"]


def test_invariant_8_technical_failure_is_not_empty_pool():
    failed = _simple_state([])
    failed["retrieval_attempts"][0]["retrieval_error"] = True
    out = evaluate_triage(failed)
    assert out["route_reason"] == "retrieval_error"
    assert out["technical_failure"] is True

    empty = evaluate_triage(_simple_state([]))
    assert empty["route_reason"] == "empty_pool"
    assert empty["technical_failure"] is False


def test_pack_input_is_never_mutated():
    """Отдельно от инвариантов: чистота упаковки на глубоком сравнении."""

    def _mutating(ps, q):
        ps[0]["metadata"]["source"] = "ИСПОРЧЕНО"
        return list(ps)

    pc.set_crossref_expander(_mutating)
    original = copy.deepcopy(GOOD)
    pack_context(GOOD, "медосмотр водителей", PLAN)
    assert GOOD == original
