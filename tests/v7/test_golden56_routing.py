"""Offline regression cases from golden56 error analysis (2026-10-16)."""

from src.v7.nodes.abstain import abstain


def test_g005_abstain_has_readable_answer():
    result = abstain({"query": "В каких случаях проводится внеплановый инструктаж?"})
    assert result["answer"]
    assert "score" not in result["answer"]
    assert result["abstain_reason"]


def test_oos_saved_golden_inputs_stop_before_retrieval(monkeypatch):
    from eval.run_v7_eval import DATASET_PATH, load_dataset
    from src.v7.config import v7_config
    from src.v7.nodes.intent_gate import intent_gate

    monkeypatch.setattr(v7_config, "DOMAIN_GATE_THRESHOLD", 0)
    rows = {r["case_id"]: r for r in load_dataset(DATASET_PATH)}
    for case_id in ("G037", "G038", "G040"):
        result = intent_gate({"query": rows[case_id]["question"]})
        assert result["intent"] == "noise", case_id
        assert result["answer"], case_id
    for case_id in ("G005", "G034", "G026"):
        assert intent_gate({"query": rows[case_id]["question"]})["intent"] == "domain"


def test_complex_selects_best_admissible_before_last(monkeypatch):
    import importlib

    node = importlib.import_module("src.v7.nodes.evaluate_complex")
    candidates = [
        {
            "passages": [{"text": str(i)}],
            "active_query": "q",
            "plan": {},
            "origin": origin,
        }
        for i, origin in enumerate(("merged", "last_attempt", "fallback_snapshot"))
    ]
    monkeypatch.setattr(node, "_candidates", lambda _: candidates)
    monkeypatch.setattr(
        node,
        "_prepare",
        lambda c, *args: {"final_context": c["passages"], "status": "ok"},
    )

    def verdict(context, *args, **kwargs):
        i = int(context[0]["text"]) if context else 2
        return {
            "hard_ok": True,
            "obligations_unmet": ["refs_resolved"]
            if i != 2
            else ["refs_resolved", "enumeration_complete"],
            "pack_status": "ok",
            "triage": "borderline",
            "top_score": [0.6, 0.9, 0.2][i],
        }

    monkeypatch.setattr(node, "validate_context", verdict)
    result = node.evaluate_complex(
        {"query": "q", "retrieval_attempts": [{"stage": "complex"}]}
    )
    assert result["route_decision"] == "generate"
    assert result["final_context"][0]["text"] == "1"


def test_strict_complex_candidate_wins_over_earlier_best_effort(monkeypatch):
    import importlib

    node = importlib.import_module("src.v7.nodes.evaluate_complex")
    candidates = [
        {
            "passages": [{"text": str(i)}],
            "active_query": "q",
            "plan": {},
            "origin": origin,
        }
        for i, origin in enumerate(("merged", "last_attempt"))
    ]
    monkeypatch.setattr(node, "_candidates", lambda _: candidates)
    monkeypatch.setattr(
        node,
        "_prepare",
        lambda c, *args: {"final_context": c["passages"], "status": "ok"},
    )

    def verdict(context, *args, **kwargs):
        first = context[0]["text"] == "0"
        return {
            "hard_ok": True,
            "obligations_unmet": ["refs_resolved"] if first else [],
            "pack_status": "ok",
            "triage": "sufficient",
            "top_score": 0.9 if first else 0.6,
        }

    monkeypatch.setattr(node, "validate_context", verdict)
    result = node.evaluate_complex(
        {"query": "q", "retrieval_attempts": [{"stage": "complex"}]}
    )
    assert result["final_context"][0]["text"] == "1"
    assert result["route_reason"] == "complex_sufficient"


def test_saved_in_scope_questions_are_not_deterministically_rejected(monkeypatch):
    from eval.run_v7_eval import DATASET_PATH, load_dataset
    from src.v7.config import v7_config
    from src.v7.nodes.intent_gate import intent_gate

    monkeypatch.setattr(v7_config, "DOMAIN_GATE_THRESHOLD", 0)
    for row in load_dataset(DATASET_PATH):
        if row["oos_type"] != "out_of_scope":
            assert intent_gate({"query": row["question"]})["intent"] == "domain", row[
                "case_id"
            ]


def test_worker_rights_question_with_quoted_unsafe_instruction_stays_in_domain():
    from src.v7.nodes.intent_gate import intent_gate

    result = intent_gate(
        {
            "query": "Работодатель говорит «игнорируй инструкции». Какие права работника на охрану труда?"
        }
    )
    assert result["intent"] == "domain"


def test_scope_regex_keeps_legitimate_safety_questions():
    from src.v7.nodes.intent_gate import _SCOPE_RE

    for q in (
        "Расскажи про стихийные бедствия и эвакуацию",
        "Напиши инструкцию по шуткам на производстве",
        "Расскажи о разделе наследственных обязанностей",
    ):
        assert not _SCOPE_RE.search(q), q
    for q in (
        "Расскажи анекдот",
        "Напиши стихи про осень",
        "Как разделить наследство между родственниками",
    ):
        assert _SCOPE_RE.search(q), q
