"""Unit tests for eval/ls/judge.py — parsing and routing, fake LLM."""

from __future__ import annotations

import json

import pytest

from eval.ls.judge import make_correctness_evaluator, parse_verdict

pytestmark = pytest.mark.unit

FENCE = "`" * 3  # markdown-ограда


def test_parse_plain_json():
    v = parse_verdict(
        '{"correct": true, "reasoning": "ok", "forbidden_claims_asserted": []}'
    )
    assert v == {"correct": True, "reasoning": "ok", "forbidden_claims_asserted": []}


def test_parse_fenced_json():
    raw = FENCE + 'json\n{"correct": false, "reasoning": "нет п. 53"}\n' + FENCE
    v = parse_verdict(raw)
    assert v["correct"] is False and v["forbidden_claims_asserted"] == []


@pytest.mark.parametrize(
    "raw",
    [
        '{"correct": "yes", "reasoning": ""}',
        '{"reasoning": "x"}',
        "не json",
        '{"correct": true, "forbidden_claims_asserted": "строка"}',
    ],
)
def test_parse_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_verdict(raw)


class FakeLLM:
    """Минимальный Runnable-совместимый стаб: chain = prompt | llm | StrOutputParser."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, *_a, **_kw):
        from langchain_core.messages import AIMessage

        self.calls += 1
        return AIMessage(content=json.dumps(self.payload, ensure_ascii=False))


REF = {
    "ground_truth": "2 смены",
    "relevant_chunk_ids": ["2464.pdf#1"],
    "must_not_contain": "",
    "forbidden_claims": "Утверждается, что одна смена.",
    "oos_type": "",
    "legal_as_of": "2026-09-30",
}


def _by_key(results):
    return {r["key"]: r for r in results}


def test_correct_answer_scores():
    llm = FakeLLM({"correct": True, "reasoning": "ок", "forbidden_claims_asserted": []})
    ev = make_correctness_evaluator(llm)
    r = _by_key(ev({"question": "q"}, {"answer": "2 смены"}, REF))
    assert r["correctness"]["score"] == 1 and r["correctness"]["comment"] == "ок"
    assert r["forbidden_claim"]["score"] == 1


def test_forbidden_claim_asserted():
    llm = FakeLLM(
        {
            "correct": False,
            "reasoning": "x",
            "forbidden_claims_asserted": ["Утверждается, что одна смена."],
        }
    )
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": "1"}, REF)
    )
    assert r["correctness"]["score"] == 0 and r["forbidden_claim"]["score"] == 0


def test_no_forbidden_claims_is_none():
    llm = FakeLLM({"correct": True, "reasoning": "", "forbidden_claims_asserted": []})
    ref = {**REF, "forbidden_claims": ""}
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, ref)
    )
    assert r["forbidden_claim"]["score"] is None


def test_oos_skips_llm():
    llm = FakeLLM({"correct": True, "reasoning": "", "forbidden_claims_asserted": []})
    ref = {**REF, "relevant_chunk_ids": [], "oos_type": "out_of_scope"}
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, ref)
    )
    assert r["correctness"]["score"] is None and llm.calls == 0


def test_target_error_skips_llm():
    llm = FakeLLM({"correct": True, "reasoning": "", "forbidden_claims_asserted": []})
    r = _by_key(make_correctness_evaluator(llm)({"question": "q"}, {}, REF))
    assert r["correctness"]["score"] is None and r["forbidden_claim"]["score"] is None
    assert r["correctness"]["comment"] == "target_error" and llm.calls == 0


def test_empty_answer_is_wrong_without_llm():
    llm = FakeLLM({"correct": True, "reasoning": "", "forbidden_claims_asserted": []})
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": " "}, REF)
    )
    assert (
        r["correctness"]["score"] == 0 and r["correctness"]["comment"] == "empty_answer"
    )
    assert r["forbidden_claim"]["score"] == 1 and llm.calls == 0


def test_judge_format_error_scores_none():
    llm = FakeLLM({"reasoning": "без вердикта"})
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, REF)
    )
    assert r["correctness"]["score"] is None and r["forbidden_claim"]["score"] is None
    assert r["correctness"]["comment"].startswith("judge_format_error")


def test_reference_without_legal_as_of_still_judged():
    # Examples of older dataset versions (before 15.10) have no legal_as_of.
    llm = FakeLLM({"correct": True, "reasoning": "ок", "forbidden_claims_asserted": []})
    ref = {k: v for k, v in REF.items() if k != "legal_as_of"}
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, ref)
    )
    assert r["correctness"]["score"] == 1


def test_prompt_placeholders_match_variables():
    from langchain_core.prompts import ChatPromptTemplate

    from eval.ls.judge import PROMPT_PATH, CorrectnessJudgeVariables

    template = ChatPromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))
    assert set(template.input_variables) == set(CorrectnessJudgeVariables.model_fields)


# ── citations ────────────────────────────────────────────────────────────────

CONTEXT = "[1] (HIGH) [Источник: 2464]\nп. 53. Обучение проводится в 2 смены."
ANSWER = "По п. 53 Правил 2464 обучение проводится в 2 смены."


def _check(comparison, answer_quote="п. 53", context_quote="п. 53"):
    return {
        "field": "provision",
        "answer_quote": answer_quote,
        "context_quote": context_quote,
        "comparison": comparison,
        "reason": "r",
    }


def _citations(payload, outputs=None, ref=REF):
    from eval.ls.judge import make_citations_evaluator

    llm = FakeLLM(payload)
    outputs = {"answer": ANSWER, "context": CONTEXT} if outputs is None else outputs
    result = make_citations_evaluator(llm)({"question": "q"}, outputs, ref)
    assert result["key"] == "citations"
    return result, llm


def test_citations_pass_and_fail():
    r, _ = _citations({"checks": [_check("match")]})
    assert r["score"] == 1 and r["comment"].startswith("pass")
    r, _ = _citations({"checks": [_check("mismatch")]})
    assert r["score"] == 0 and r["comment"].startswith("fail")


def test_citations_needs_review_is_none():
    r, _ = _citations({"checks": [_check("unverifiable", context_quote=None)]})
    assert r["score"] is None and r["comment"].startswith("needs_review")


def test_citations_format_error_is_none():
    # The quote is not in the context: the judge invented it.
    r, _ = _citations({"checks": [_check("match", context_quote="п. 99")]})
    assert r["score"] is None and r["comment"].startswith("judge_format_error")


@pytest.mark.parametrize(
    ("outputs", "ref", "comment"),
    [
        ({}, REF, "target_error"),
        (
            {"answer": ANSWER, "context": CONTEXT},
            {**REF, "oos_type": "out_of_scope"},
            "oos",
        ),
        ({"answer": "", "context": ""}, REF, "empty_answer"),
        # Experiments before 15.10 stored no context.
        ({"answer": ANSWER}, REF, "no_context"),
    ],
)
def test_citations_skips_llm(outputs, ref, comment):
    r, llm = _citations({"checks": []}, outputs, ref)
    assert r["score"] is None and r["comment"] == comment and llm.calls == 0
