"""Unit tests for eval/ls/judge.py — parsing and routing, fake LLM."""

from __future__ import annotations

import json

import pytest

from eval.ls.judge import make_correctness_evaluator, parse_verdict

pytestmark = pytest.mark.unit

FENCE = "`" * 3  # markdown-ограда


ANS = "Стажировка длится не менее 2 смен по п. 23 Правил."


def fact(status, quote="не менее 2 смен", text="не менее 2 смен"):
    return {"fact": text, "answer_quote": quote, "status": status}


def verdict(facts=None, errors=None, asserted=None, reasoning="ok"):
    return {
        "facts": [fact("present")] if facts is None else facts,
        "errors": errors or [],
        "reasoning": reasoning,
        "forbidden_claims_asserted": asserted or [],
    }


def test_parse_plain_json():
    v = parse_verdict(json.dumps(verdict()), ANS)
    assert v["correct"] is True and v["reasoning"] == "ok"
    assert v["forbidden_claims_asserted"] == []


def test_parse_fenced_json():
    raw = FENCE + "json\n" + json.dumps(verdict([fact("absent", None)])) + "\n" + FENCE
    assert parse_verdict(raw, ANS)["correct"] is False


# Decision 2026-10-15 (alignment, G001/G028): the code computes the verdict from
# per-fact evidence; a "present" fact must be quoted from the answer.
@pytest.mark.parametrize(
    ("payload", "correct"),
    [
        (verdict(), True),
        (verdict([fact("present"), fact("absent", None)]), False),
        (verdict([fact("contradicted")]), False),
        (verdict(errors=[{"answer_quote": "п. 23", "reason": "пункт 13"}]), False),
        (verdict(asserted=["Утверждается, что одна смена."]), False),
    ],
)
def test_verdict_computed_from_facts(payload, correct):
    assert parse_verdict(json.dumps(payload), ANS)["correct"] is correct


def test_quote_matches_ignoring_case_and_line_breaks():
    payload = verdict([fact("present", "НЕ МЕНЕЕ\n2 смен")])
    assert parse_verdict(json.dumps(payload), ANS)["correct"] is True


@pytest.mark.parametrize(
    "payload",
    [
        verdict([fact("present", "не менее 5 смен")]),  # invented quote
        verdict([fact("present", None)]),  # present without evidence
        verdict([fact("contradicted", None)]),
        verdict([fact("absent", "не менее 2 смен")]),  # absent yet quoted
        verdict([fact("unknown")]),
        verdict([]),  # no facts from the reference
        verdict(errors=[{"answer_quote": "нет такого", "reason": "x"}]),
        verdict(errors=[{"answer_quote": None, "reason": "x"}]),
        {"facts": "строка", "reasoning": "x"},
        {**verdict(), "forbidden_claims_asserted": "строка"},
        {**verdict(), "reasoning": 5},
    ],
)
def test_parse_rejects_off_contract(payload):
    with pytest.raises(ValueError):
        parse_verdict(json.dumps(payload), ANS)


# G022 (rejudge 15.10): the judge quotes a sentence without the answer's
# "[Фрагмент N]" marker that sits before the final full stop.
def test_quote_ignores_fragment_markers():
    answer = (
        "Проводит работодатель по договору [Фрагмент 9]. Далее [Фрагмент 2, 2464.md]."
    )
    payload = verdict([fact("present", "Проводит работодатель по договору.")])
    assert parse_verdict(json.dumps(payload), answer)["correct"] is True


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("не json", ANS)


def test_comment_names_missing_facts():
    payload = verdict([fact("present"), fact("absent", None, "после срока СОУТ")])
    v = parse_verdict(json.dumps(payload), ANS)
    assert "absent: после срока СОУТ" in v["comment"]


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
    llm = FakeLLM(verdict(reasoning="ок"))
    ev = make_correctness_evaluator(llm)
    r = _by_key(ev({"question": "q"}, {"answer": ANS}, REF))
    assert r["correctness"]["score"] == 1 and r["correctness"]["comment"] == "ок"
    assert r["forbidden_claim"]["score"] == 1


def test_forbidden_claim_asserted():
    llm = FakeLLM(verdict(asserted=["Утверждается, что одна смена."]))
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": ANS}, REF)
    )
    assert r["correctness"]["score"] == 0 and r["forbidden_claim"]["score"] == 0


def test_no_forbidden_claims_is_none():
    llm = FakeLLM(verdict())
    ref = {**REF, "forbidden_claims": ""}
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": ANS}, ref)
    )
    assert r["forbidden_claim"]["score"] is None


def test_oos_skips_llm():
    llm = FakeLLM(verdict())
    ref = {**REF, "relevant_chunk_ids": [], "oos_type": "out_of_scope"}
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": "a"}, ref)
    )
    assert r["correctness"]["score"] is None and llm.calls == 0


def test_target_error_skips_llm():
    llm = FakeLLM(verdict())
    r = _by_key(make_correctness_evaluator(llm)({"question": "q"}, {}, REF))
    assert r["correctness"]["score"] is None and r["forbidden_claim"]["score"] is None
    assert r["correctness"]["comment"] == "target_error" and llm.calls == 0


def test_empty_answer_is_wrong_without_llm():
    llm = FakeLLM(verdict())
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
    llm = FakeLLM(verdict())
    ref = {k: v for k, v in REF.items() if k != "legal_as_of"}
    r = _by_key(
        make_correctness_evaluator(llm)({"question": "q"}, {"answer": ANS}, ref)
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
    results = make_citations_evaluator(llm)({"question": "q"}, outputs, ref)
    assert [r["key"] for r in results] == ["citations", "citations_complete"]
    return results[0], llm


def test_citations_complete_is_a_separate_score():
    from eval.ls.judge import make_citations_evaluator

    missing = _check("missing_in_answer", answer_quote=None)
    llm = FakeLLM({"checks": [_check("match"), missing]})
    accuracy, complete = make_citations_evaluator(llm)(
        {"question": "q"}, {"answer": ANSWER, "context": CONTEXT}, REF
    )
    assert accuracy["score"] == 1
    assert complete["score"] == 0


def test_citations_complete_is_none_when_judge_skipped():
    from eval.ls.judge import make_citations_evaluator

    llm = FakeLLM({"checks": []})
    results = make_citations_evaluator(llm)({"question": "q"}, {}, REF)
    assert [r["score"] for r in results] == [None, None]


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
