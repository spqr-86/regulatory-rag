import json

import pytest

from eval.judges.citations import judge_citations

QUESTION = "Какой срок хранения?"
RESPONSE = "Согласно разделу I пункту 71 срок хранения 5 лет."
CONTEXT = "Раздел I. Пункт 71. Срок хранения составляет 5 лет."


class FakeLLM:
    def __init__(self, content, response_metadata=None):
        self.content = content
        self.response_metadata = response_metadata or {}
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)

        class Reply:
            pass

        reply = Reply()
        reply.content = self.content
        reply.response_metadata = self.response_metadata
        return reply


def make_check(**overrides):
    check = {
        "field": "provision",
        "answer_quote": "пункту 71",
        "context_quote": "Пункт 71",
        "comparison": "match",
        "reason": "Пункт 71 совпадает во фрагменте и в ответе.",
    }
    check.update(overrides)
    return check


def run(content):
    llm = FakeLLM(content if isinstance(content, str) else json.dumps(content))
    return judge_citations(llm, QUESTION, RESPONSE, CONTEXT), llm


def test_match_gives_pass_and_status_is_added():
    verdict, llm = run({"checks": [make_check()]})
    assert verdict["status"] == "pass"
    assert len(verdict["checks"]) == 1
    assert QUESTION in llm.prompts[0] and CONTEXT in llm.prompts[0]


def test_mismatch_gives_fail():
    check = make_check(comparison="mismatch")
    verdict, _ = run({"checks": [check]})
    assert verdict["status"] == "fail"


def test_empty_checks_gives_needs_review():
    verdict, _ = run({"checks": []})
    assert verdict["status"] == "needs_review"


def test_invalid_json_raises():
    with pytest.raises(ValueError, match="invalid JSON"):
        run("не JSON")


def test_invented_quote_raises():
    with pytest.raises(ValueError, match="answer_quote must appear in response"):
        run({"checks": [make_check(answer_quote="пункту 99")]})


def test_unknown_comparison_raises_in_parser():
    with pytest.raises(ValueError, match="Unknown comparison value"):
        run({"checks": [make_check(comparison="maybe")]})


def test_truncated_reply_names_the_token_limit():
    """finish_reason=length (12.10, G022): say so instead of "invalid JSON"."""
    llm = FakeLLM("", response_metadata={"finish_reason": "length"})
    with pytest.raises(ValueError, match="CITATION_JUDGE_MAX_TOKENS"):
        judge_citations(llm, QUESTION, RESPONSE, CONTEXT)
