"""LLM judges as LangSmith evaluators: binary correctness and normative citations.

Evaluators get (inputs, outputs, reference_outputs) by name, like the code evaluators.
Score 1/0/None; None = not applicable, target failed or the judge broke its format.
Provider errors are not caught: LangSmith logs them on the row.
"""

# ANCHOR: make_correctness_evaluator(llm) → evaluator returning
# [{"key": "correctness"}, {"key": "forbidden_claim"}]; prompt eval/prompts/judge_correctness.md.
# make_citations_evaluator(llm) → evaluator returning {"key": "citations"} over
# outputs["context"] (the generator's context, see eval/ls/target.py).
# OOS, failed target and empty answer never reach the LLM.

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from eval.judges.citations import JudgeFormatError, judge_citations
from eval.ls.evaluators import _result, _target_failed

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "judge_correctness.md"
)

_CITATION_SCORES = {"pass": 1, "fail": 0}


class CorrectnessJudgeVariables(BaseModel):
    question: str
    legal_as_of: str
    ground_truth: str
    forbidden_claims: str
    answer: str


def parse_verdict(raw: str) -> dict:
    """Judge reply → {"correct": bool, "reasoning": str, "forbidden_claims_asserted": [str]}.

    Accepts a ```json fence. Anything else off-contract raises JudgeFormatError (a ValueError).
    """
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            # Drop the opening fence line with any language tag, then the closing fence.
            raw = raw.partition("\n")[2].removesuffix("```")
        data = json.loads(raw)
    # TypeError/AttributeError: content is not a string (None, or a list of parts).
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise JudgeFormatError("Judge returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise JudgeFormatError("Expected a JSON object")

    correct = data.get("correct")
    # bool only: "yes" or 1 would be a guess about what the judge meant.
    if not isinstance(correct, bool):
        raise JudgeFormatError(f"'correct' must be true or false, got {correct!r}")
    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise JudgeFormatError("'reasoning' must be a string")
    asserted = data.get("forbidden_claims_asserted", [])
    if not isinstance(asserted, list) or any(not isinstance(x, str) for x in asserted):
        raise JudgeFormatError("'forbidden_claims_asserted' must be a list of strings")
    return {
        "correct": correct,
        "reasoning": reasoning,
        "forbidden_claims_asserted": asserted,
    }


def _reply_text(reply) -> str:
    metadata = getattr(reply, "response_metadata", None) or {}
    if metadata.get("finish_reason") == "length":
        raise JudgeFormatError("Judge reply truncated (finish_reason=length)")
    return reply.content


def _forbidden_claim(asserted: list[str], forbidden_claims: str) -> dict:
    """1 = no listed forbidden claim asserted; None when the example lists none."""
    if not forbidden_claims:
        return _result("forbidden_claim", None, "no_forbidden_claims")
    if asserted:
        return _result("forbidden_claim", 0, "; ".join(asserted))
    return _result("forbidden_claim", 1, "clean")


def _both_none(comment: str) -> list[dict]:
    return [
        _result("correctness", None, comment),
        _result("forbidden_claim", None, comment),
    ]


def make_correctness_evaluator(llm) -> Callable:
    """Evaluator: is the answer correct against the reference, and no forbidden claims?"""
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_template(PROMPT_PATH.read_text(encoding="utf-8"))

    def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> list[dict]:
        if _target_failed(outputs):
            return _both_none("target_error")
        if reference_outputs.get("oos_type"):
            # The right answer is a refusal; oos_handled scores it.
            return _both_none("oos")
        forbidden_claims = (reference_outputs.get("forbidden_claims") or "").strip()
        answer = (outputs.get("answer") or "").strip()
        if not answer:
            # Nothing said: wrong, and nothing forbidden asserted.
            return [
                _result("correctness", 0, "empty_answer"),
                _forbidden_claim([], forbidden_claims),
            ]

        variables = CorrectnessJudgeVariables(
            question=inputs["question"],
            # Examples of dataset versions before 15.10 have no legal_as_of.
            legal_as_of=reference_outputs.get("legal_as_of") or "",
            ground_truth=reference_outputs.get("ground_truth") or "",
            forbidden_claims=forbidden_claims,
            answer=answer,
        )
        reply = llm.invoke(template.format_messages(**variables.model_dump()))
        try:
            verdict = parse_verdict(_reply_text(reply))
        except JudgeFormatError as exc:
            # A judge error, not an answer error: excluded from the rates.
            return _both_none(f"judge_format_error: {exc}")
        return [
            _result("correctness", int(verdict["correct"]), verdict["reasoning"]),
            _forbidden_claim(verdict["forbidden_claims_asserted"], forbidden_claims),
        ]

    return correctness


def make_citations_evaluator(llm) -> Callable:
    """Evaluator: do the answer's normative references match the generator's context?

    pass → 1, fail → 0, needs_review → None (a human looks; rules in eval/judges/README.md).
    """

    def citations(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        if _target_failed(outputs):
            return _result("citations", None, "target_error")
        if reference_outputs.get("oos_type"):
            return _result("citations", None, "oos")
        answer = (outputs.get("answer") or "").strip()
        if not answer:
            return _result("citations", None, "empty_answer")
        if "context" not in outputs:
            # Experiments run before 15.10 did not store the context.
            return _result("citations", None, "no_context")
        try:
            verdict = judge_citations(
                llm, inputs["question"], answer, outputs["context"]
            )
        except JudgeFormatError as exc:
            return _result("citations", None, f"judge_format_error: {exc}")

        status = verdict["status"]
        problems = [
            f"{c['field']}: {c['comparison']} — {c['reason']}"
            for c in verdict["checks"]
            if c["comparison"] != "match"
        ]
        comment = "; ".join([status, *problems])
        return _result("citations", _CITATION_SCORES.get(status), comment)

    return citations
