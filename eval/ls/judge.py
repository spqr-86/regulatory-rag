"""LLM judges as LangSmith evaluators: binary correctness and normative citations.

Evaluators get (inputs, outputs, reference_outputs) by name, like the code evaluators.
Score 1/0/None; None = not applicable, target failed or the judge broke its format.
Provider errors are not caught: LangSmith logs them on the row.
"""

# ANCHOR: make_correctness_evaluator(llm) → evaluator returning
# [{"key": "correctness"}, {"key": "forbidden_claim"}]; prompt eval/prompts/judge_correctness.md.
# make_citations_evaluator(llm) → evaluator returning
# [{"key": "citations"}, {"key": "citations_complete"}] over
# outputs["context"] (the generator's context, see eval/ls/target.py).
# OOS, failed target and empty answer never reach the LLM.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from eval.judges.citations import (
    JudgeFormatError,
    _normalize,
    is_complete,
    judge_citations,
)
from eval.ls.evaluators import _result, _target_failed

PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "judge_correctness.md"
)

_CITATION_SCORES = {"pass": 1, "fail": 0}


class CorrectnessJudgeVariables(BaseModel):
    question: str
    legal_as_of: str
    ground_truth: str
    ground_truth_extra: str
    forbidden_claims: str
    answer: str


FACT_STATUSES = ("present", "absent", "contradicted")


_FRAGMENT_MARKER = re.compile(r"\s*\[Фрагмент[^\]]*\]")


def _quotable(text: str) -> str:
    # The judge drops "[Фрагмент N]" markers and may cut the final full stop (G022).
    return _normalize(_FRAGMENT_MARKER.sub("", text)).rstrip(".")


def _quote(value, answer: str, name: str) -> str:
    """A non-empty string found in the answer, else JudgeFormatError."""
    if not isinstance(value, str) or not value.strip():
        raise JudgeFormatError(f"{name} must be a non-empty string")
    if _quotable(value) not in _quotable(answer):
        raise JudgeFormatError(f"{name} must appear in the answer: {value!r}")
    return value


def parse_verdict(raw: str, answer: str) -> dict:
    """Judge reply → {"correct", "reasoning", "comment", "forbidden_claims_asserted"}.

    The judge lists the reference's key facts with evidence; the code checks every
    quote against the answer and computes "correct" (decision 15.10, alignment
    G001/G028: the judge called present facts missing). Accepts a ```json fence.
    Anything off-contract raises JudgeFormatError (a ValueError).
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

    facts = data.get("facts")
    if not isinstance(facts, list) or not facts:
        raise JudgeFormatError("'facts' must be a non-empty list")
    problems = []
    for item in facts:
        if not isinstance(item, dict):
            raise JudgeFormatError("each fact must be an object")
        status = item.get("status")
        if status not in FACT_STATUSES:
            raise JudgeFormatError(f"unknown fact status: {status!r}")
        if status == "absent":
            if item.get("answer_quote") is not None:
                # "Not in the answer" with a quote from the answer: self-contradiction.
                raise JudgeFormatError("absent fact must have answer_quote null")
        else:
            _quote(item.get("answer_quote"), answer, "answer_quote")
        if status != "present":
            problems.append(f"{status}: {item.get('fact', '')}")

    errors = data.get("errors", [])
    if not isinstance(errors, list):
        raise JudgeFormatError("'errors' must be a list")
    for item in errors:
        if not isinstance(item, dict):
            raise JudgeFormatError("each error must be an object")
        _quote(item.get("answer_quote"), answer, "error answer_quote")
        problems.append(f"error: {item.get('reason', '')}")

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        raise JudgeFormatError("'reasoning' must be a string")
    asserted = data.get("forbidden_claims_asserted", [])
    if not isinstance(asserted, list) or any(not isinstance(x, str) for x in asserted):
        raise JudgeFormatError("'forbidden_claims_asserted' must be a list of strings")
    problems += [f"forbidden: {claim}" for claim in asserted]

    return {
        "correct": not problems,
        "reasoning": reasoning,
        "comment": "; ".join([reasoning, *problems]) if problems else reasoning,
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
            # Examples before 16.10 have no supplementary facts.
            ground_truth_extra=reference_outputs.get("ground_truth_extra") or "",
            forbidden_claims=forbidden_claims,
            answer=answer,
        )
        reply = llm.invoke(template.format_messages(**variables.model_dump()))
        try:
            verdict = parse_verdict(_reply_text(reply), answer)
        except JudgeFormatError as exc:
            # A judge error, not an answer error: excluded from the rates.
            return _both_none(f"judge_format_error: {exc}")
        return [
            _result("correctness", int(verdict["correct"]), verdict["comment"]),
            _forbidden_claim(verdict["forbidden_claims_asserted"], forbidden_claims),
        ]

    return correctness


def make_citations_evaluator(llm) -> Callable:
    """Evaluator: do the answer's normative references match the generator's context?

    "citations" is accuracy: pass → 1, fail → 0, needs_review → None (a human looks;
    rules in eval/judges/README.md). "citations_complete": 1 if the answer names every
    document and provision the judge found for its claims, else 0.
    """

    def skipped(comment: str) -> list[dict]:
        return [
            _result("citations", None, comment),
            _result("citations_complete", None, comment),
        ]

    def citations(inputs: dict, outputs: dict, reference_outputs: dict) -> list[dict]:
        if _target_failed(outputs):
            return skipped("target_error")
        if reference_outputs.get("oos_type"):
            return skipped("oos")
        answer = (outputs.get("answer") or "").strip()
        if not answer:
            return skipped("empty_answer")
        if "context" not in outputs:
            # Experiments run before 15.10 did not store the context.
            return skipped("no_context")
        try:
            verdict = judge_citations(
                llm, inputs["question"], answer, outputs["context"]
            )
        except JudgeFormatError as exc:
            return skipped(f"judge_format_error: {exc}")

        status = verdict["status"]
        problems = [
            f"{c['field']}: {c['comparison']} — {c['reason']}"
            for c in verdict["checks"]
            if c["comparison"] != "match"
        ]
        comment = "; ".join([status, *problems])
        missing = [
            f"{c['field']}: {c['reason']}"
            for c in verdict["checks"]
            if c["comparison"] == "missing_in_answer"
        ]
        return [
            _result("citations", _CITATION_SCORES.get(status), comment),
            _result(
                "citations_complete",
                int(is_complete(verdict)),
                "; ".join(missing) or "complete",
            ),
        ]

    return citations
