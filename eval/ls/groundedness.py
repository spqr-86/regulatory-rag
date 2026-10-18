"""Evidence-checked groundedness evaluator over the generator's exact context."""

# ANCHOR: answer/context → validated claim checks → 1/0 (None: no checks); no reference.

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from eval.judges.citations import JudgeFormatError, _normalize
from eval.ls.evaluators import _result, _target_failed
from eval.ls.judge import _reply_text
from eval.ls.policy import is_oos

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts/judge_groundedness.md"
# One check per claim with verbatim quotes: a long list answer overflowed 16000.
MAX_TOKENS = 32000


class Variables(BaseModel):
    question: str
    answer: str
    context: str


class Check(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_quote: str = Field(min_length=1)
    context_quote: str | None
    status: Literal["supported", "unsupported", "unverifiable"]
    reason: str = Field(min_length=1)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    checks: list[Check]


def _quote_text(text: str) -> str:
    # The judge quotes answers without Markdown bold ("**"); "__" stays, it is a
    # blank-to-fill in normative forms.
    return _normalize(text.replace("**", ""))


def parse_groundedness(raw: str, answer: str, context: str) -> dict:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.partition("\n")[2].removesuffix("```")
        verdict = Verdict.model_validate(json.loads(raw))
    except (ValueError, TypeError, AttributeError, ValidationError) as exc:
        raise JudgeFormatError("Invalid groundedness JSON contract") from exc
    for check in verdict.checks:
        if not check.answer_quote.strip() or _quote_text(
            check.answer_quote
        ) not in _quote_text(answer):
            raise JudgeFormatError("answer_quote must appear in answer")
        if check.context_quote is not None:
            if not check.context_quote.strip() or _normalize(
                check.context_quote
            ) not in _normalize(context):
                raise JudgeFormatError("context_quote must appear in generator context")
        if check.status == "supported" and not check.context_quote:
            raise JudgeFormatError("supported requires context_quote")
    statuses = {check.status for check in verdict.checks}
    # unverifiable = the context gives no support, so the answer is not grounded
    # in it (Petr, 2026-10-18); no checks at all stays unscored.
    score = (1 if statuses == {"supported"} else 0) if statuses else None
    return {"score": score, "checks": [check.model_dump() for check in verdict.checks]}


def make_groundedness_evaluator(llm, keep_raw: bool = False):
    """keep_raw adds the judge reply text as "raw" for offline debugging."""
    from langchain_core.prompts import ChatPromptTemplate

    template = ChatPromptTemplate.from_template(
        importlib.resources.files("eval")
        .joinpath("prompts/judge_groundedness.md")
        .read_text(encoding="utf-8")
    )

    def groundedness(inputs, outputs, reference_outputs):
        if _target_failed(outputs):
            return _result("groundedness", None, "target_error")
        if is_oos(reference_outputs):
            return _result("groundedness", None, "oos")
        if not (outputs.get("answer") or "").strip():
            return _result("groundedness", None, "empty_answer")
        if not (outputs.get("context") or "").strip():
            return _result("groundedness", None, "no_context")
        variables = Variables(
            question=inputs["question"],
            answer=outputs["answer"],
            context=outputs["context"],
        )
        reply = llm.invoke(template.format_messages(**variables.model_dump()))
        extra = {"raw": getattr(reply, "content", None)} if keep_raw else {}
        try:
            verdict = parse_groundedness(
                _reply_text(reply), variables.answer, variables.context
            )
        except JudgeFormatError as exc:
            return _result("groundedness", None, f"judge_format_error: {exc}") | extra
        problems = [
            f"{c['status']}: {c['answer_quote']} — {c['reason']}"
            for c in verdict["checks"]
            if c["status"] != "supported"
        ]
        return (
            _result(
                "groundedness",
                verdict["score"],
                "; ".join(problems)
                or ("supported" if verdict["checks"] else "needs_review: no_checks"),
            )
            | extra
        )

    return groundedness
