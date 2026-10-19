"""Evidence-checked groundedness evaluator over the generator's exact context."""

# ANCHOR: answer/context → validated claim checks → 1/0 (None: no checks); no reference.
# Quotes match fuzzily: QUOTE_ERROR_RATE edits per "..." piece, numbers and
# negations exact.

from __future__ import annotations

import importlib.resources
import json
import re
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
# deepseek-v4-pro drops letters copying Cyrillic (run3: "созаются", "результы"
# = 2 edits in 58 chars); 0.05 made 17/19 run3 replies valid, 0.03 — 16.
QUOTE_ERROR_RATE = 0.05

_FRAGMENT_MARKER = re.compile(r"\[Фрагмент[^\]]*\]")
_ELLIPSIS = re.compile(r"\.{3,}|…")
# The judge writes the Ukrainian "і" for "и" (run3 GR12: 134 times in one reply).
_HOMOGLYPHS = str.maketrans({"і": "и", "ё": "е"})
# One edit away yet a different claim: such tokens must match exactly.
_EXACT_TOKENS = re.compile(r"\d+|(?<!\w)(?:не|ни|нет|без)(?!\w)")


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
    # The judge quotes answers without Markdown bold ("**") and inline fragment
    # markers; "__" stays, it is a blank-to-fill in normative forms.
    text = _FRAGMENT_MARKER.sub("", text.replace("**", ""))
    return _normalize(text).translate(_HOMOGLYPHS)


def _closest_end(pattern: str, text: str) -> tuple[int, int]:
    """Fewest edits turning pattern into some substring of text, and its end.

    Myers' bit-parallel approximate matching, O(len(text)) big-int steps.
    """
    masks: dict[str, int] = {}
    for i, char in enumerate(pattern):
        masks[char] = masks.get(char, 0) | 1 << i
    full, top = (1 << len(pattern)) - 1, 1 << (len(pattern) - 1)
    pv, mv, distance = full, 0, len(pattern)
    best = (distance, 0)
    for j, char in enumerate(text):
        eq = masks.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & full)
        mh = pv & xh
        if ph & top:
            distance += 1
        elif mh & top:
            distance -= 1
        ph = (ph << 1) & full
        mh = (mh << 1) & full
        pv = mh | (~(xv | ph) & full)
        mv = ph & xv
        if distance < best[0]:
            best = (distance, j + 1)
    return best


def _piece_end(piece: str, text: str) -> int | None:
    """End of piece's match in text, or None when it is not there."""
    if piece in text:
        return text.index(piece) + len(piece)
    budget = int(len(piece) * QUOTE_ERROR_RATE)
    distance, end = _closest_end(piece, text)
    if distance > budget:
        return None
    # The window may overhang the match by the budget, so the quote's exact
    # tokens must be a run inside the window's.
    window = _EXACT_TOKENS.findall(text[max(0, end - len(piece) - budget) : end + 1])
    tokens = _EXACT_TOKENS.findall(piece)
    if any(
        window[i : i + len(tokens)] == tokens
        for i in range(len(window) - len(tokens) + 1)
    ):
        return end
    return None


def _quote_found(quote: str, text: str) -> bool:
    """Each "..."-separated piece of quote appears in text, in order."""
    text = _quote_text(text)
    pieces = [p for p in map(_quote_text, _ELLIPSIS.split(quote)) if p]
    for piece in pieces:
        end = _piece_end(piece, text)
        if end is None:
            return False
        text = text[end:]
    return bool(pieces)


def _within_one_edit(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[i + 1 :] == b[i + 1 :] or a[i + 1 :] == b[i:] or a[i:] == b[i + 1 :]


def _repair_keys(data):
    """Map one-edit typos of Check field names back (run3 GR09: "st tus")."""
    if not isinstance(data, dict) or not isinstance(data.get("checks"), list):
        return data
    fields = set(Check.model_fields)
    for check in data["checks"]:
        if not isinstance(check, dict):
            continue
        for key in [k for k in check if k not in fields]:
            known = [f for f in fields - check.keys() if _within_one_edit(key, f)]
            if len(known) == 1:
                check[known[0]] = check.pop(key)
    return data


def parse_groundedness(raw: str, answer: str, context: str) -> dict:
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.partition("\n")[2].removesuffix("```")
        verdict = Verdict.model_validate(_repair_keys(json.loads(raw)))
    except (ValueError, TypeError, AttributeError, ValidationError) as exc:
        raise JudgeFormatError("Invalid groundedness JSON contract") from exc
    for check in verdict.checks:
        if not _quote_found(check.answer_quote, answer):
            raise JudgeFormatError("answer_quote must appear in answer")
        if check.context_quote is not None and not _quote_found(
            check.context_quote, context
        ):
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
