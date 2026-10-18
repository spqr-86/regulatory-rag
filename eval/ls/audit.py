"""Record judge evidence/usage and contain provider errors without scoring them zero."""

# ANCHOR: evaluator context + model proxy → append-only audit records, no paid retries.

from __future__ import annotations

import contextvars
import inspect
import json

from eval.ls.evaluators import _result
from eval.ls.policy import is_oos
from src.v7.usage import usage_from_response

_CONTEXT = contextvars.ContextVar("eval_judge_context", default=None)


class AuditedModel:
    def __init__(self, model, model_name: str):
        self.model, self.model_name = model, model_name

    def invoke(self, messages):
        context = _CONTEXT.get()
        try:
            reply = self.model.invoke(messages)
        except Exception as exc:
            if context is not None:
                context["usage"].append(
                    {"model": self.model_name, "usage_unknown": True}
                )
                context["sink"].append(
                    "judges",
                    {
                        **context["identity"],
                        "provider_error": type(exc).__name__,
                        "usage_unknown": True,
                    },
                )
            raise
        usage = usage_from_response(reply, self.model_name, "eval_judge")
        metadata = (
            getattr(reply, "usage_metadata", None)
            or (getattr(reply, "response_metadata", None) or {}).get("token_usage")
            or {}
        )
        counters = [
            metadata.get("input_tokens", metadata.get("prompt_tokens")),
            metadata.get("output_tokens", metadata.get("completion_tokens")),
        ]
        usage["usage_unknown"] = not all(type(v) is int and v >= 0 for v in counters)
        if context is not None:
            context["usage"].append(usage)
            content = reply.content
            try:
                raw = content.strip()
                if raw.startswith("```"):
                    raw = raw.partition("\n")[2].removesuffix("```")
                content = json.loads(raw)
            except (ValueError, TypeError, AttributeError):
                pass
            context["sink"].append(
                "judges",
                {
                    **context["identity"],
                    "usage": usage,
                    "verdict": content,
                    "finish_reason": (
                        getattr(reply, "response_metadata", None) or {}
                    ).get("finish_reason"),
                },
            )
        return reply


def audited(evaluator, keys: tuple[str, ...], sink, usages: dict, cancelled=None):
    parameters = inspect.signature(evaluator).parameters

    def evaluate(run, example):
        identity = {
            "case_id": (example.metadata or {}).get("case_id"),
            "example_id": str(example.id),
            "langsmith_run_id": str(run.id),
            "query_id": (run.outputs or {}).get("query_id"),
            "evaluator": evaluator.__name__,
        }
        usage = []
        token = _CONTEXT.set({"identity": identity, "sink": sink, "usage": usage})
        try:
            try:
                if cancelled is not None and cancelled.is_set():
                    raise InterruptedError("Evaluation cancelled")
                arguments = {
                    "inputs": run.inputs or {},
                    "outputs": run.outputs or {},
                    "reference_outputs": example.outputs or {},
                }
                result = evaluator(
                    **{k: v for k, v in arguments.items() if k in parameters}
                )
                scores = result if isinstance(result, list) else [result]
            except Exception as exc:
                scores = [
                    _result(key, None, f"evaluator_error: {type(exc).__name__}")
                    for key in keys
                ]
            if is_oos(example.outputs or {}):
                scores = [
                    (
                        s
                        if s["key"] in {"oos_handled", "forbidden"}
                        else _result(s["key"], None, "oos")
                    )
                    for s in scores
                ]
            if (example.outputs or {}).get("oos_type") == "false_premise":
                correction = next(
                    (s for s in scores if s["key"] == "correctness"), None
                )
                if correction:
                    scores.append({**correction, "key": "premise_corrected"})
            sink.append("assessments", {**identity, "scores": scores})
            for score in scores:
                score["evaluator_info"] = {"assessment_id": sink.assessment_id}
            usages.setdefault(str(run.id), []).extend(usage)
            return {"results": scores}
        finally:
            _CONTEXT.reset(token)

    evaluate.__name__ = evaluator.__name__
    return evaluate
