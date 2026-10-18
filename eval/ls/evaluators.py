"""Code evaluators for LangSmith experiments — pure functions, no network.

Each evaluator returns {"key", "score", "comment"}; score is 1, 0 or None
(None = not applicable or target failed, excluded from aggregates). LangSmith binds evaluator
arguments by name, so the parameters must be `outputs` and `reference_outputs`.
"""

from __future__ import annotations

import re

from eval.ls.policy import is_oos
from eval.metrics import compute_inversion_detected

# Same as run_v7_eval._REFUSAL_RE plus "неверно" (false premise refuted).
# Not imported: run_v7_eval loads settings, LLM factory and logging on import.
_REFUSAL_RE = re.compile(r"\bнет\b|не могу|неверно")
_REFUSAL_WINDOW = 200
_RANK_RE = re.compile(r"rank=(\d+)")


def _result(key: str, score: int | None, comment: str) -> dict:
    return {"key": key, "score": score, "comment": comment}


def _target_failed(outputs: dict) -> bool:
    # LangSmith still calls evaluators when the target raised, with outputs={}.
    # state_to_outputs always sets "answer", so its absence means a failed run.
    return bool(
        "answer" not in outputs
        or outputs.get("generation_failed")
        or outputs.get("technical_failure")
        or outputs.get("target_error")
    )


def doc_number(chunk_id: str) -> str:
    """Source file name without extension: '782н.pdf#431' -> '782н'.

    doc_cited matches this name in the answer, so for files named by title
    ('тк рф', '1479 ппр') it checks that the source file was named.
    """
    # rstrip: 'Статья 5.27.1..pdf' -> 'Статья 5.27.1'
    return chunk_id.split("#")[0].rsplit(".", 1)[0].rstrip(".")


def retrieval_hit(outputs: dict, reference_outputs: dict) -> dict:
    """1 if any relevant chunk was retrieved; comment holds the first hit rank (from 1)."""
    if _target_failed(outputs):
        return _result("retrieval_hit", None, "target_error")
    if is_oos(reference_outputs):
        return _result("retrieval_hit", None, "oos")
    relevant = set(reference_outputs.get("relevant_chunk_ids") or [])
    if not relevant:
        return _result("retrieval_hit", None, "no_labels")
    for rank, chunk_id in enumerate(outputs.get("chunk_ids") or [], start=1):
        if chunk_id in relevant:
            return _result("retrieval_hit", 1, f"rank={rank}")
    return _result("retrieval_hit", 0, "miss")


def doc_cited(outputs: dict, reference_outputs: dict) -> dict:
    """1 if the answer names the number of any relevant document, case-insensitive."""
    if _target_failed(outputs):
        return _result("doc_cited", None, "target_error")
    if is_oos(reference_outputs):
        return _result("doc_cited", None, "oos")
    answer = (outputs.get("answer") or "").lower()
    relevant = reference_outputs.get("relevant_chunk_ids") or []
    if not relevant:
        return _result("doc_cited", None, "no_labels")
    # sorted: a stable comment when several documents are cited
    for doc in sorted({doc_number(chunk_id) for chunk_id in relevant}):
        # (?<!\d)/(?!\d): number must not be part of another number ("2464" not in "24640")
        if re.search(rf"(?<!\d){re.escape(doc.lower())}(?!\d)", answer):
            return _result("doc_cited", 1, f"cited={doc}")
    return _result("doc_cited", 0, "not_cited")


def forbidden(outputs: dict, reference_outputs: dict) -> dict:
    """0 if the answer contains a must_not_contain pattern (norm inversion), else 1.

    forbidden_claims are free-text statements and are checked by the judge, not here.
    """
    if _target_failed(outputs):
        return _result("forbidden", None, "target_error")
    pattern = (reference_outputs.get("must_not_contain") or "").strip()
    if not pattern:
        return _result("forbidden", None, "no_pattern")
    if compute_inversion_detected(pattern, outputs.get("answer") or ""):
        return _result("forbidden", 0, "inversion")
    return _result("forbidden", 1, "clean")


def oos_handled(outputs: dict, reference_outputs: dict) -> dict:
    """1 if an OOS question was refused: abstain route, empty answer or refusal text."""
    if _target_failed(outputs):
        return _result("oos_handled", None, "target_error")
    if reference_outputs.get("oos_type") == "false_premise":
        return _result("oos_handled", None, "false_premise_scored_by_correctness")
    if not is_oos(reference_outputs):
        return _result("oos_handled", None, "in_scope")
    if outputs.get("route_decision") == "abstain":
        return _result("oos_handled", 1, "abstain")
    answer = (outputs.get("answer") or "").strip()
    if not answer:
        # Domain gate refused before the graph: empty answer, no route.
        return _result("oos_handled", 1, "empty")
    if _REFUSAL_RE.search(answer.lower()[:_REFUSAL_WINDOW]):
        return _result("oos_handled", 1, "refusal")
    return _result("oos_handled", 0, "answered")


CODE_EVALUATORS = [retrieval_hit, doc_cited, forbidden, oos_handled]


def summarize(rows: list[dict[str, dict]]) -> dict:
    """Aggregate per-example results into rates and MRR.

    rows: [{evaluator_key: {"score": ..., "comment": ...}}, ...]
    Returns {key: share of 1 among non-None scores, "mrr": float | None, "n": {key: count}}.
    MRR is over retrieval_hit rows with a score; a miss counts as 0.
    """
    scores: dict[str, list[int]] = {}
    reciprocal_ranks: list[float] = []
    for row in rows:
        for key, result in row.items():
            score = result.get("score")
            if score is None:
                continue
            scores.setdefault(key, []).append(score)
            if key == "retrieval_hit":
                match = _RANK_RE.search(result.get("comment") or "")
                reciprocal_ranks.append(1 / int(match.group(1)) if match else 0.0)

    summary: dict = {key: sum(values) / len(values) for key, values in scores.items()}
    summary["mrr"] = (
        sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else None
    )
    summary["n"] = {key: len(values) for key, values in scores.items()}
    return summary


def summarize_cost(outputs: list[dict]) -> dict:
    """Total and mean cost per example from target outputs.

    Models without a rate card are listed in `unpriced_models`: their calls count
    as $0, so a non-empty list means the totals are a lower bound.
    """
    costs = [o["cost_usd"] for o in outputs if o.get("cost_usd") is not None]
    unpriced = sorted({m for o in outputs for m in o.get("unpriced_models") or []})
    total = sum(costs)
    return {
        "total_usd": total,
        "mean_usd": total / len(costs) if costs else None,
        "n": len(costs),
        "unpriced_models": unpriced,
    }
