"""Code evaluators for LangSmith experiments — pure functions, no network.

Each evaluator returns {"key", "score", "comment"}; score is 1, 0 or None
(None = not applicable, excluded from aggregates). LangSmith binds evaluator
arguments by name, so the parameters must be `outputs` and `reference_outputs`.
"""

from __future__ import annotations

import re

from eval.metrics import compute_inversion_detected

# Same as run_v7_eval._REFUSAL_RE plus "неверно" (false premise refuted).
# Not imported: run_v7_eval loads settings, LLM factory and logging on import.
_REFUSAL_RE = re.compile(r"\bнет\b|не могу|неверно")
_REFUSAL_WINDOW = 200
_RANK_RE = re.compile(r"rank=(\d+)")


def _result(key: str, score: int | None, comment: str) -> dict:
    return {"key": key, "score": score, "comment": comment}


def doc_number(chunk_id: str) -> str:
    """'782н.pdf#431' -> '782н'."""
    return chunk_id.split(".")[0]


def retrieval_hit(outputs: dict, reference_outputs: dict) -> dict:
    """1 if any relevant chunk was retrieved; comment holds the first hit rank (from 1)."""
    if reference_outputs.get("oos_type"):
        return _result("retrieval_hit", None, "oos")
    relevant = set(reference_outputs.get("relevant_chunk_ids") or [])
    for rank, chunk_id in enumerate(outputs.get("chunk_ids") or [], start=1):
        if chunk_id in relevant:
            return _result("retrieval_hit", 1, f"rank={rank}")
    return _result("retrieval_hit", 0, "miss")


def doc_cited(outputs: dict, reference_outputs: dict) -> dict:
    """1 if the answer names the number of any relevant document, case-insensitive."""
    if reference_outputs.get("oos_type"):
        return _result("doc_cited", None, "oos")
    answer = (outputs.get("answer") or "").lower()
    relevant = reference_outputs.get("relevant_chunk_ids") or []
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
    pattern = (reference_outputs.get("must_not_contain") or "").strip()
    if not pattern:
        return _result("forbidden", None, "no_pattern")
    if compute_inversion_detected(pattern, outputs.get("answer") or ""):
        return _result("forbidden", 0, "inversion")
    return _result("forbidden", 1, "clean")


def oos_handled(outputs: dict, reference_outputs: dict) -> dict:
    """1 if an OOS question was refused: abstain route, empty answer or refusal text."""
    if not reference_outputs.get("oos_type"):
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
