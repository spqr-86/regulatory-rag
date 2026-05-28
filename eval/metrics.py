"""Deterministic evaluation metrics for V7/V8 RAG pipeline.

No LLM calls. Uses pymorphy3 lemmatization (same as nlp_core.extract_keywords).

Public API:
    extract_key_phrases(text) -> set[str]
    compute_completeness(ground_truth, answer) -> float
    compute_abstain_rate(results) -> float
    compute_false_abstain_rate(results) -> float
    compute_correct_abstain_rate(results) -> float
    compute_retrieval_stats(results) -> dict
    compute_inversion_detected(must_not_contain, answer) -> bool
    compute_inversion_rate(results) -> float
    parse_citations(answer) -> list[dict]
    compute_citation_rate(answer) -> float
    compute_citation_in_retrieval(answer, passages) -> float
    compute_citation_doc_match(answer, passages) -> float
"""

from __future__ import annotations

import re

from src.v7.nlp_core import extract_keywords

# Regex for [Фрагмент N: Document, п. X.X] or [Фрагмент N: Document, без пункта]
_CITATION_RE = re.compile(
    r"\[Фрагмент\s+(\d+):\s*([^,\]]+?)(?:,\s*п\.\s*([^\]]+?)|,\s*без\s+пункта)?\s*\]",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")


def extract_key_phrases(text: str) -> set[str]:
    """Key lemmas from text (wrapper around nlp_core.extract_keywords)."""
    return extract_keywords(text)


def compute_completeness(ground_truth: str, answer: str) -> float:
    """Fraction of ground_truth key phrases found in the answer.

    Algorithm:
    1. Extract key lemmas from ground_truth
    2. Extract key lemmas from answer
    3. completeness = |intersection| / |gt_lemmas|

    Returns:
        float in [0.0, 1.0]. 1.0 if ground_truth is empty (nothing to check).
    """
    gt_lemmas = extract_key_phrases(ground_truth)
    if not gt_lemmas:
        return 1.0
    if not answer:
        return 0.0
    answer_lemmas = extract_key_phrases(answer)
    found = len(gt_lemmas & answer_lemmas)
    return found / len(gt_lemmas)


def _is_abstained(result: dict) -> bool:
    """True if the answer is missing or explicitly marked as abstained."""
    if result.get("abstained") is True:
        return True
    return not bool(result.get("answer", "").strip())


def compute_abstain_rate(results: list[dict]) -> float:
    """Fraction of requests for which the system declined to answer.

    Args:
        results: list of dicts with fields "answer" (str) and optionally "abstained" (bool)

    Returns:
        float in [0.0, 1.0]
    """
    if not results:
        return 0.0
    abstained = sum(1 for r in results if _is_abstained(r))
    return abstained / len(results)


def compute_false_abstain_rate(results: list[dict]) -> float:
    """Fraction of in-domain queries (is_oos=False) for which the system abstained.

    False abstain = system stayed silent on a question it should have answered.
    Target: = 0.

    Args:
        results: list of dicts with fields "answer", "is_oos" (bool)
    """
    if not results:
        return 0.0
    domain_results = [r for r in results if not r.get("is_oos", False)]
    if not domain_results:
        return 0.0
    false_abstains = sum(1 for r in domain_results if _is_abstained(r))
    return false_abstains / len(domain_results)


def compute_correct_abstain_rate(results: list[dict]) -> float:
    """Fraction of OOS queries (is_oos=True) for which the system correctly abstained.

    Target: = 1.0 (system always stays silent on out-of-scope questions).

    Args:
        results: list of dicts with fields "answer", "is_oos" (bool)
    """
    if not results:
        return 1.0
    oos_results = [r for r in results if r.get("is_oos", False)]
    if not oos_results:
        return 1.0  # no OOS rows → condition trivially satisfied
    correct = sum(1 for r in oos_results if _is_abstained(r))
    return correct / len(oos_results)


def compute_inversion_detected(must_not_contain: str, answer: str) -> bool:
    """True if the answer contains a forbidden pattern (norm inversion).

    Inversion = system stated an incorrect numeric value or regulatory norm.
    Example: "repeat briefing once a year" instead of "once every 6 months".

    Args:
        must_not_contain: patterns separated by '|' (case-insensitive substring match).
                          Empty string → check is skipped.
        answer: system answer.

    Returns:
        True = inversion detected (bad). False = all clear or check not applicable.
    """
    if not must_not_contain or not must_not_contain.strip() or not answer:
        return False
    answer_lower = answer.lower()
    patterns = [p.strip() for p in must_not_contain.split("|") if p.strip()]
    return any(p.lower() in answer_lower for p in patterns)


def compute_inversion_rate(results: list[dict]) -> float:
    """Fraction of answers with inversions among rows where must_not_contain is set.

    Rows without must_not_contain are excluded.

    Args:
        results: list of dicts with fields "must_not_contain" (str) and "inversion_detected" (bool).

    Returns:
        float in [0.0, 1.0]. 0.0 if no checkable rows exist.
    """
    checkable = [r for r in results if r.get("must_not_contain", "").strip()]
    if not checkable:
        return 0.0
    inversions = sum(1 for r in checkable if r.get("inversion_detected", False))
    return inversions / len(checkable)


def parse_citations(answer: str) -> list[dict]:
    """Extract citations in the format [Фрагмент N: Document, п. X.X] from the answer.

    Returns:
        List of dicts with keys:
            n (int)       — fragment number
            doc (str)     — document name
            section (str) — section number or "" if "без пункта"
    """
    if not answer:
        return []
    results = []
    for m in _CITATION_RE.finditer(answer):
        results.append(
            {
                "n": int(m.group(1)),
                "doc": m.group(2).strip(),
                "section": (m.group(3) or "").strip(),
            }
        )
    return results


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences without cutting inside [...] brackets.

    Citations are masked before splitting so dots inside [Фрагмент N: Doc, п. X.X]
    do not break sentence boundaries.
    """
    # Replace each character inside [...] with 'X' (length preserved)
    masked = re.sub(r"\[[^\]]*\]", lambda m: "X" * len(m.group()), text)
    parts: list[str] = []
    last = 0
    for m in _SENTENCE_SPLIT_RE.finditer(masked):
        end = m.end()
        parts.append(text[last:end].strip())
        last = end
    if last < len(text):
        tail = text[last:].strip()
        if tail:
            parts.append(tail)
    return [p for p in parts if p]


def compute_citation_rate(answer: str) -> float:
    """Fraction of sentences that contain at least one citation.

    Sentences are delimited by .!? outside square brackets.

    Returns:
        float in [0.0, 1.0]. 0.0 if there are no sentences or answer is empty.
    """
    if not answer:
        return 0.0
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0
    with_citation = sum(1 for s in sentences if _CITATION_RE.search(s))
    return with_citation / len(sentences)


def compute_citation_in_retrieval(answer: str, passages: list[dict]) -> float:
    """Fraction of citations where N is within [1, len(passages)].

    Guards against hallucinated fragment numbers (N > number of actual passages).

    Args:
        answer: system answer.
        passages: list of passages returned from final_passages.

    Returns:
        float in [0.0, 1.0]. 0.0 if there are no citations or passages is empty.
    """
    citations = parse_citations(answer)
    if not citations or not passages:
        return 0.0
    n_passages = len(passages)
    valid = sum(1 for c in citations if 1 <= c["n"] <= n_passages)
    return valid / len(citations)


def compute_citation_doc_match(answer: str, passages: list[dict]) -> float:
    """Fraction of citations where the document name matches the source of fragment N.

    Matching: cited_doc is a substring of source (case-insensitive).
    Skips citations with N outside the passages range.

    Args:
        answer: system answer.
        passages: list of passages with metadata.source.

    Returns:
        float in [0.0, 1.0]. 0.0 if there are no checkable citations.
    """
    citations = parse_citations(answer)
    if not citations or not passages:
        return 0.0
    checkable = [c for c in citations if 1 <= c["n"] <= len(passages)]
    if not checkable:
        return 0.0
    matched = 0
    for c in checkable:
        source = passages[c["n"] - 1].get("metadata", {}).get("source", "").lower()
        cited_doc = c["doc"].lower().strip()
        if cited_doc and cited_doc in source:
            matched += 1
    return matched / len(checkable)


def compute_retrieval_stats(results: list[dict]) -> dict:
    """Average retrieval stats across all queries.

    Args:
        results: list of dicts with fields "top_score" (float), "passage_count" (int)

    Returns:
        dict with keys "avg_top_score", "avg_passage_count"
    """
    if not results:
        return {"avg_top_score": 0.0, "avg_passage_count": 0.0}

    top_scores = [r.get("top_score", 0.0) for r in results]
    passage_counts = [r.get("passage_count", 0) for r in results]
    n = len(results)

    return {
        "avg_top_score": round(sum(top_scores) / n, 4),
        "avg_passage_count": round(sum(passage_counts) / n, 2),
    }
