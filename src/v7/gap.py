"""V7: структурный gap и детектор enumeration — общий код решения.

Живёт отдельно от узлов: и validate_context, и evaluate_triage читают
отсюда. До 09.09.2026 код лежал в узле, что делало чистый валидатор
зависимым от слоя графа.
"""

from __future__ import annotations

# ANCHOR: Builds a node-free structural gap from passage text and detects
# enumeration intent. Inputs are passages/query; output is TriageGap/bool.

import re
from typing import Dict, List, Optional

from src.v7.cross_ref import _extract_refs
from src.v7.state_types import GapRef, TriageGap

# Enumeration question patterns — require complete coverage of all categories/conditions.
# For such queries rag_simple may return an incomplete answer even at high top_score.
ENUMERATION_PATTERNS = [
    r"\bкто\s+проходит\b",
    r"\bкто\s+обязан\b",
    r"\bкакие\s+категори[яи]\b",
    r"\bв\s+каких\s+случаях\b",
    r"\bкогда\s+не\s+требуется\b",
    r"\bкому\s+не\s+требуется\b",
    r"\bкто\s+освобождается\b",
    r"\bперечислите\b",
    r"\bкакие\s+работники\b",
    r"\bкаким\s+работникам\b",
]


def has_enumeration_intent(query: str) -> bool:
    """True if the query requires complete enumeration of categories/conditions.

    Such queries are routed to rag_complex even when simple-triage is sufficient,
    because the answer is often spread across multiple document clauses.
    """
    q = query.lower()
    return any(re.search(pattern, q) for pattern in ENUMERATION_PATTERNS)


def passage_source(passage: dict) -> str:
    return passage.get("metadata", {}).get("source") or passage.get("doc_id", "")


def _ref_present(kind: str, num: str, content: str) -> bool:
    """True if the chunk text structurally *contains* the referenced unit.

    Deliberately narrower than cross_ref._ref_matches_doc: a chunk merely
    naming "пункт 12" is what creates the gap, so a phrase match must not
    count as its resolution. Only a structural heading does.
    """
    value = re.escape(num)
    if kind == "clause":
        return bool(re.search(rf"(?m)^\s*{value}\.(?=\s)", content))
    if kind == "article":
        return bool(
            re.search(rf"(?mi)^\s*(?:стать\w+\s+)?{value}[.\s]", content)
            and re.search(rf"(?i)стать\w+\s+{value}\b", content)
        )
    if kind == "subpara":
        # The chunker flattens ordered lists and prepends a running item number,
        # so "46. а)" is stored as "6. а)". Tolerate an optional leading number.
        return bool(re.search(rf"(?mi)^\s*(?:\d+\.\s+)?{value}\)", content))
    return False


def build_gap(
    passages: List[dict], resolve_in: Optional[List[dict]] = None
) -> TriageGap:
    """Describe what the retrieved text names but does not contain.

    Refs are extracted from the top-5 passages — the same slice that trips
    the crossref escalation — and deduplicated by (doc_id, kind, num).
    Resolution is checked across the whole of `resolve_in` (defaults to
    `passages`) within the same source: a clause sitting at position 9 is
    not a gap. Markers carry no doc_id, so one number named in two documents
    gives two refs and a single marker; the marker stays open while any of
    its refs is unresolved.
    """
    haystack = passages if resolve_in is None else resolve_in

    refs: List[GapRef] = []
    seen: set[tuple[str, str, str]] = set()
    for passage in passages[:5]:
        source = passage_source(passage)
        for kind, num in _extract_refs(passage.get("text", "")):
            key = (source, kind, num)
            if key in seen:
                continue
            seen.add(key)
            refs.append({"kind": kind, "num": num, "doc_id": source})

    by_source: Dict[str, List[str]] = {}
    for passage in haystack:
        by_source.setdefault(passage_source(passage), []).append(
            passage.get("text", "")
        )

    resolved: Dict[str, bool] = {}
    for ref in refs:
        marker = f"{ref['kind']}:{ref['num']}"
        present = any(
            _ref_present(ref["kind"], ref["num"], text)
            for text in by_source.get(ref["doc_id"], [])
        )
        resolved[marker] = resolved.get(marker, True) and present

    closed = [marker for marker, resolved_ in resolved.items() if resolved_]
    open_ = [marker for marker, resolved_ in resolved.items() if not resolved_]

    return {"kind": "unresolved_ref", "refs": refs, "closed": closed, "open": open_}
