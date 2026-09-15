"""Department scope filters for the two-level retrieval (spec §6, issue #36 stage 2).

# ANCHOR: scope filter
# Role: one filter shape shared by dense (Chroma where), BM25 and cross_ref, so a
#   foreign unit's document cannot enter context through any path.
# Input: unit_id or None. Output: (external_filter, internal_filter).
# Shape: flat dict, value is a scalar (equality) or {"$in": [...]}; Chroma accepts
#   it via ChromaBackend.get_by_filter ($and wrapping), BM25 via matches_filter.
# No unit selected: internal search sees company documents only (spec §6).
"""

from __future__ import annotations

from typing import Optional

COMPANY_AUDIENCE = "company"


def build_scope_filters(unit_id: Optional[str]) -> tuple[dict, dict]:
    audiences = [COMPANY_AUDIENCE] + ([unit_id] if unit_id else [])
    external = {"source_type": "external"}
    internal = {"source_type": "internal", "audience": {"$in": audiences}}
    return external, internal


def matches_filter(metadata: dict, filters: Optional[dict]) -> bool:
    for key, expected in (filters or {}).items():
        value = metadata.get(key)
        if isinstance(expected, dict):
            if set(expected) != {"$in"}:
                raise ValueError(f"unsupported filter operator for {key}: {expected}")
            if value not in expected["$in"]:
                return False
        elif value != expected:
            return False
    return True
