"""Shared applicability rules for the generic V7 evaluation."""

# ANCHOR: reference → case type; a false premise requires a substantive correction.

from __future__ import annotations


def case_type(reference: dict) -> str:
    value = (reference.get("oos_type") or "").strip()
    if value not in ("", "false_premise", "out_of_scope"):
        raise ValueError(f"Unknown oos_type: {value!r}")
    return value or "regular"


def is_oos(reference: dict) -> bool:
    return case_type(reference) == "out_of_scope"
