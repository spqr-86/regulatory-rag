"""Saved route observations, with exact question matching and explicit uncertainty."""

# ANCHOR: local JSON/JSONL → dated route evidence → latest observed route or unknown.
# No inherited path from retrieval-only reports, no semantic/fuzzy question matching.
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Evidence:
    route: str
    file: str
    line: int
    timestamp: str
    field: str


@dataclass(frozen=True)
class Route:
    question: str
    route: str
    source: str
    selected: Evidence | None
    evidence: tuple[Evidence, ...]
    conflicting_history: bool
    reason: str


def key(text: str) -> str:
    return " ".join(text.split())


def choose_route(question: str, evidence: list[Evidence]) -> Route:
    evidence = sorted(set(evidence), key=lambda e: (e.timestamp, e.file, e.line))
    if not evidence:
        return Route(
            question,
            "unknown",
            "c",
            None,
            (),
            False,
            "no saved route; production simple cache unavailable",
        )
    latest = evidence[-1].timestamp
    recent = [e for e in evidence if e.timestamp == latest]
    conflict = len({e.route for e in evidence}) > 1
    if len({e.route for e in recent}) > 1:
        return Route(
            question,
            "unknown",
            "c",
            None,
            tuple(evidence),
            True,
            "conflicting latest observations",
        )
    return Route(
        question,
        recent[-1].route,
        "a",
        recent[-1],
        tuple(evidence),
        conflict,
        "latest saved observation; not a current replay",
    )


def scan_file(path: Path) -> list[tuple[str, Evidence]]:
    raw = path.read_text(encoding="utf-8")
    try:
        documents = [(json.loads(raw), 0)]
    except json.JSONDecodeError:
        documents = []
        for line, text in enumerate(raw.splitlines(), 1):
            try:
                documents.append((json.loads(text), line))
            except json.JSONDecodeError:
                continue
    result = []
    date_match = re.search(r"20\d\d-\d\d-\d\d", str(path))
    fallback_date = date_match.group() if date_match else ""

    def visit(value, date, base_line):
        if isinstance(value, list):
            for item in value:
                visit(item, date, base_line)
        elif isinstance(value, dict):
            date = str(
                value.get("ts") or value.get("timestamp") or value.get("date") or date
            )
            question = value.get("question")
            route, field = value.get("path"), "path"
            # Terminal abstain can hide an executed complex branch.
            usages = value.get("usage") or value.get("models") or []
            if not isinstance(usages, list):
                usages = []
            attempts = value.get("retrieval_attempts") or []
            if not isinstance(attempts, list):
                attempts = []
            if any(
                isinstance(a, dict) and a.get("stage") == "complex"
                for a in usages + attempts
            ):
                route, field = "complex", "usage/models/retrieval_attempts.stage"
            if isinstance(question, str) and route in ("simple", "complex"):
                needle = json.dumps(question, ensure_ascii=False)
                pos = raw.find(needle)
                if pos < 0:
                    pos = raw.find(json.dumps(question))
                line = base_line or (raw.count("\n", 0, pos) + 1 if pos >= 0 else 1)
                result.append((question, Evidence(route, str(path), line, date, field)))
            for child in value.values():
                if isinstance(child, (list, dict)):
                    visit(child, date, base_line)

    for document, line in documents:
        visit(document, fallback_date, line)
    return result
