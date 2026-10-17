"""Нормы, привязки и полнота реально переданного контекста; чистые функции."""

# ANCHOR: текст метки/чанка + parse_units → норма, контекст A/B, found/complete.
# Пороги: match ≥0.6, разрыв <0.1 спорный, complete ≥0.9 уникальных слов ≥4 знаков.
# Родитель >6000: целые абзацы вокруг лучшего совпадения, до 3000 знаков с каждой стороны.

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property

from eval.chunking_ab.run import words


@dataclass(frozen=True)
class Norm:
    doc: str
    kind: str
    number: str
    paragraphs: tuple[str, ...]
    ordinal: int

    @property
    def id(self) -> str:
        return f"{self.doc}::{self.kind}:{self.number}:{self.ordinal}"

    @cached_property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    @cached_property
    def words(self) -> frozenset[str]:
        return words(self.text)


@dataclass(frozen=True)
class Candidate:
    norm: Norm
    score: float


@dataclass(frozen=True)
class Match:
    norm: Norm | None
    score: float
    reason: str | None
    candidates: tuple[Candidate, ...]


def match_norm(doc: str, text: str, norms: list[Norm]) -> Match:
    target = words(text)
    if not target:
        return Match(None, 0.0, "empty_words", ())
    ranked = sorted(
        (
            (n, Fraction(len(target & n.words), len(target)))
            for n in norms
            if n.doc == doc
        ),
        key=lambda pair: (-pair[1], len(pair[0].text), pair[0].id),
    )
    candidates = tuple(Candidate(n, float(s)) for n, s in ranked[:2])
    score = candidates[0].score if candidates else 0.0
    if not ranked or ranked[0][1] < Fraction(3, 5):
        return Match(None, score, "below_threshold", candidates)
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < Fraction(1, 10):
        return Match(None, score, "ambiguous", candidates)
    return Match(ranked[0][0], score, None, candidates)


@dataclass(frozen=True)
class Chunk:
    doc: str
    text: str  # текст для эмбеддинга и бюджета A
    body: str  # без заголовка раздела
    parent: Norm | None


@dataclass(frozen=True)
class Context:
    doc: str
    text: str
    body: str
    parent: Norm | None
    chunk_index: int
    expanded: bool


def expand_parent(norm: Norm, body: str) -> str:
    if len(norm.text) <= 6000:
        return norm.text
    # Сначала точная последовательность слов (сохраняет положение сквозь переносы PDF).
    token_re = re.compile(r"[а-яёa-z0-9]+")
    tokens = token_re.findall(body.lower())
    para_tokens = [token_re.findall(p.lower()) for p in norm.paragraphs]
    flat, owners = [], []
    for i, ts in enumerate(para_tokens):
        flat.extend(ts)
        owners.extend([i] * len(ts))
    anchors = None
    if tokens:
        for i in range(len(flat) - len(tokens) + 1):
            if flat[i : i + len(tokens)] == tokens:
                anchors = (owners[i], owners[i + len(tokens) - 1] + 1)
                break
    if anchors is None:
        target = words(body)
        scores = [len(words(p) & target) for p in norm.paragraphs]
        anchor = max(range(len(scores)), key=scores.__getitem__)
        anchors = (anchor, anchor + 1)
    first, last = anchors
    lo, hi = first, last
    while lo > 0 and len("\n".join(norm.paragraphs[lo - 1 : first])) <= 3000:
        lo -= 1
    while (
        hi < len(norm.paragraphs)
        and len("\n".join(norm.paragraphs[last : hi + 1])) <= 3000
    ):
        hi += 1
    return "\n".join(norm.paragraphs[lo:hi])


def select_context(
    order: list[int], chunks: list[Chunk], scheme: str, budget: int
) -> list[Context]:
    if scheme not in ("A", "B"):
        raise ValueError(f"unknown scheme: {scheme}")
    chosen: list[Context] = []
    seen: set[str] = set()
    total = 0
    for i in order:
        c = chunks[i]
        expanded = scheme == "B" and c.parent is not None
        if expanded:
            if c.parent.id in seen:
                continue
            text = body = expand_parent(c.parent, c.body)
        else:
            text, body = c.text, c.body
        cost = len(text) + bool(chosen)  # newline между элементами тоже занимает бюджет
        if total + cost > budget:
            break
        chosen.append(Context(c.doc, text, body, c.parent, int(i), expanded))
        total += cost
        if expanded:
            seen.add(c.parent.id)
    return chosen


def context_metrics(
    context: list[Context], norm: Norm, scheme: str
) -> tuple[bool, bool, float]:
    local = [c for c in context if c.doc == norm.doc]
    covered: set[str] = set()
    found = False
    for c in local:
        w = words(c.body)
        covered.update(w)
        found |= c.parent == norm or (
            scheme == "A"
            and bool(norm.words)
            and len(w & norm.words) / len(norm.words) >= 0.6
        )
    coverage = len(covered & norm.words) / len(norm.words) if norm.words else 0.0
    return found, coverage >= 0.9, coverage
