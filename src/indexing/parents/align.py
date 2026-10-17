"""Позиционное выравнивание детей и структурных норм; чистые offline-функции."""

# ANCHOR: последовательность тел C + упорядоченные parse_units → родители DP/Витерби.
# Пара = доля слов C в единице; metadata bonus ≤0.06, rollback penalty=0.15.
# Отказ при overlap <0.60; неоднозначность словаря разрешает путь, а не ослабление порога.
# Ported from eval/chunking_ab/align.py; the evaluation copy stays unchanged.

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .norms import Norm, words

THRESHOLD = 0.60
ROLLBACK_PENALTY = 0.15


@dataclass(frozen=True)
class AlignInput:
    body: str
    article: str = ""
    part: str = ""
    heading_path: str = ""


@dataclass(frozen=True)
class Alignment:
    parent: Norm | None
    candidate: Norm | None
    score: float
    best_score: float
    bonus: float
    reason: str | None
    rollback: bool


def metadata_bonus(c: AlignInput, n: Norm) -> float:
    bonus = 0.03 if n.kind == "article" and c.article == n.number else 0.0
    if c.part and re.search(rf"(?:^|\n){re.escape(c.part)}\.\s", n.text):
        bonus += 0.015
    hint = words(c.heading_path)
    if hint:
        bonus += 0.015 * len(hint & n.words) / len(hint)
    return bonus


def align_document(
    chunks: list[AlignInput],
    units: list[Norm],
    rollback: int = 1,
    threshold: float = THRESHOLD,
) -> list[Alignment]:
    if rollback not in (0, 1):
        raise ValueError("rollback must be 0 or 1")
    if not units:
        return [Alignment(None, None, 0, 0, 0, "no_units", False) for _ in chunks]
    if not chunks:
        return []
    cw = [words(c.body) for c in chunks]
    scores = np.array(
        [[len(w & n.words) / len(w) if w else 0.0 for n in units] for w in cw]
    )
    bonuses = np.array([[metadata_bonus(c, n) for n in units] for c in chunks])
    # Небольшая специфичность (≤0.01) разруливает точные словарные ничьи.
    specificity = np.array(
        [[0.01 * len(w & n.words) / max(1, len(n.words)) for n in units] for w in cw]
    )
    emission = scores + bonuses + specificity
    prev = emission[0].copy()
    back = np.zeros(scores.shape, dtype=np.int32)
    js = np.arange(len(units))
    for i in range(1, len(chunks)):
        prefix = np.maximum.accumulate(prev)
        # Первый индекс максимума делает ничьи детерминированными.
        peaks = np.r_[True, prefix[1:] > prefix[:-1]]
        args = np.maximum.accumulate(np.where(peaks, js, 0))
        best = prefix.copy()
        if rollback:
            candidate = prev[1:] - ROLLBACK_PENALTY
            better = candidate > best[:-1]
            best[:-1] = np.where(better, candidate, best[:-1])
            args[:-1] = np.where(better, js[1:], args[:-1])
        back[i] = args
        prev = best + emission[i]
    path = [int(np.argmax(prev))]
    for i in range(len(chunks) - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    result = []
    for i, j in enumerate(path):
        score = float(scores[i, j])
        reason = (
            "empty_words"
            if not cw[i]
            else "below_threshold" if score < threshold else None
        )
        result.append(
            Alignment(
                units[j] if reason is None else None,
                units[j],
                score,
                float(scores[i].max()),
                float(bonuses[i, j]),
                reason,
                i > 0 and j < path[i - 1],
            )
        )
    return result
