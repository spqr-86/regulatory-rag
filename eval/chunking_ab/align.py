"""Позиционное выравнивание C и структурные дети; чистые offline-функции."""

# ANCHOR: последовательность тел C + упорядоченные parse_units → родители DP/Витерби.
# Пара = доля слов C в единице; metadata bonus ≤0.06, rollback penalty=0.15.
# Отказ при overlap <0.60; неоднозначность словаря разрешает путь, а не ослабление порога.
# Окно B2 и дети K имеют строгий общий лимит 3000/450, включая префикс и вводную.

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from eval.chunking_ab.chunkers import SUBITEM_RE, parse_units, split_sentences
from eval.chunking_ab.norms import Norm
from eval.chunking_ab.run import words

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
            else "below_threshold"
            if score < threshold
            else None
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


def anchor_span(n: Norm, body: str) -> tuple[int, int]:
    token_re = re.compile(r"[а-яёa-z0-9]+")
    target = token_re.findall(body.lower())
    flat, owners = [], []
    for i, p in enumerate(n.paragraphs):
        ts = token_re.findall(p.lower())
        flat.extend(ts)
        owners.extend([i] * len(ts))
    if target:
        for i in range(len(flat) - len(target) + 1):
            if flat[i : i + len(target)] == target:
                return owners[i], owners[i + len(target) - 1] + 1
    w = words(body)
    anchor = max(
        range(len(n.paragraphs)), key=lambda i: len(words(n.paragraphs[i]) & w)
    )
    return anchor, anchor + 1


def bounded_pieces(text: str, limit: int) -> list[str]:
    """Предложения, затем слова; слишком длинный токен — явный крайний fallback."""
    out = []
    for sentence in split_sentences(text, limit):
        if len(sentence) <= limit:
            out.append(sentence)
            continue
        buf = ""
        for token in sentence.split():
            if buf and len(buf) + 1 + len(token) > limit:
                out.append(buf)
                buf = ""
            while len(token) > limit:
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(token[:limit])
                token = token[limit:]
            buf = (buf + " " + token).strip()
        if buf:
            out.append(buf)
    return out


def parent_window(n: Norm, body: str, limit: int = 3000) -> str:
    if len(n.text) <= limit:
        return n.text
    lo, hi = anchor_span(n, body)
    anchor = "\n".join(n.paragraphs[lo:hi])
    if len(anchor) > limit:
        # Целые абзацы невозможны: выбираем фрагмент с максимальным overlap.
        pieces = bounded_pieces(anchor, limit)
        w = words(body)
        return max(pieces, key=lambda p: len(words(p) & w))
    while True:
        candidates = []
        if lo:
            candidates.append((lo - 1, hi))
        if hi < len(n.paragraphs):
            candidates.append((lo, hi + 1))
        candidates = [
            (a, b) for a, b in candidates if len("\n".join(n.paragraphs[a:b])) <= limit
        ]
        if not candidates:
            break
        # Добавляем ближайшую короткую сторону, при равенстве левую.
        lo, hi = min(
            candidates, key=lambda ab: len("\n".join(n.paragraphs[ab[0] : ab[1]]))
        )
    return "\n".join(n.paragraphs[lo:hi])


@dataclass(frozen=True)
class Child:
    id: str
    parent_id: str
    text: str
    body: str
    word_fallback: bool
    intro_shortened: bool


def compact(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_children(
    doc: str, title: str, paragraphs: list[str], limit: int = 450
) -> list[Child]:
    result = []
    for ordinal, u in enumerate(parse_units(paragraphs)):
        n = Norm(doc, u.kind, u.number, tuple(u.paragraphs), ordinal)
        address = (
            f"ст. {u.number}"
            if u.kind == "article"
            else f"п. {u.number}"
            if u.kind == "item"
            else "вводная"
        )
        prefix = " › ".join(
            x for x in (compact(title, 80), compact(u.heading, 70), address) if x
        )
        room = limit - len(prefix) - 1
        intro = ""
        for p in u.paragraphs:
            is_sub = bool(SUBITEM_RE.match(p))
            repeated = compact(intro, min(120, room // 3)) if is_sub and intro else ""
            capacity = room - (len(repeated) + 1 if repeated else 0)
            pieces = bounded_pieces(p, capacity)
            fallback = any(len(s) > capacity for s in split_sentences(p, capacity))
            for piece in pieces:
                body = f"{repeated}\n{piece}" if repeated else piece
                result.append(
                    Child(
                        f"{n.id}::child:{len(result)}",
                        n.id,
                        f"{prefix}\n{body}",
                        body,
                        fallback,
                        bool(repeated and repeated != intro),
                    )
                )
            if not is_sub:
                intro = p if p.rstrip().endswith(":") else ""
    return result
