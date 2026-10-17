"""Нарезки текста для эксперимента A/B (docs/evaluation/experiments/2026-10-17-law-chunking-ab.md).

Все функции чистые: на вход текст/абзацы, на выход список строк-чанков. Структурный
парсер — прототип будущего `src/indexing/law_chunker.py`, в индекс не пишет.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*(?:-\d+)?)\.\s*(.*)$")
# «16.», «16\.», «2(1).», «3.1.» — номер пункта в начале абзаца
ITEM_RE = re.compile(r"^(\d+(?:\(\d+\))?(?:\.\d+)*)\\?\.\s+\S")
SUBITEM_RE = re.compile(r"^(?:[а-яё]|\d+)\)\s")
HEADING_RE = re.compile(r"^(?:Раздел|РАЗДЕЛ|Глава|ГЛАВА|[IVXLC]+\.)\s+\S.{0,150}$")
SENT_SPLIT_RE = re.compile(r"(?<=[.;:!?])\s+(?=[А-ЯЁ«\"(0-9])")
TERMINAL = ".;:!?"
BARE_NUMBER_RE = re.compile(r"^\d+(?:\(\d+\))?(?:\.\d+)*\.$")
NOISE_RE = re.compile(r"^(?:\d{1,4}|КонсультантПлюс.*|Документ предоставлен.*)$")


def _unescape(line: str) -> str:
    return line.replace("\\.", ".").replace("\\(", "(").replace("\\)", ")")


def _is_marker(line: str) -> bool:
    return bool(
        ARTICLE_RE.match(line)
        or ITEM_RE.match(line)
        or SUBITEM_RE.match(line)
        or HEADING_RE.match(line)
    )


def paragraphs_from_text(text: str, wrapped: bool) -> list[str]:
    """Логические абзацы. wrapped=True — PDF: строки склеиваются, пока абзац не кончился."""
    lines = [_unescape(ln).strip() for ln in text.splitlines()]
    lines = [ln.lstrip("#").strip() if ln.startswith("#") else ln for ln in lines]
    lines = [ln for ln in lines if ln and not NOISE_RE.match(ln)]
    if not wrapped:
        return lines
    paras: list[str] = []
    for ln in lines:
        if (
            paras
            and not _is_marker(ln)
            and paras[-1][-1] not in TERMINAL
            and not ln[0].isupper()
        ):
            paras[-1] += " " + ln
        elif (
            paras
            and not _is_marker(ln)
            and paras[-1][-1] not in TERMINAL
            and not HEADING_RE.match(paras[-1])
        ):
            paras[-1] += " " + ln
        else:
            paras.append(ln)
    return paras


def split_sentences(text: str, limit: int) -> list[str]:
    """Режет по границе предложения, пакуя до limit знаков; одно предложение не режет."""
    raw = SENT_SPLIT_RE.split(" ".join(text.split()))
    sents: list[str] = []
    for r in raw:  # «44.» — номер пункта, а не конец предложения
        if sents and BARE_NUMBER_RE.match(sents[-1]):
            sents[-1] = f"{sents[-1]} {r}"
        else:
            sents.append(r)
    out: list[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + 1 + len(s) > limit:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


@dataclass
class Unit:
    kind: str  # article | item | preamble
    number: str
    title: str
    heading: str
    paragraphs: list[str] = field(default_factory=list)


def parse_units(paras: list[str]) -> list[Unit]:
    """Разбор на единицы: статья (если документ в статьях) либо пункт верхнего уровня."""
    n_articles = sum(1 for p in paras if ARTICLE_RE.match(p))
    article_mode = n_articles >= 2
    units: list[Unit] = [Unit("preamble", "", "", "")]
    heading = ""
    for p in paras:
        if HEADING_RE.match(p) and not ITEM_RE.match(p):
            heading = p[:100]
            units[-1].paragraphs.append(p)
            continue
        m_art = ARTICLE_RE.match(p)
        m_item = ITEM_RE.match(p)
        if article_mode and m_art:
            units.append(Unit("article", m_art.group(1), m_art.group(2), heading, [p]))
        elif not article_mode and m_item:
            units.append(Unit("item", m_item.group(1), "", heading, [p]))
        else:
            units[-1].paragraphs.append(p)
    return [u for u in units if u.paragraphs]


def _prefix(doc: str, unit: Unit, part: str = "") -> str:
    bits = [doc]
    if unit.heading:
        bits.append(unit.heading)
    if unit.kind == "article":
        bits.append(f"Статья {unit.number}. {unit.title}".strip())
    elif unit.kind == "item":
        bits.append(f"п. {unit.number}")
    if part:
        bits.append(part)
    return " › ".join(bits)


def _pack(intro: str | None, paras: list[str], limit: int) -> list[str]:
    """Пакует абзацы в тела чанков ≤ limit; intro (вводная) повторяется в каждом."""
    bodies: list[str] = []
    cur: list[str] = [intro] if intro else []
    base = len(intro) if intro else 0
    size = base
    for p in paras:
        if len(p) + base > limit:
            # абзац сам не влезает — по предложениям
            for s in split_sentences(p, max(limit - base, 200)):
                if size + 1 + len(s) > limit and len(cur) > (1 if intro else 0):
                    bodies.append("\n".join(cur))
                    cur, size = ([intro] if intro else []), base
                cur.append(s)
                size += 1 + len(s)
            continue
        if size + 1 + len(p) > limit and len(cur) > (1 if intro else 0):
            bodies.append("\n".join(cur))
            cur, size = ([intro] if intro else []), base
        cur.append(p)
        size += 1 + len(p)
    if len(cur) > (1 if intro else 0) or not bodies:
        bodies.append("\n".join(cur))
    return bodies


def _split_unit_body(unit: Unit, limit: int) -> list[tuple[str, str]]:
    """Тела чанков единицы как (часть, тело)."""
    paras = unit.paragraphs
    head_len = 0
    first_is_head = unit.kind == "article" and ARTICLE_RE.match(paras[0])
    content = paras[1:] if first_is_head else paras
    if not content:
        return [("", paras[0])]
    whole = "\n".join(content)
    if len(whole) + head_len <= limit:
        return [("", whole)]
    # статья: делим по частям «N.» ; пункт: сам пункт — одна часть
    parts: list[tuple[str, list[str]]] = []
    if unit.kind == "article":
        for p in content:
            m = ITEM_RE.match(p)
            if m or not parts:
                parts.append((m.group(1) if m else "", [p]))
            else:
                parts[-1][1].append(p)
    else:
        parts = [(unit.number, content)]
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    buf_nums: list[str] = []

    def flush() -> None:
        nonlocal buf, buf_nums
        if buf:
            label = ""
            nums = [n for n in buf_nums if n]
            if nums:
                label = (
                    f"часть {nums[0]}"
                    if len(nums) == 1
                    else f"части {nums[0]}–{nums[-1]}"
                )
            out.append((label, "\n".join(buf)))
        buf, buf_nums = [], []

    for num, plines in parts:
        ptext = "\n".join(plines)
        if len(ptext) > limit:
            flush()
            intro = (
                plines[0] if len(plines) > 1 and SUBITEM_RE.match(plines[1]) else None
            )
            if intro:
                for body in _pack(intro, plines[1:], limit):
                    out.append(
                        (f"часть {num}" if num and unit.kind == "article" else "", body)
                    )
            else:
                for body in _pack(None, plines, limit):
                    out.append(
                        (f"часть {num}" if num and unit.kind == "article" else "", body)
                    )
            continue
        if buf and len("\n".join(buf)) + 1 + len(ptext) > limit:
            flush()
        buf.append(ptext)
        buf_nums.append(num)
    flush()
    return out


def chunk_structural_parts(
    paras: list[str], doc: str, max_chars: int
) -> list[tuple[str, str]]:
    """Чанки как (префикс, тело); тело — текст нормы без префикса."""
    chunks: list[tuple[str, str]] = []
    for unit in parse_units(paras):
        if unit.kind == "preamble":
            chunks += [(doc, body) for body in _pack(None, unit.paragraphs, max_chars)]
            continue
        for part, body in _split_unit_body(unit, max_chars):
            chunks.append((_prefix(doc, unit, part), body))
    return chunks


def chunk_structural(
    paras: list[str], doc: str, max_chars: int, prefix: bool = True
) -> list[str]:
    parts = chunk_structural_parts(paras, doc, max_chars)
    return [f"{h}\n{b}" if prefix else b for h, b in parts]


def chunk_window(text: str, size: int = 350, stride: int = 300) -> list[str]:
    words = text.split()
    out = []
    for start in range(0, max(len(words) - size + stride, 1), stride):
        piece = words[start : start + size]
        if piece:
            out.append(" ".join(piece))
        if start + size >= len(words):
            break
    return out


def chunk_paragraphs(
    paras: list[str], min_chars: int = 200, max_chars: int = 3000
) -> list[str]:
    out: list[str] = []
    cur = ""
    for p in paras:
        cur = f"{cur}\n{p}".strip() if cur else p
        if len(cur) >= min_chars:
            out.extend(
                split_sentences(cur, max_chars) if len(cur) > max_chars else [cur]
            )
            cur = ""
    if cur:
        out.append(cur)
    return out
