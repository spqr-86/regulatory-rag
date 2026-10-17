"""Pure source paragraph parsing into ordered articles or top-level items."""

# ANCHOR: source paragraphs → ordered articles/items; copied from eval/chunking_ab.
# The evaluation prototype remains unchanged for reproducibility.

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
