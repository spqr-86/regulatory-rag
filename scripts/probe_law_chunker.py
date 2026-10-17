"""Пробная нарезка закона/правил из текста (дизайн: docs/design/2026-10-17-law-chunking-design.md).

Ничего не пишет в индекс. Печатает чанки выбранного пункта/статьи и проверяет тест-кейсы
G005 (п. 16 Правил 2464) и G047 (ст. 229 ТК).

    .venv/bin/python scripts/probe_law_chunker.py 2464 --item 16
    .venv/bin/python scripts/probe_law_chunker.py tk --article 229
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MAX_CHARS = 1800
ITEM_RE = re.compile(r"^(\d+(?:\.\d+)*)\\?\.\s+\S")
SUBITEM_RE = re.compile(r"^([а-яё]|\d+)\)\s")
ARTICLE_RE = re.compile(r"^\s*Статья\s+(\d+(?:\.\d+)*)\.\s+(.*)$")
SENT_END_RE = re.compile(r"(?<=[.;:])\s+(?=[А-ЯЁ«\"(])")


def split_items(md: str) -> dict[str, list[str]]:
    """Пункт «N.» → его строки (вводная, подпункты, абзацы-продолжения)."""
    items: dict[str, list[str]] = {}
    cur: str | None = None
    for line in md.splitlines():
        line = line.strip()
        if not line:
            continue
        m = ITEM_RE.match(line)
        if m:
            cur = m.group(1)
            items[cur] = [line.replace("\\.", ".", 1)]
        elif cur is not None and not line.startswith("#"):
            items[cur].append(line)
    return items


def chunk_item(lines: list[str], prefix: str) -> list[str]:
    """Пункт целиком; не влезает — по подпунктам, каждый чанк с вводной."""
    whole = "\n".join(lines)
    if len(whole) + len(prefix) <= MAX_CHARS:
        return [f"{prefix}\n{whole}"]
    intro = lines[0]
    chunks: list[str] = []
    cur = [intro]
    for line in lines[1:]:
        if SUBITEM_RE.match(line) and len("\n".join(cur + [line])) > MAX_CHARS:
            chunks.append(f"{prefix}\n" + "\n".join(cur))
            cur = [intro]
        cur.append(line)
    chunks.append(f"{prefix}\n" + "\n".join(cur))
    return chunks


def chunk_text_by_sentences(text: str, prefix: str) -> list[str]:
    """Статья без нумерованных частей (ТК): режем на границе предложения, не посреди."""
    sents = SENT_END_RE.split(" ".join(text.split()))
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) + len(prefix) > MAX_CHARS:
            chunks.append(f"{prefix}\n{cur}")
            cur = ""
        cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(f"{prefix}\n{cur}")
    return chunks


def tk_article_text(pdf: Path, number: str) -> tuple[str, str]:
    import pymupdf

    lines: list[str] = []
    for page in pymupdf.open(pdf):
        lines.extend(page.get_text().splitlines())
    found = False
    title = ""
    body: list[str] = []
    for line in lines:
        m = ARTICLE_RE.match(line)
        if m and m.group(1) == number and not found:
            found, title = True, m.group(2).strip()
        elif found:
            if ARTICLE_RE.match(line):
                break
            body.append(line)
    if not found:
        sys.exit(f"статья {number} не найдена в {pdf}")
    return title, "\n".join(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc", choices=["2464", "tk"])
    ap.add_argument("--item")
    ap.add_argument("--article")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    if args.doc == "2464":
        items = split_items((root / "source_docs/2464.md").read_text(encoding="utf-8"))
        n = args.item or "16"
        chunks = chunk_item(items[n], f"Постановление Правительства РФ № 2464, п. {n}")
        intro = items[n][0].split(".", 1)[1].strip()[:40]
        ok = all(intro in ch for ch in chunks) and all(
            any(f"\n{c})" in ch for ch in chunks) for c in "абвгдежз"
        )
        verdict = "G005: перечень а)–з) с вводной — " + ("OK" if ok else "FAIL")
    else:
        n = args.article or "229"
        title, body = tk_article_text(root / "source_docs/тк рф.pdf", n)
        chunks = chunk_text_by_sentences(
            body, f"Трудовой кодекс РФ, Статья {n}. {title}"
        )
        ok = any("Комиссию возглавляет" in ch for ch in chunks) and all(
            ch.rstrip()[-1] in ".;:" for ch in chunks
        )
        verdict = "G047: ст. 229, фраза в чанке, без обрывов — " + (
            "OK" if ok else "FAIL"
        )
    for i, ch in enumerate(chunks):
        print(f"--- chunk {i} ({len(ch)} зн.)\n{ch}\n")
    print(verdict)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
