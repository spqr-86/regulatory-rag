"""Restore context that chunking cut off from list items and table sub-rows.

HybridChunker splits NPA enumerations item by item, so «б) изменениями
должностных обязанностей…» lands in a chunk without its lead-in «Внеплановый
инструктаж проводится в случаях, обусловленных:». Same with hierarchical
tables in 29н: row «18.1 Категории A, B, BE… 1 раз в 2 года» loses its parent
row «18 Управление наземными транспортными средствами». Retrieval then cannot
match the question to the chunk, and the generator cannot tell which list an
item belongs to (issue #64).

The fix prepends the missing line to the chunk text. Chunk boundaries do not
change, so chunk ids and retrieval ground truth stay valid.
"""

from __future__ import annotations

import re

# «а) …» or Docling's enumerator artifact «9. а) …».
_ITEM_RE = re.compile(r"^(?:\d+\.\s+)?[а-яё]\)\s")
# Lead-in longer than this would crowd the chunk out of the reranker window.
MAX_STEM_CHARS = 400
# Table cells serialised by Docling as «<row>, <col> = <value>».
_PARENT_ROW_RE = re.compile(
    r"(?<![\d.])(\d+), (?:1|<\d+>:) = (.+?)(?=\. \d+(?:\.\d+)?, |\.?\s*$)"
)
_SUBROW_RE = re.compile(r"(?<![\d.])(\d+)\.\d+, ")


def add_structural_context(texts: list[str]) -> list[str]:
    """Return chunk texts of one document with list stems / parent rows prepended.

    ``texts`` must be in document order; the result has the same length.
    """
    out: list[str] = []
    stem: str | None = None
    item_open = False  # previous chunk ended in the middle of a list item
    row_labels: dict[str, str] = {}

    for text in texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        prefixes: list[str] = []

        if lines and stem and stem not in text:
            if _ITEM_RE.match(lines[0]) or item_open:
                prefixes.append(stem)

        for i, line in enumerate(lines):
            if line.endswith(":"):
                stem = line if len(line) <= MAX_STEM_CHARS else None
            elif _ITEM_RE.match(line) or (i == 0 and item_open):
                continue
            else:
                stem = None
        item_open = bool(stem and lines and not lines[-1].endswith((";", ".", ":")))

        for n in dict.fromkeys(_SUBROW_RE.findall(text)):
            if n in row_labels and not re.search(rf"(?<![\d.]){n}, ", text):
                prefixes.append(f"{n}. {row_labels[n]}")
                break
        for n, label in _PARENT_ROW_RE.findall(text):
            label = label.strip(" .")
            if len(re.findall(r"[А-Яа-яЁё]", label)) >= 3:
                row_labels[n] = label

        out.append("\n".join([*prefixes, text]) if prefixes else text)
    return out
