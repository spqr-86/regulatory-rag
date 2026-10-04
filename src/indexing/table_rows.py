"""Row-level chunks for numbered regulatory tables (e.g. 29н appendices)."""

# ANCHOR: Docling tables + HybridChunker chunks -> chunks with numbered tables
# re-serialized one row per chunk: section heading, parent row ("Пункт 18: ..."),
# then the row cells. Page continuations of the same width are merged first, so a
# sub-row on the next page still finds its parent. Rows without a number in the
# first column are kept as plain row blocks, so no table text is lost; short ones
# join a neighbour row (lowercase start or footnote -> previous row, else -> next). Tables
# without numbered rows keep their original chunks. Budget: cl100k tokens per chunk
# (the embedding model's tokenizer).
# Evidence: eval/runs/table_rows_29n_hybrid_2026-10-04, golden90_29n_regression_2026-10-04.
from __future__ import annotations

import json
import re
from typing import Any, List

import tiktoken
from langchain_core.documents import Document

ROW_ID = re.compile(r"^\d+(\.\d+)*\.?$")
ROW_BUDGET = 400
# Unnumbered runs shorter than this (group titles, words torn by a page break)
# join a neighbouring row instead of becoming a near-empty chunk.
LOOSE_MIN = 20
NO_HEADING = "Document start"


def _page(table: Any) -> int | None:
    return table.prov[0].page_no if table.prov else None


def merge_page_continuations(tables: List[Any]) -> List[List[Any]]:
    """Group consecutive tables that continue on the next page with the same width."""
    groups: List[List[Any]] = []
    for table in tables:
        prev = groups[-1][-1] if groups else None
        if (
            prev is not None
            and table.data.num_cols == prev.data.num_cols
            and _page(table) is not None
            and _page(prev) is not None
            and _page(table) - _page(prev) == 1
        ):
            groups[-1].append(table)
        else:
            groups.append([table])
    return groups


def _rows(group: List[Any]) -> List[List[str]]:
    """Non-empty grid rows in order; Docling repeats some rows, keep the first copy."""
    rows, seen = [], set()
    for table in group:
        for row in table.data.grid:
            texts = [c.text.strip() for c in row]
            key = tuple(texts)
            if not any(texts) or key in seen:
                continue
            seen.add(key)
            rows.append(texts)
    return rows


def _is_numbered(texts: List[str]) -> bool:
    return bool(ROW_ID.match(texts[0]))


def _parent_of(row_id: str) -> str | None:
    return row_id.rsplit(".", 1)[0] if "." in row_id else None


def _tokens(encoding: Any, text: str) -> int:
    # encode_ordinary: document text may contain special-token strings.
    return len(encoding.encode_ordinary(text))


def _split_words(line: str, encoding: Any, budget: int) -> List[str]:
    """Split one oversized cell on word boundaries; never truncate."""
    if _tokens(encoding, line) <= budget:
        return [line]
    pieces, current = [], []
    for word in line.split():
        if current and _tokens(encoding, " ".join(current + [word])) > budget:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    return pieces + [" ".join(current)]


def _pack(head: List[str], body: List[str], encoding: Any, budget: int) -> List[str]:
    """Lines -> texts within budget; every text repeats the head lines."""
    head_tokens = _tokens(encoding, "\n".join(head))
    lines = []
    for line in body:
        lines += _split_words(line, encoding, budget - head_tokens - 8)
    parts, current = [], list(head)
    for line in lines:
        if (
            len(current) > len(head)
            and _tokens(encoding, "\n".join(current + [line])) > budget
        ):
            parts.append(current)
            current = list(head)
        current.append(line)
    parts.append(current)
    return ["\n".join(p) for p in parts]


def _group_units(group: List[Any], encoding: Any, budget: int) -> List[tuple]:
    """(row_id | None, text) units for one merged table, in table order."""
    units: List[tuple] = []
    by_id: dict[str, List[str]] = {}
    loose: List[str] = []
    pending: List[str] = []  # short titles waiting for the next row

    def to_previous(text):
        row_id, prev = units[-1]
        units[-1] = (row_id, f"{prev}\n{text}")

    def flush_loose():
        if not loose:
            return
        text = "\n".join(loose)
        if _tokens(encoding, text) < LOOSE_MIN:
            # Lowercase start or no letters (footnote "<2>:") continues the previous row.
            first = next((ch for ch in text if ch.isalpha()), "")
            if (not first or first.islower()) and units and not pending:
                to_previous(text)
            else:
                pending.append(text)
        else:
            body = pending + loose
            pending.clear()
            units.extend((None, t) for t in _pack([], body, encoding, budget))
        loose.clear()

    for texts in _rows(group):
        if not _is_numbered(texts):
            # Section titles span all columns: keep each distinct cell once.
            loose.append(" | ".join(dict.fromkeys(t for t in texts if t)))
            continue
        flush_loose()
        row_id = texts[0].rstrip(".")
        by_id.setdefault(row_id, texts)
        parent_id = _parent_of(row_id)
        parent = by_id.get(parent_id) if parent_id else None
        head = pending[:]
        pending.clear()
        if parent and len(parent) > 1:
            head.append(f"Пункт {parent_id}: {parent[1]}")
        head.append(f"Колонка 1: {texts[0]}")
        body = [t for t in texts[1:] if t]
        if body:
            head.append(body.pop(0))
        units.extend((row_id, t) for t in _pack(head, body, encoding, budget))
    flush_loose()
    if pending:
        tail = "\n".join(pending)
        if units:
            to_previous(tail)
        else:
            units.append((None, tail))
    return units


def explicit_table_chunks(
    tables: Any, chunks: List[Document], budget: int = ROW_BUDGET
) -> List[Document]:
    """Replace chunks of numbered tables with row-level chunks, in place."""
    if not isinstance(tables, list) or not tables:
        return chunks
    encoding = tiktoken.get_encoding("cl100k_base")
    group_of: dict[str, int] = {}
    replacement: dict[int, List[tuple]] = {}
    for gi, group in enumerate(merge_page_continuations(tables)):
        if any(_is_numbered(r) for r in _rows(group)):
            replacement[gi] = _group_units(group, encoding, budget)
            group_of.update({t.self_ref: gi for t in group})
    if not replacement:
        return chunks

    out: List[Document] = []
    emitted: set[int] = set()
    for chunk in chunks:
        refs = json.loads(chunk.metadata.get("doc_item_refs") or "[]")
        groups = {group_of.get(r) for r in refs}
        # Only chunks made purely of one numbered table are replaced.
        if not refs or len(groups) != 1 or None in groups:
            out.append(chunk)
            continue
        gi = groups.pop()
        if gi in emitted:
            continue
        emitted.add(gi)
        section = chunk.metadata.get("parent_section", "")
        heading = section if section and section != NO_HEADING else ""
        base = {
            k: v
            for k, v in chunk.metadata.items()
            if k not in {"bbox", "doc_item_refs"}
        }
        base["element_type"] = "table"
        for row_id, text in replacement[gi]:
            meta = dict(base)
            if row_id is not None:
                meta["table_row_id"] = row_id
            out.append(
                Document(page_content=f"{heading}\n{text}".strip(), metadata=meta)
            )
    return out
