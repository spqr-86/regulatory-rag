"""Диагностика _process_docling_document на 2464.pdf.

Цель: понять, где теряется пункт «Повторный инструктаж … не реже 1 раза в 6 месяцев».
Логирует каждый item: page, bbox.height, item_type, text[:80], решение (kept/dropped+reason).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docling.document_converter import DocumentConverter
from docling_core.types.doc import ListItem, SectionHeaderItem, TextItem

from src.indexing.file_handler import BLACKLIST_PHRASES, MAX_CHUNK_SIZE, MIN_BBOX_HEIGHT

TARGET_SUBSTRINGS = [
    "Повторный инструктаж",
    "не реже 1 раза в 6 месяцев",
    "не реже одного раза в 6 месяцев",
    "не реже одного раза в шесть месяцев",
]
PDF = Path(__file__).resolve().parents[1] / "source_docs" / "2464.pdf"


def item_type(it):
    if isinstance(it, SectionHeaderItem):
        return "Header"
    if isinstance(it, ListItem):
        return "ListItem"
    if isinstance(it, TextItem):
        return "TextItem"
    return type(it).__name__


def main():
    print(f"MIN_BBOX_HEIGHT={MIN_BBOX_HEIGHT}  MAX_CHUNK_SIZE={MAX_CHUNK_SIZE}")
    print(f"PDF: {PDF}")
    print(f"Targets: {TARGET_SUBSTRINGS}\n")

    conv = DocumentConverter()
    res = conv.convert(str(PDF))
    doc = res.document

    items = list(doc.texts) if hasattr(doc, "texts") else []
    print(f"Total items: {len(items)}\n")

    target_hits = []
    drop_stats = {"empty": 0, "blacklist": 0, "bbox_lt_min": 0, "kept": 0}

    current_page = None
    current_len = 0
    chunk_breaks = 0

    for i, item in enumerate(items):
        text = (item.text or "").strip()
        is_target = any(t in text for t in TARGET_SUBSTRINGS)

        if not text:
            drop_stats["empty"] += 1
            continue

        if any(p in text for p in BLACKLIST_PHRASES):
            drop_stats["blacklist"] += 1
            if is_target:
                print(f"[{i}] *** TARGET DROPPED (blacklist) ***  {text[:120]}")
            continue

        bbox_h = None
        page = None
        if hasattr(item, "prov") and item.prov:
            prov = item.prov[0]
            if hasattr(prov, "bbox") and prov.bbox:
                bt = (
                    prov.bbox.as_tuple()
                    if hasattr(prov.bbox, "as_tuple")
                    else prov.bbox
                )
                bbox_h = abs(bt[3] - bt[1])
            if hasattr(prov, "page_no"):
                page = prov.page_no

        decision = "kept"
        reason = ""
        if bbox_h is not None and bbox_h < MIN_BBOX_HEIGHT:
            decision = "DROPPED"
            reason = f"bbox_h={bbox_h:.2f} < {MIN_BBOX_HEIGHT}"
            drop_stats["bbox_lt_min"] += 1
        else:
            drop_stats["kept"] += 1

        page_flip = (
            current_page is not None and page is not None and page != current_page
        )
        if page_flip:
            chunk_breaks += 1
        if decision == "kept":
            if current_len + len(text) > MAX_CHUNK_SIZE:
                chunk_breaks += 1
                current_len = 0
            current_len += len(text)
            current_page = page

        if is_target:
            target_hits.append(i)
            print(
                f"[{i}] *** TARGET *** type={item_type(item)} page={page} "
                f"bbox_h={bbox_h} decision={decision} {reason}"
            )
            print(f"      text={text[:200]}")

    print("\n--- Stats ---")
    print(drop_stats)
    print(f"chunk_breaks ~= {chunk_breaks}")
    print(f"target hits: {len(target_hits)} at indices {target_hits}")


if __name__ == "__main__":
    main()
