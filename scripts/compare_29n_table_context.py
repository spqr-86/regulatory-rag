"""Offline 29н table-context check on real Docling IR; does not build or activate a DB."""

# ANCHOR: saved 29н IR -> current chunks vs page-merged, parent-aware row serialization.
# Measures whether each sub-row (N.M) shares a chunk with its parent row N text and
# Column headings are reported per merged table only: Docling marks continuation
# rows as headers, so heading text in chunks is not a reliable signal.
# Structural check, not a retrieval benchmark.
import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"

import tiktoken
from docling_core.types.doc import DoclingDocument

from src.indexing.file_handler import DocumentProcessor

ROW_ID = re.compile(r"^\d+(\.\d+)*\.?$")
PERIODICITY = "периодичность"


def norm(text):
    return " ".join(text.replace("- ", "").split()).lower()


def table_page(table):
    return table.prov[0].page_no if table.prov else None


def merge_page_continuations(doc):
    """Group consecutive tables that continue on the next page with the same width."""
    groups = []
    for table in doc.tables:
        prev = groups[-1][-1] if groups else None
        if (
            prev is not None
            and table.data.num_cols == prev.data.num_cols
            and table_page(table) is not None
            and table_page(prev) is not None
            and table_page(table) - table_page(prev) == 1
        ):
            groups[-1].append(table)
        else:
            groups.append([table])
    return groups


def header_texts(group):
    """Column headings from an early grid row of the first page that names the columns."""
    for row in group[0].data.grid[:3]:
        texts = [c.text.strip() for c in row]
        if any(PERIODICITY in t.lower() for t in texts):
            return texts
    return None


def data_rows(group):
    rows, seen = [], set()
    for table in group:
        for row in table.data.grid:
            texts = [c.text.strip() for c in row]
            if not texts or not ROW_ID.match(texts[0]):
                continue
            key = tuple(texts)
            # Docling repeats some rows in the grid; keep the first copy.
            if key in seen:
                continue
            seen.add(key)
            rows.append(texts)
    return rows


def parent_of(row_id):
    return row_id.rsplit(".", 1)[0] if "." in row_id else None


def split_words(line, encoding, budget):
    """Split one oversized cell on word boundaries; never truncate."""
    if len(encoding.encode(line)) <= budget:
        return [line]
    pieces, current = [], []
    for word in line.split():
        if current and len(encoding.encode(" ".join(current + [word]))) > budget:
            pieces.append(" ".join(current))
            current = []
        current.append(word)
    return pieces + [" ".join(current)]


def explicit_rows(groups, encoding, budget=400):
    units = []
    for group in groups:
        headers = header_texts(group)
        by_id = {}
        for texts in data_rows(group):
            row_id = texts[0].rstrip(".")
            by_id.setdefault(row_id, texts)
            parent = by_id.get(parent_of(row_id))
            cols = headers or [f"Колонка {i + 1}" for i in range(len(texts))]
            lines = [f"Пункт {parent_of(row_id)}: {parent[1]}"] if parent else []
            lines += [f"{h}: {v}" for h, v in zip(cols, texts) if v]
            # Oversized rows split by cell; every part repeats parent and row name.
            head = lines[: 3 if parent else 2]
            parts, current = [], list(head)
            body = []
            head_tokens = len(encoding.encode("\n".join(head)))
            for line in lines[len(head) :]:
                body += split_words(line, encoding, budget - head_tokens - 8)
            for line in body:
                if len(encoding.encode("\n".join(current + [line]))) > budget:
                    parts.append(current)
                    current = list(head)
                current.append(line)
            parts.append(current)
            for part in parts:
                text = "\n".join(part)
                if len(encoding.encode(text)) > budget:
                    raise ValueError(f"row {row_id}: one cell exceeds token budget")
                units.append({"row_id": row_id, "parts": len(parts), "text": text})
    return units


def sub_rows(groups):
    cases = []
    for group in groups:
        by_id = {}
        for texts in data_rows(group):
            row_id = texts[0].rstrip(".")
            by_id.setdefault(row_id, texts)
            parent = by_id.get(parent_of(row_id))
            if parent and parent[1]:
                cases.append({"row_id": row_id, "item": texts[1], "parent": parent[1]})
    return cases


def coverage(cases, chunks):
    located = parent_ok = 0
    missing_parent, not_located = [], []
    for case in cases:
        probe = norm(case["item"])[:60]
        hits = [c for c in chunks if probe in norm(c)]
        if not hits:
            not_located.append(case["row_id"])
            continue
        located += 1
        if any(norm(case["parent"]) in norm(c) for c in hits):
            parent_ok += 1
        else:
            missing_parent.append(case["row_id"])
    return {
        "sub_rows": len(cases),
        "located": located,
        "with_parent": parent_ok,
        "missing_parent": missing_parent,
        "not_located": not_located,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir", type=Path, help="saved Docling IR JSON for 29н")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--structural-context",
        type=Path,
        help="structural_context.py from fix/chunk-structural-context (eeaf9f1)",
    )
    args = parser.parse_args()

    doc = DoclingDocument.model_validate_json(args.ir.read_text(encoding="utf-8"))
    encoding = tiktoken.get_encoding("cl100k_base")
    merged = merge_page_continuations(doc)
    # Ground truth for parentage needs the merged view: a per-page view cannot see
    # a parent row that ended the previous page.
    cases = sub_rows(merged)
    current = [
        c.page_content
        for c in DocumentProcessor()._process_docling_document(doc, "29н.pdf")
    ]
    explicit = explicit_rows(merged, encoding)
    variants = {"current": current}
    if args.structural_context:
        spec = importlib.util.spec_from_file_location("sc", args.structural_context)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        variants["prefix_eeaf9f1"] = module.add_structural_context(current)
    variants["explicit"] = [u["text"] for u in explicit]
    report = {
        "tables": len(doc.tables),
        "merged_groups": len(merged),
        "groups_with_headings": sum(1 for g in merged if header_texts(g)),
        "parent_across_page_break": sum(
            1
            for g in merged
            for i, t in enumerate(g[1:], 1)
            if any(
                parent_of(r[0].rstrip("."))
                not in {x[0].rstrip(".") for x in data_rows([t])}
                and parent_of(r[0].rstrip(".")) is not None
                for r in data_rows([t])[:1]
            )
        ),
        "coverage": {name: coverage(cases, texts) for name, texts in variants.items()},
        "explicit_units": len(explicit),
        "explicit_split_rows": len({u["row_id"] for u in explicit if u["parts"] > 1}),
        "max_explicit_tokens": max(len(encoding.encode(u["text"])) for u in explicit),
        "samples": [u for u in explicit if u["row_id"] in {"18.1", "18.2"}],
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
