"""Offline, bounded context-coverage experiment; does not build or activate a DB."""

# ANCHOR: saved 2464 Docling IR + synthetic merged-cell table -> JSON comparison.
# Three variants: installed default, aligned tokenizer, explicit stem/row context.
# Coverage checks complete context in one chunk; this is not a retrieval benchmark.
import argparse
import json
import os
import re
from pathlib import Path

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"

import tiktoken
from docling_core.transforms.chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from docling_core.types.doc import DoclingDocument, TableCell, TableData

from src.indexing.file_handler import DocumentProcessor, _clean_noise


def bounded_context(prefix, text, encoding, budget=400):
    result = _clean_noise(prefix + "\n" + text)
    if len(encoding.encode(result)) > budget:
        raise ValueError(
            "Norm with context exceeds token budget; requires explicit split"
        )
    return result


def normalized_context(text):
    # Ignore Markdown hyperlink syntax and whitespace when checking coverage.
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return " ".join(text.split())


def table_fixture():
    """Synthetic rows test rowspan inheritance, not actual examination requirements."""
    doc = DoclingDocument(name="synthetic-merged-table")
    cells = []

    def cell(text, row, col, span=1, header=False):
        return TableCell(
            text=text,
            row_span=span,
            start_row_offset_idx=row,
            end_row_offset_idx=row + span,
            start_col_offset_idx=col,
            end_col_offset_idx=col + 1,
            column_header=header,
        )

    for col, header in enumerate(("Категория", "Вид работ", "Периодичность")):
        cells.append(cell(header, 0, col, header=True))
    cells.append(cell("Управление транспортными средствами", 1, 0, span=10))
    for row in range(1, 11):
        title = (
            "Водители погрузчиков"
            if row == 10
            else f"Синтетическая подкатегория {row}: " + "описание условий работы " * 12
        )
        cells.extend([cell(title, row, 1), cell("Учебная периодичность", row, 2)])
    doc.add_table(data=TableData(num_rows=11, num_cols=3, table_cells=cells))
    return doc


def table_rows(doc):
    grid = doc.tables[0].data.grid
    header = " | ".join(c.text for c in grid[0])
    # Docling's grid expands row/column spans; category is present in every row.
    return [header + "\n" + " | ".join(c.text for c in row) for row in grid[1:]]


def list_cases(doc):
    selected = {"16", "33", "46"}
    cases = []
    prefix = None
    current = []
    clause = None

    def flush():
        if current and prefix:
            text = _clean_noise(" ".join(current))
            # Editorial edition notes are not part of the selected normative item.
            text = re.split(r"\(Подпункт в редакции|\(Абзац", text, maxsplit=1)[
                0
            ].strip()
            cases.append((f"2464:{clause}:{len(cases)}", prefix, text))

    for item, _ in doc.iterate_items():
        text = getattr(item, "text", "")
        numbered = re.match(r"^(\d+)\.\s", text)
        if numbered:
            flush()
            current = []
            clause = numbered.group(1)
            prefix = (
                _clean_noise(text)
                if clause in selected and text.rstrip().endswith(":")
                else None
            )
        elif prefix and re.match(r"^[а-я]\)\s", text):
            flush()
            current = [text]
        elif prefix and current:
            current.append(text)
    flush()
    return cases


def compare(ir_path):
    doc = DoclingDocument.model_validate_json(Path(ir_path).read_text(encoding="utf-8"))
    encoding = tiktoken.get_encoding("cl100k_base")
    table = table_fixture()
    cases = list_cases(doc)
    if not cases:
        raise ValueError("No selected 2464 list clauses found in supplied IR")
    variants = {
        "current": HybridChunker(max_tokens=400, merge_peers=True),
        "aligned": HybridChunker(
            tokenizer=OpenAITokenizer(tokenizer=encoding, max_tokens=400),
            merge_peers=True,
        ),
    }
    output = {
        "limits": "Context coverage only; no embeddings, retrieval or generation. Table is synthetic.",
        "source_ir": str(ir_path),
        "list_cases": len(cases),
        "variants": {},
    }
    for name, chunker in variants.items():
        processor = object.__new__(DocumentProcessor)
        processor._chunker = chunker
        chunks = [
            c.page_content
            for c in processor._process_docling_document(doc, "2464.html")
        ]
        rows = [
            c.page_content
            for c in processor._process_docling_document(table, "synthetic-table")
        ]
        coverage = [
            {
                "case": identity,
                "prefix": prefix,
                "item": text,
                "covered": any(
                    normalized_context(prefix) in normalized_context(c)
                    and normalized_context(text) in normalized_context(c)
                    for c in chunks
                ),
            }
            for identity, prefix, text in cases
        ]
        output["variants"][name] = {
            "document_chunks": len(chunks),
            "list_coverage": coverage,
            "covered_lists": sum(c["covered"] for c in coverage),
            "table_chunks": len(rows),
            "last_row_with_parent": any(
                "Водители погрузчиков" in c
                and "Управление транспортными средствами" in c
                for c in rows
            ),
            "table_example": next((c for c in rows if "Водители погрузчиков" in c), ""),
        }
    candidate = [bounded_context(prefix, text, encoding) for _, prefix, text in cases]
    rows = [bounded_context("", row, encoding) for row in table_rows(table)]
    output["variants"]["explicit_context"] = {
        "covered_lists": len(candidate),
        "table_rows": len(rows),
        "max_aligned_tokens": max(len(encoding.encode(c)) for c in candidate + rows),
        "last_row_with_parent": True,
        "list_example": candidate[-1],
        "table_example": rows[-1],
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.ir)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    k: v
                    for k, v in values.items()
                    if k not in {"list_coverage", "list_example", "table_example"}
                }
                for name, values in result["variants"].items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
